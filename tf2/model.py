# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Model specification for SimCLR."""

import math
from absl import flags

import data_util
import lars_optimizer
import resnet
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

# 选择优化器
def build_optimizer(learning_rate):
  """Returns the optimizer."""
  if FLAGS.optimizer == 'momentum':
    return tf.keras.optimizers.SGD(learning_rate, FLAGS.momentum, nesterov=True)
  elif FLAGS.optimizer == 'adam':
    return tf.keras.optimizers.Adam(learning_rate)
  elif FLAGS.optimizer == 'lars':
    return lars_optimizer.LARSOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        weight_decay=FLAGS.weight_decay,
        exclude_from_weight_decay=[
            'batch_normalization', 'bias', 'head_supervised'
        ])
  else:
    raise ValueError('Unknown optimizer {}'.format(FLAGS.optimizer))

# 权重衰减
def add_weight_decay(model, adjust_per_optimizer=True):
  """Compute weight decay from flags."""
  if adjust_per_optimizer and 'lars' in FLAGS.optimizer:
    # Weight decay are taking care of by optimizer for these cases.
    # Except for supervised head, which will be added here.
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_variables
        if 'head_supervised' in v.name and 'bias' not in v.name
    ]
    if l2_losses:
      return FLAGS.weight_decay * tf.add_n(l2_losses)
    else:
      return 0

  # TODO(srbs): Think of a way to avoid name-based filtering here.
  l2_losses = [
      tf.nn.l2_loss(v)
      for v in model.trainable_weights
      if 'batch_normalization' not in v.name
  ]
  loss = FLAGS.weight_decay * tf.add_n(l2_losses)
  return loss


def get_train_steps(num_examples):
  """Determine the number of training steps."""
  return FLAGS.train_steps or (
      num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)

# 给定学习率的预热调度
class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self, base_learning_rate, num_examples, name=None):
    '''
    num_examples:预热时使用的样本数
    '''
    # 继承WarmUpAndCosineDecay的初始化方法
    super(WarmUpAndCosineDecay, self).__init__()
    # 学习率
    self.base_learning_rate = base_learning_rate
    # 样本数
    self.num_examples = num_examples
    # 实例的变量名
    self._name = name

  def __call__(self, step):
    # 
    with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
      # 预热的总步数
      warmup_steps = int(
          round(FLAGS.warmup_epochs * self.num_examples //
                FLAGS.train_batch_size))
      # 线性学习率衰减的学习率
      if FLAGS.learning_rate_scaling == 'linear':
        scaled_lr = self.base_learning_rate * FLAGS.train_batch_size / 256.
      # 根方衰减的学习率
      elif FLAGS.learning_rate_scaling == 'sqrt':
        scaled_lr = self.base_learning_rate * math.sqrt(FLAGS.train_batch_size)
      else:
        raise ValueError('Unknown learning rate scaling {}'.format(
            FLAGS.learning_rate_scaling))
      learning_rate = (
          step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

      # Cosine decay learning rate schedule
      total_steps = get_train_steps(self.num_examples)
      # TODO(srbs): Cache this object.
      cosine_decay = tf.keras.experimental.CosineDecay(
          scaled_lr, total_steps - warmup_steps)
      
      learning_rate = tf.where(step < warmup_steps, learning_rate,
                               cosine_decay(step - warmup_steps))

      return learning_rate

  def get_config(self):
    return {
        'base_learning_rate': self.base_learning_rate,
        'num_examples': self.num_examples,
    }

# 线性层-全连接层
class LinearLayer(tf.keras.layers.Layer):

  def __init__(self,
               num_classes,
               use_bias=True,
               use_bn=False,
               name='linear_layer',
               **kwargs):
    # Note: use_bias is ignored for the dense layer when use_bn=True.
    # However, it is still used for batch norm.
    # 继承父类的 初始化方法
    super(LinearLayer, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.use_bias = use_bias
    self.use_bn = use_bn
    self._name = name
    # 使用残差网络的BatchNorm
    if self.use_bn:
      self.bn_relu = resnet.BatchNormRelu(relu=False, center=use_bias)

  # 创建全连接层
  def build(self, input_shape):
    # TODO(srbs): Add a new SquareDense layer.
    # num_classes是被调用的对象，将它看作一个可以接受变量的函数
    if callable(self.num_classes):
      num_classes = self.num_classes(input_shape)
    # num_classes是变量
    else:
      num_classes = self.num_classes
    # 初始化全连接层
    self.dense = tf.keras.layers.Dense(
        num_classes,# 输出类别
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        # 在不使用batch normaliztion且使用bias时使用bias
        use_bias=self.use_bias and not self.use_bn)
    # 继承父类的build方法
    super(LinearLayer, self).build(input_shape)
  
  # 当调用这个类时，call方法起作用
  def call(self, inputs, training):
    # 只接受2维输入
    assert inputs.shape.ndims == 2, inputs.shape
    # 根据输入确定全连接层
    inputs = self.dense(inputs)
    # 使用batch normalization
    if self.use_bn:
      inputs = self.bn_relu(inputs, training=training)
    return inputs


class ProjectionHead(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    # 获取投影头输出维度
    out_dim = FLAGS.proj_out_dim
    
    self.linear_layers = []
    
    # 投影头的初始化方式

    # 空
    if FLAGS.proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens
    # 线性：只包含一个线性层
    elif FLAGS.proj_head_mode == 'linear':
      self.linear_layers = [
        # 使用本文中的线性层
          LinearLayer(
              num_classes=out_dim, use_bias=False, use_bn=True, name='l_0')
      ]
    # 非线性：包含非线性激活函数
    elif FLAGS.proj_head_mode == 'nonlinear':
      # 根据指定的投影头的层数来添加线性层
      for j in range(FLAGS.num_proj_layers):
        # 除了最后一层；都使用偏置函数
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          self.linear_layers.append(
              LinearLayer(
                  num_classes=lambda input_shape: int(input_shape[-1]),
                  use_bias=True,
                  use_bn=True,
                  name='nl_%d' % j))
        else:
          # for the final layer, neither bias nor relu is used.
          self.linear_layers.append(
              LinearLayer(
                  num_classes=FLAGS.proj_out_dim,
                  use_bias=False,
                  use_bn=True,
                  name='nl_%d' % j))
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    
    # 继承父类的方法，**kwargs传入未命名的参数
    super(ProjectionHead, self).__init__(**kwargs)
  
  def call(self, inputs, training):
    if FLAGS.proj_head_mode == 'none':
      return inputs  # directly use the output hiddens as hiddens
    hiddens_list = [tf.identity(inputs, 'proj_head_input')]
    if FLAGS.proj_head_mode == 'linear':
      assert len(self.linear_layers) == 1, len(self.linear_layers)
      return hiddens_list.append(self.linear_layers[0](hiddens_list[-1],
                                                       training))
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        hiddens = self.linear_layers[j](hiddens_list[-1], training)
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          hiddens = tf.nn.relu(hiddens)
        hiddens_list.append(hiddens)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    # The first element is the output of the projection head.
    # The second element is the input of the finetune head.
    proj_head_output = tf.identity(hiddens_list[-1], 'proj_head_output')
    return proj_head_output, hiddens_list[FLAGS.ft_proj_selector]

# 定义监督头，本质上是全连接层
class SupervisedHead(tf.keras.layers.Layer):

  def __init__(self, num_classes, name='head_supervised', **kwargs):
    # 继承监督头模块
    super(SupervisedHead, self).__init__(name=name, **kwargs)
    # 初始化线性层
    self.linear_layer = LinearLayer(num_classes)

  def call(self, inputs, training):
    # 线性变换
    inputs = self.linear_layer(inputs, training)
    # 标识线性变换后的结果
    inputs = tf.identity(inputs, name='logits_sup')
    return inputs


class Model(tf.keras.models.Model):
  """Resnet model with projection or supervised layer."""

  def __init__(self, num_classes, **kwargs):
    super(Model, self).__init__(**kwargs)
    # 初始化残差网络并传入命令行参数
    self.resnet_model = resnet.resnet(
        resnet_depth=FLAGS.resnet_depth,
        width_multiplier=FLAGS.width_multiplier,
        cifar_stem=FLAGS.image_size <= 32)
    # 初始化投影头
    self._projection_head = ProjectionHead()
    # 在需要微调/预训练线性验证时设置输出层（监督头）
    if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
      self.supervised_head = SupervisedHead(num_classes)

  def __call__(self, inputs, training):
    features = inputs
    # 预训练
    if training and FLAGS.train_mode == 'pretrain':
      # 在预训练时，不能冻结参数层
      if FLAGS.fine_tune_after_block > -1:
        raise ValueError('Does not support layer freezing during pretraining,'
                         'should set fine_tune_after_block<=-1 for safety.')
    # 输入通道第四个维度：通道数为空会引起异常
    if inputs.shape[3] is None:
      raise ValueError('The input channels dimension must be statically known '
                       f'(got input shape {inputs.shape})')
    num_transforms = inputs.shape[3] // 3
    num_transforms = tf.repeat(3, num_transforms)
    # Split channels, and optionally apply extra batched augmentation.
    # 分割通道
    features_list = tf.split(
        features, num_or_size_splits=num_transforms, axis=-1)
    # 预训练使用图像增强
    #通过tf.concat函数将模糊处理后的特征张量列表features_list
    # 沿着轴0进行拼接，得到一个形状为(num_transforms * bsz, h, w, c)
    # 的特征张量features。其中，num_transforms表示特征张量的数量，bsz
    # 是批量大小，h和w是图像的高度和宽度，c是通道数。
    if FLAGS.use_blur and training and FLAGS.train_mode == 'pretrain':
      features_list = data_util.batch_random_blur(features_list,
                                                  FLAGS.image_size,
                                                  FLAGS.image_size)
    features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

    # 在给定输入特征 features 的情况下，通过调用 self.resnet_model 方法
    # 对特征进行前向传播
    hiddens = self.resnet_model(features, training=training)

    # 增加映射头和监督头，并连接前面的隐藏层
    projection_head_outputs, supervised_head_inputs = self._projection_head(
        hiddens, training)

    # 微调只需要返回监督头
    if FLAGS.train_mode == 'finetune':
      supervised_head_outputs = self.supervised_head(supervised_head_inputs,
                                                     training)
      return None, supervised_head_outputs
    # 需要验证的预训练会返回映射头和监督头
    elif FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
      # When performing pretraining and linear evaluation together we do not
      # want information from linear eval flowing back into pretraining network
      # so we put a stop_gradient.
      supervised_head_outputs = self.supervised_head(
          tf.stop_gradient(supervised_head_inputs), training)
      return projection_head_outputs, supervised_head_outputs
    # 不需要验证的预训练不需要监督头
    else:
      return projection_head_outputs, None
