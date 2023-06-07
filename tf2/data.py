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
"""Data pipeline."""

import functools
from absl import flags
from absl import logging

import data_util
import tensorflow.compat.v2 as tf
# （TFDS）是一个用于访问和管理各种公共数据集的库
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


# 这个输入函数会返回一个数据集，其中包含了经过预处理的图像和独热编码的标签。
def build_input_fn(builder, global_batch_size, topology, is_training):
  """Build input function.

  Args:
    builder: TFDS builder for specified dataset.
    global_batch_size: Global batch size.
    topology: An instance of `tf.tpu.experimental.Topology` or None.
    is_training: Whether to build in training mode.

  Returns:
    A function that accepts a dict of params and returns a tuple of images and
    features, to be used as the input_fn in TPUEstimator.
  """

  def _input_fn(input_context):
    """Inner input function."""
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    # 得到预训练预处理好的图像的函数（只差输入图像数据）
    # 获取预处理函数，一个用于预训练阶段，一个用于微调阶段。
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    # 数据集中的类别数量
    num_classes = builder.info.features['label'].num_classes


    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      # 预训练的训练阶段
      if is_training and FLAGS.train_mode == 'pretrain':
        xs = []
        # 对图像进行拼接以实现增强的目的
        for _ in range(2):  # Two transformations
          xs.append(preprocess_fn_pretrain(image))
        image = tf.concat(xs, -1)
      else:
        # 对于下游微调来说，不进行增强操作，只进行预处理
        image = preprocess_fn_finetune(image)

      label = tf.one_hot(label, num_classes)
      # 返回处理好的图像和标签
      return image, label

    #获取输入上下文（input_context）中的输入管道（并行计算）数量。
    logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
    # 将builder的数据转为tf数据集对象
    dataset = builder.as_dataset(
        split=FLAGS.train_split if is_training else FLAGS.eval_split,
        shuffle_files=is_training,
        as_supervised=True,
        # Passing the input_context to TFDS makes TFDS read different parts
        # of the dataset on different workers. We also adjust the interleave
        # parameters to achieve better performance.
        # 这里定义了TFDS的读取配置。使用了交织读取，
        # 可以在多个文件之间并行地读取数据，这对于在
        # 分布式训练环境中提高数据读取效率非常有用。
        # interleave_cycle_length是文件读取周期的长度，
        # interleave_block_length是每个周期内读取的文件
        # 块数量。
        read_config=tfds.ReadConfig(
            # 并行读取数据时使用的周期长度
            interleave_cycle_length=32,
            # 每个周期内要读取的文件块的数量
            interleave_block_length=1,
            # 输入上下文对象
            input_context=input_context))
    # 是否缓存整个数据集在内存中
    if FLAGS.cache_dataset:
      dataset = dataset.cache()
    if is_training:
      # 通过设置适当的选项，可以根据需求调整数据集的行为和性能。
      options = tf.data.Options()
      # 使数据集在多线程环境中提高数据读取效率，因为不需要维持元素的顺序。
      options.experimental_deterministic = False
      # 允许数据加载和处理的时间有所变化，以适应不同的计算资源或负载情况。
      options.experimental_slack = True
      # 使用上面的配置
      dataset = dataset.with_options(options)
      # 如果 FLAGS.image_size 小于或等于 32，则 buffer_multiplier 
      # 被设置为 50。这意味着数据集的缓冲区大小是批次大小（batch_size）
      # 的 50 倍。较小的图像大小可能需要更大的缓冲区以确保数据加载的效率。
      buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
      dataset = dataset.shuffle(batch_size * buffer_multiplier)
      dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy,
                              topology):
  input_fn = build_input_fn(builder, batch_size, topology, is_training)
  return strategy.distribute_datasets_from_function(input_fn)


# 从命令行获取一个图像信息，返回一个预处理好的图像
def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  # Disable test cropping for small images (e.g. CIFAR)
  if FLAGS.image_size <= 32:
    test_crop = False
  else:
    test_crop = True
  color_jitter_strength = FLAGS.color_jitter_strength if is_pretrain else 0.
  return functools.partial(
      data_util.preprocess_image,
      height=FLAGS.image_size,
      width=FLAGS.image_size,
      is_training=is_training,
      color_jitter_strength=color_jitter_strength,
      test_crop=test_crop)
