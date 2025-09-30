# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

try:
  import deeplake
except ImportError:  # pragma: no cover - handled at runtime
  deeplake = None

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def downsampling(image, resolution):
  
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  kh, kw = h//resolution[0], w//resolution[1]
  image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), kh, kw, padding='VALID') * kh
  return tf.squeeze(image)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return downsampling(img, [config.data.image_size, config.data.image_size])
    
  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'MNIST':
    dataset_builder = tfds.builder('mnist')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)
      return img

  elif config.data.dataset == 'TINYIMAGENET':
    if getattr(config.data, 'use_deeplake', False):
      if deeplake is None:
        raise ImportError('DeepLake is not installed. Please install deeplake to use the TinyImageNet DeepLake loader.')

      deeplake_uris = {
          'train': getattr(config.data, 'deeplake_train_uri', 'hub://activeloop/tiny-imagenet-train'),
          'validation': getattr(config.data, 'deeplake_validation_uri', 'hub://activeloop/tiny-imagenet-test'),
          'test': getattr(config.data, 'deeplake_test_uri', 'hub://activeloop/tiny-imagenet-test')
      }

      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

      def build_deeplake_dataset(split, max_examples=None, shuffle=True):
        uri = deeplake_uris.get(split, split)
        ds = deeplake.load(uri, read_only=True)

        image_spec = tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8)
        label_spec = tf.TensorSpec(shape=(), dtype=tf.int64)

        def generator():
          count = 0
          for sample in ds:
            if max_examples is not None and count >= max_examples:
              break
            image = sample['image'].numpy()
            if image.ndim == 2:
              image = np.expand_dims(image, -1)
            if image.shape[-1] == 1:
              image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] > 3:
              image = image[..., :3]

            label_tensor = None
            for key in ('labels', 'label', 'class_id', 'targets'):
              if key in sample:
                label_tensor = sample[key].numpy()
                break
            if label_tensor is None:
              label = np.int64(-1)
            else:
              label_np = np.array(label_tensor).reshape(-1)
              label = np.int64(label_np[0])
            yield dict(image=image.astype(np.uint8), label=label)
            count += 1

        def preprocess_sample(d):
          img = resize_op(d['image'])
          if config.data.random_flip and not evaluation and split == 'train':
            img = tf.image.random_flip_left_right(img)
          if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
          return dict(image=img, label=d['label'])

        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1

        ds_tf = tf.data.Dataset.from_generator(generator, output_signature={'image': image_spec, 'label': label_spec})
        ds_tf = ds_tf.with_options(dataset_options)
        ds_tf = ds_tf.repeat(count=num_epochs)

        if shuffle:
          effective_buffer = shuffle_buffer_size
          if max_examples is not None:
            effective_buffer = min(max_examples, shuffle_buffer_size)
          ds_tf = ds_tf.shuffle(effective_buffer, reshuffle_each_iteration=True)

        ds_tf = ds_tf.map(preprocess_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_tf = ds_tf.batch(batch_size, drop_remainder=True)
        return ds_tf.prefetch(prefetch_size)

      train_limit = getattr(config.data, 'deeplake_max_examples', None)
      eval_limit = getattr(config.data, 'deeplake_eval_max_examples', None)
      eval_split = getattr(config.data, 'deeplake_eval_split', 'validation')

      train_ds = build_deeplake_dataset('train', max_examples=train_limit, shuffle=not evaluation)
      eval_ds = build_deeplake_dataset(eval_split, max_examples=eval_limit, shuffle=False)
      return train_ds, eval_ds, None
    else:
      dataset_builder = tfds.builder('tiny_imagenet')
      train_split_name = 'train'
      eval_split_name = 'validation'

      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img
    else:
      def downsampling_mod(image, resolution, img_res=256):
          kh, kw = img_res//resolution[0], img_res//resolution[1]
          image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), kh, kw, padding='VALID') * kh
          return tf.squeeze(image)

      def resize_op(img):
        img = crop_resize(img, 256)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return downsampling_mod(img, [config.data.image_size, config.data.image_size])

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'
   
  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    
    def downsampling_mod(image, resolution, img_res=256):
      kh, kw = img_res//resolution[0], img_res//resolution[1]
      image = tf.nn.avg_pool2d(tf.expand_dims(image, 0), kh, kw, padding='VALID') * kh
      return tf.squeeze(image)
    
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      img = downsampling_mod(img, [config.data.image_size, config.data.image_size])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else: #####
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    #ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
