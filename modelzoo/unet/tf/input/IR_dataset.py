# Copyright 2022 Cerebras Systems.
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

"""
Severstal Dataset class

Owners: {vithu, kamran}@cerebras.net
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf


class IRDataset:
    """
    IR Dataset class
    """

    def __init__(self, params=None):

        self.data_dir = params["train_input"]["dataset_path"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                "The dataset directory `%s` does not exist." % self.data_dir
            )

        self.mixed_precision = params["model"]["mixed_precision"]
        self.image_shape = params["train_input"]["image_shape"]
        self.large_patch_size = params["train_input"]["large_patch_size"]
        self.num_classes = params["train_input"]["num_classes"]

        self.data_format = params["model"]["data_format"]
        self.seed = params["train_input"].get("seed", None)

        self.shuffle_buffer_size = params["train_input"]["shuffle_buffer_size"]

        

    def dataset_fn(
        self, batch_size, augment_data=True, shuffle=True, is_training=True,
    ):
        split = 'train' if is_training else 'val'
        self.IR = sorted(glob(os.path.join(self.data_dir,split,'IR/*.npy')))
        self.label = sorted(glob(os.path.join(self.data_dir,split,'label/*.npy')))
        dataset = tf.data.Dataset.range(len(self.IR))
        
        def _load_npy(idx):
            IR_path = self.IR[idx]
            label_path = self.label[idx]
            IR = np.load(IR_path)
            label = np.load(label_path)
            IR = np.moveaxis(IR,0,2)
            label = np.moveaxis(label,0,2)
            return tf.convert_to_tensor(IR), tf.convert_to_tensor(label)
            
        def _load_data(idx):
            IR, label = tf.py_function(_load_npy, [idx], Tout=[tf.float32,tf.int32])
            IR.set_shape(self.image_shape)
            label.set_shape(self.image_shape[:2])
            return tf.data.Dataset.from_tensor_slices(
                ([IR], [label])
            )

        dataset = dataset.interleave(
          _load_data,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.cache()
            

        if is_training and shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size, self.seed)

        def _resize_augment_images(input_image, mask_image):
            if augment_data:
                horizontal_flip = (
                    tf.random.uniform(shape=(), seed=self.seed) > 0.5
                )
                adjust_brightness = (
                    tf.random.uniform(shape=(), seed=self.seed) > 0.5
                )
                h = (tf.random.uniform(shape=(),
                                       maxval=self.large_patch_size[0]-self.image_shape[0], 
                                       dtype=tf.int32,
                                       seed=self.seed))
                w = (tf.random.uniform(shape=(),
                                       maxval=self.large_patch_size[1]-self.image_shape[1], 
                                       dtype=tf.int32,
                                       seed=self.seed))
                
                input_image = tf.image.crop_to_bounding_box(
                    image=input_image, 
                    offset_height=h, 
                    offset_width=w, 
                    target_height=self.image_shape[0], 
                    target_width=self.image_shape[1]
                )
                
                input_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(input_image),
                    false_fn=lambda: input_image,
                )

                input_image = tf.cond(
                    pred=adjust_brightness,
                    true_fn=lambda: tf.image.adjust_brightness(
                        input_image, delta=0.2
                    ),
                    false_fn=lambda: input_image,
                )

                mask_image = tf.expand_dims(mask_image,-1)
                mask_image = tf.image.crop_to_bounding_box(
                    image=mask_image, 
                    offset_height=h, 
                    offset_width=w, 
                    target_height=self.image_shape[0], 
                    target_width=self.image_shape[1]
                )
                
                mask_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(mask_image),
                    false_fn=lambda: mask_image,
                )

                n_rots = tf.random.uniform(
                    shape=(), dtype=tf.int32, minval=0, maxval=3, seed=self.seed
                )

                if self.image_shape[0] != self.image_shape[1]:
                    n_rots = n_rots * 2

                input_image = tf.image.rot90(input_image, k=n_rots)

                mask_image = tf.image.rot90(mask_image, k=n_rots)

                # input_image = tf.image.resize_with_crop_or_pad(
                #     input_image,
                #     target_height=self.image_shape[0],
                #     target_width=self.image_shape[1],
                # )

                # mask_image = tf.image.resize_with_crop_or_pad(
                #     mask_image,
                #     target_height=self.image_shape[0],
                #     target_width=self.image_shape[1],
                # )

            if self.data_format == "channels_first":
                input_image = tf.transpose(a=input_image, perm=[2, 0, 1])

            reshaped_mask_image = tf.reshape(mask_image, [-1])

            # handle mixed precision for float variables
            # int variables remain untouched
            if self.mixed_precision:
                input_image = tf.cast(input_image, dtype=tf.float16)
                reshaped_mask_image = tf.cast(
                    reshaped_mask_image, dtype=tf.float16
                )

            return input_image, reshaped_mask_image

        dataset = dataset.map(
            map_func=_resize_augment_images,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        if is_training:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
