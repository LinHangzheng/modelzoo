import tensorflow as tf
import os
import h5py
from copy import *
import numpy as np
from patchify import patchify
# import matplotlib.image

class IR_dataset:
    def __init__(self, params=None):
        self.data_dir = params["train_input"]["dataset_path"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                "The dataset directory `%s` does not exist." % self.data_dir
            )
        self.mixed_precision = params["model"]["mixed_precision"]
        self.num_classes = params["train_input"]["num_classes"]
        self.IR_threshold = params["train_input"]["IR_threshould"]
        self.patch_size = params["train_input"]["patch_size"]
        self.train_test_split = params["train_input"]["train_test_split"]
        self.patch_step = params["train_input"]["patch_step"]
        self.shuffle = params["train_input"]["shuffle"]
        self.shuffle_buffer_size = params["train_input"]["shuffle_buffer_size"]
        self.seed = params["train_input"].get("seed", None)
        self.num_parallel_calls = params["train_input"].get("num_parallel_calls", 0)
        IR = np.array(h5py.File(os.path.join(self.data_dir,'IR.mat'), 'r')['X'])
        IR = self.normolize(IR)
        
        Class = np.array(h5py.File(os.path.join(self.data_dir,'Class.mat'), 'r')['CL'])
        IR = np.moveaxis(IR, 0, -1)

        self.train_IR = None
        self.test_IR = None
        self.train_Class = None
        self.test_Class = None 
        self.data_split(IR,Class,self.train_test_split)
        
        self.train_IR_ds, self.train_Class_ds = self.prepare_data(self.train_IR, self.train_Class)
        self.test_IR_ds, self.test_Class_ds = self.prepare_data(self.test_IR, self.test_Class)

        self.train_ds = tf.data.Dataset.from_tensor_slices((self.train_IR_ds, self.train_Class_ds))
        self.test_ds = tf.data.Dataset.from_tensor_slices((self.test_IR_ds, self.test_Class_ds))
        # matplotlib.image.imsave(f'train.jpeg', self.train_Class)
        # matplotlib.image.imsave(f'test.jpeg', self.test_Class)
        # for i in range(10):
        #     matplotlib.image.imsave(f'train_c0_{i}.jpeg', self.train_IR_ds[i,0])
        # matplotlib.image.imsave(f'class.jpeg', Class)
        # matplotlib.image.imsave(f'IR_c1.jpeg', IR[:,:,0])
        # matplotlib.image.imsave(f'IR_c7.jpeg', IR[:,:,6])
    
    def normolize(self, IR):
        negative_idx = np.where(IR<0)
        IR[negative_idx] = 0
        return IR
    
    def data_split(self, IR, Class, train_test_split):
        split_col = int(IR.shape[1]*train_test_split)
        self.train_IR = IR[:,0:split_col]
        self.test_IR = IR[:,split_col:]
        self.train_Class = Class[:,:split_col]
        self.test_Class = Class[:,split_col:]
    
    def prepare_data(self, IR, Class):
        patches = patchify(IR[:,:,0], 
                           (self.patch_size,self.patch_size), 
                           step=self.patch_step)
        patches_idx = np.where(np.mean(patches,axis=(2,3))>self.IR_threshold)
        IR_ds = []
        
        for i in range(IR.shape[2]):
            patches = patchify(IR[:,:,i], 
                           (self.patch_size,self.patch_size), 
                           step=self.patch_step)
            IR_ds.append(patches[patches_idx])
        IR_ds = np.stack(IR_ds) #[19,N,H,W]
        IR_ds = np.moveaxis(IR_ds,0,1) #[N,19,H,W]
        Class_ds = patchify(Class,
                        (self.patch_size,self.patch_size),
                        step=self.patch_step)
        Class_ds = Class_ds[patches_idx]
        Class_ds = Class_ds.reshape((Class_ds.shape[0],-1))
        if self.mixed_precision:
                Class_ds = tf.cast(Class_ds, dtype=tf.float16)
                IR_ds = tf.cast(
                    IR_ds, dtype=tf.float16
                )
        return IR_ds, Class_ds
            
    def _augmentation(self,sample, labels):
        # TODO: use affine transformation for data augmentation
        
        return sample,labels
    
    def dataset_fn(
        self, batch_size, augment_data=True, shuffle=True, is_training=True,
    ):
        dataset = self.train_ds if is_training else self.test_ds
            
        # dataset = dataset.cache()
        if is_training and shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size, self.seed)
        if is_training:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
        dataset = dataset.map(
            self._augmentation,
            num_parallel_calls=self.num_parallel_calls
            if self.num_parallel_calls > 0
            else tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
   
