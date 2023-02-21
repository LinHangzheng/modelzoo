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
UNet model to be used with TF Estimator
"""
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.compat.v1.losses import Reduction
from tensorflow.python.keras.layers import Flatten, concatenate

from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.Conv2DTransposeLayer import Conv2DTransposeLayer
from modelzoo.common.tf.layers.MaxPool2DLayer import MaxPool2DLayer
from modelzoo.common.tf.metrics.dice_coefficient import dice_coefficient_metric
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel

from modelzoo.unet.tf.utils import color_codes
from layers import TransformerBlock, ConvBlock, DeConvBlock, PermuteLayer

# from layers.permute import PermuteLayer
from modelzoo.common.tf.layers.ReshapeLayer import ReshapeLayer

from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import CrossEntropyFromLogitsLayer

class UNetr2DModel(TFBaseModel):
    """
    UNet model to be used with TF Estimator
    """

    def __init__(self, params):
        super(UNetr2DModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        self.num_classes = params["train_input"]["num_classes"]
        self.num_output_channels = 1

        self.logging_dict = {}
        
        patch_size = params["train_input"]["patch_size"]
        IR_channel_level = params["train_input"]["IR_channel_level"]
        vit_patch_size = params["train_input"]["vit_patch_size"]
        self.patch_dim = int(patch_size/vit_patch_size)
        ### Model params
        mparams = params["model"]
        self.hidden_size = mparams['hidden_size']
        heads_num = mparams['heads_num']
        mlp_dim = mparams['mlp_dim']
        encoders_num = mparams['encoders_num']
        # mlp_head_dim = params['model']['mlp_head_dim']
        dropout_rate = mparams['dropout_rate']
        tf_summary = mparams["tf_summary"]
        layer_norm_epsilon = mparams["layer_norm_epsilon"]
        boundary_casting = mparams["boundary_casting"]
        ret_scores = mparams['ret_scores']
        extract_layers = mparams['extract_layers']
        ##print("TFBaseModel dtype: ", self.policy)
        
        # self.skip_connect = mparams["skip_connect"]
        self.eval_ignore_classes = mparams.get("eval_ignore_classes", [])

        self.data_format = mparams["data_format"]
        self.features_axis = 1 if self.data_format == "channels_first" else -1
        self.downscale_method = mparams.get("downscale_method", "max_pool")

        self.enable_bias = mparams["enable_bias"]
        # self.nonlinearity = mparams["nonlinearity"]
        # self.nonlinearity_params = mparams.get("nonlinearity_params", dict())
        # self.nonlinearity = getattr(tf.keras.layers, self.nonlinearity)(
        #     **{**self.nonlinearity_params, **dict(dtype=self.policy)},
        # )

        self.initial_conv_filters = mparams.get("initial_conv_filters")
        self.convs_per_block = mparams.get(
            "convs_per_block", ["3x3_conv", "3x3_conv"]
        )

        self.eval_metrics = mparams.get(
            "eval_metrics", ["mIOU", "DSC", "MPCA", "Acc"]
        )

        self.initializer = mparams["initializer"]
        self.initializer_params = mparams.get("initializer_params")
        if self.initializer_params:
            self.initializer = getattr(
                tf.compat.v1.keras.initializers, self.initializer
            )(**self.initializer_params)

        self.bias_initializer = mparams["bias_initializer"]
        self.bias_initializer_params = mparams.get("bias_initializer_params")
        if self.bias_initializer_params:
            self.bias_initializer = getattr(
                tf.compat.v1.keras.initializers, self.bias_initializer
            )(**self.bias_initializer_params)

        # CS util params for layers
        self.boundary_casting = mparams["boundary_casting"]
        self.tf_summary = mparams["tf_summary"]

        self.output_dir = params["runconfig"]["model_dir"]
        self.log_image_summaries = mparams.get("log_image_summaries", False)
        self.mixed_precision = mparams["mixed_precision"]


        self.cross_entroy_loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=self.boundary_casting,
            tf_summary=self.tf_summary,
            dtype=self.policy,
        )
        
        self.transformer = TransformerBlock(
                self.hidden_size,
                dropout_rate,
                layer_norm_epsilon,
                encoders_num,
                heads_num,
                mlp_dim,
                ret_scores,
                extract_layers,
                boundary_casting,
                tf_summary,
                dtype=self.policy
        )
        
        self.decoder0 = tf.keras.Sequential([
            ConvBlock(                 
                 32, 
                 self.data_format,
                 self.enable_bias,
                 self.initializer,
                 self.bias_initializer,
                 layer_norm_epsilon,
                 kernel_size=3,
                 boundary_casting = boundary_casting,
                 tf_summary=tf_summary, 
                 dtype=self.policy),
            ConvBlock(                 
                 64, 
                 self.data_format,
                 self.enable_bias,
                 self.initializer,
                 self.bias_initializer,
                 layer_norm_epsilon,
                 kernel_size=3,
                 boundary_casting = boundary_casting,
                 tf_summary=tf_summary, 
                 dtype=self.policy)
        ])
        
        self.decoder3 = tf.keras.Sequential([
            DeConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy),
            DeConvBlock(                 
                256, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy),
            DeConvBlock(                 
                128, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy),
        ])
        
        
        self.decoder6 = tf.keras.Sequential([
            DeConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy),
            DeConvBlock(                 
                256, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy)
        ])
        
        self.decoder9 = DeConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy)
    
        self.decoder12_upsampler = Conv2DTransposeLayer(
                filters=512,
                kernel_size=2,
                strides=2,
                padding="same",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
        )
        
        self.decoder9_upsampler = tf.keras.Sequential([
            ConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            ConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            ConvBlock(                 
                512, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy),
            Conv2DTransposeLayer(
                filters=256,
                kernel_size=2,
                strides=2,
                padding="same",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,)
        ])
        
        self.decoder6_upsampler = tf.keras.Sequential([
            ConvBlock(                 
                256, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            ConvBlock(                 
                256, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            Conv2DTransposeLayer(
                filters=128,
                kernel_size=2,
                strides=2,
                padding="same",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,)
        ])
        
        
        self.decoder3_upsampler = tf.keras.Sequential([
            ConvBlock(                 
                128, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            ConvBlock(                 
                128, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            Conv2DTransposeLayer(
                filters=64,
                kernel_size=2,
                strides=2,
                padding="same",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,)
        ])
        
        self.decoder0_header = tf.keras.Sequential([
            ConvBlock(                 
                64, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            ConvBlock(                 
                64, 
                self.data_format,
                self.enable_bias,
                self.initializer,
                self.bias_initializer,
                layer_norm_epsilon,
                kernel_size=3,
                boundary_casting = boundary_casting,
                tf_summary=tf_summary, 
                dtype=self.policy),
            Conv2DLayer(
                filters=self.num_classes,
                kernel_size=1,
                strides=1,
                padding="same",
                # name=("enc_" if encoder else "dec_")
                # + f"conv{block_idx}_{conv_idx}",
                data_format=self.data_format,
                use_bias=self.enable_bias,
                kernel_initializer=self.initializer,
                bias_initializer=self.bias_initializer,
                boundary_casting=self.boundary_casting,
                tf_summary=self.tf_summary,
                dtype=self.policy,
            )
        ])
        
        self.permute_layer = PermuteLayer([2,1],
                                          dtype=self.policy, 
                                          boundary_casting=boundary_casting,
                                          tf_summary=tf_summary)
        self.reshape_layer = ReshapeLayer(
                               [self.hidden_size,self.patch_dim,self.patch_dim],
                               dtype=self.policy, 
                               boundary_casting=self.boundary_casting,
                               tf_summary=self.tf_summary)
        
        self.permute_logits_layer = PermuteLayer([2,3,1],
                                          dtype=self.policy, 
                                          boundary_casting=boundary_casting,
                                          tf_summary=tf_summary)
        self.reshape_logits_layer = ReshapeLayer(
                               [patch_size*patch_size,self.num_classes],
                               dtype=self.policy, 
                               boundary_casting=self.boundary_casting,
                               tf_summary=self.tf_summary)
        

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )


    def build_model(self, features, mode):
        image, x = features[0],features[1]
        batch = x.shape[0]
        
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = image, *z
        # [B, embding, vit_patch_size^2]  
        z3 = self.reshape_layer(self.permute_layer(z3))
        z6 = self.reshape_layer(self.permute_layer(z6))
        z9 = self.reshape_layer(self.permute_layer(z9))
        z12 = self.reshape_layer(self.permute_layer(z12))
        # z3 = tf.reshape(tf.transpose(z3, [0,2,1]), [batch,self.hidden_size,self.patch_dim,self.patch_dim])   
        # z6 = tf.reshape(tf.transpose(z6, [0,2,1]), [batch,self.hidden_size,self.patch_dim,self.patch_dim])   
        # z9 = tf.reshape(tf.transpose(z9, [0,2,1]), [batch,self.hidden_size,self.patch_dim,self.patch_dim])   
        # z12 = tf.reshape(tf.transpose(z12, [0,2,1]), [batch,self.hidden_size,self.patch_dim,self.patch_dim])
        
        z12 = self.decoder12_upsampler(z12)                             # [B,512,H/8,W/8]
        z9 = self.decoder9(z9)                                          # [B,512,H/8,W/8]
        z9 = self.decoder9_upsampler(tf.concat([z9, z12], axis=1))      # [B,256,H/4,W/4]
        z6 = self.decoder6(z6)                                          # [B,256,H/4,W/4]
        z6 = self.decoder6_upsampler(tf.concat([z6, z9], axis=1))       # [B,128,H/2,W/2]
        z3 = self.decoder3(z3)                                          # [B,256,H/2,W/2]
        z3 = self.decoder3_upsampler(tf.concat([z3, z6], axis=1))       # [B,64,H,W]
        z0 = self.decoder0(z0)                                          # [B,64,H,W]
        logits = self.decoder0_header(tf.concat([z0, z3], axis=1))      # [B,num_class,H,W]
        logits = self.permute_logits_layer(logits)                      # [B,H,W,num_class]
        logits = self.reshape_logits_layer(logits)                      # [B,H*W,num_class]
        return logits   


    def build_total_loss(self, logits, features, labels, mode):
        # Get input image and corresponding gt mask.
        input_image = features[0]
        reshaped_mask_image = labels
        reshaped_logits = logits
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # reshaped_logits = tf.transpose(logits,[0,2,3,1])    # 
        # reshaped_logits = tf.reshape(reshaped_logits, [reshaped_logits.shape[0], -1, reshaped_logits.shape[-1]])
        
        loss = self.cross_entroy_loss_layer(reshaped_mask_image,reshaped_logits)
        loss = tf.reduce_mean(input_tensor=loss)

        if self.log_image_summaries and is_training:
            self._write_image_summaries(
                logits, input_image, reshaped_mask_image, is_training=True,
            )

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def _write_image_summaries(
        self, logits, input_image, mask_image, is_training=True
    ):
        def _get_image_summary(img):
            """
            Make an image summary for 4d tensor image.
            """
            V = img - tf.reduce_min(input_tensor=img)
            V = V / tf.reduce_max(input_tensor=V)
            V *= 255
            V = tf.cast(V, tf.uint8)

            return V

        def _convert_mask_to_rgb(image):
            color_tensors = []
            for i in range(self.num_classes):
                color_tensors.append(
                    tf.concat(
                        [
                            (color_codes[i][0] / 255) * tf.ones_like(image),
                            (color_codes[i][1] / 255) * tf.ones_like(image),
                            (color_codes[i][2] / 255) * tf.ones_like(image),
                        ],
                        axis=-1,
                    )
                )

            image = tf.tile(image, tf.constant([1, 1, 1, 3], tf.int32))
            image_int32 = tf.cast(image, tf.int32)
            for i in range(self.num_classes):
                image = tf.where(
                    tf.math.equal(
                        image_int32,
                        tf.constant(i, shape=mask_image.shape, dtype=tf.int32),
                    ),
                    color_tensors[i],
                    image,
                )

            return image

        if is_training:
            eval_suffix = ""
        else:
            eval_suffix = "_eval"

        # Display original input image.
        input_image = tf.transpose(a=input_image, perm=[0, 2, 3, 1])

        mask_image = tf.reshape(mask_image, input_image.shape[0:3] + [1])
        mask_image = tf.cast(
            mask_image, tf.float16 if self.mixed_precision else tf.float32
        )

        if input_image.shape[-1] != 3:
            input_image = tf.tile(
                input_image, tf.constant([1, 1, 1, 3], tf.int32)
            )

        tf.compat.v1.summary.image(
            "Input_image" + eval_suffix, _get_image_summary(input_image), 3,
        )

        tf.compat.v1.summary.image(
            "Original_mask" + eval_suffix,
            _get_image_summary(_convert_mask_to_rgb(mask_image)),
            3,
        )

        tf.compat.v1.summary.image(
            "Input_image_mask_overlayed" + eval_suffix,
            _get_image_summary(
                0.6 * input_image + 0.4 * _convert_mask_to_rgb(mask_image)
            ),
            3,
        )

        if self.num_output_channels == 1:
            logits = tf.concat(
                [tf.ones(logits.shape, dtype=logits.dtype) - logits, logits,],
                axis=-1,
            )

        preds = tf.argmax(input=logits, axis=3)
        preds = tf.expand_dims(preds, -1)
        preds = tf.cast(preds, mask_image.dtype)

        # Display reconstructed mask from U-Net.
        tf.compat.v1.summary.image(
            "Reconstruction_mask" + eval_suffix,
            _get_image_summary(_convert_mask_to_rgb(preds)),
            3,
        )

        tf.compat.v1.summary.image(
            "Reconstruction_mask_overlayed" + eval_suffix,
            _get_image_summary(
                0.6 * input_image + 0.4 * _convert_mask_to_rgb(preds)
            ),
            3,
        )

    def build_eval_metric_ops(self, logits, labels, features):
        """
        Evaluation metrics
        """
        reshaped_mask_image = labels

        reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)

        # Ensure channels are the last dimension for the rest of eval
        # metric calculations. Otherwise, need to do the rest of ops
        # according to the channels dimension
        if self.data_format == "channels_first":
            logits = tf.transpose(a=logits, perm=[0, 2, 3, 1])

        pred = tf.reshape(
            logits, [tf.shape(input=logits)[0], -1, self.num_output_channels],
        )

        if self.num_output_channels == 1:
            pred = tf.concat(
                [tf.ones(pred.shape, dtype=pred.dtype) - pred, pred], axis=-1
            )

        pred = tf.argmax(pred, axis=-1)

        # ignore void classes
        ignore_classes_tensor = tf.constant(
            False, shape=reshaped_mask_image.shape, dtype=tf.bool
        )
        for ignored_class in self.eval_ignore_classes:
            ignore_classes_tensor = tf.math.logical_or(
                ignore_classes_tensor,
                tf.math.equal(
                    reshaped_mask_image,
                    tf.constant(
                        ignored_class,
                        shape=reshaped_mask_image.shape,
                        dtype=tf.int32,
                    ),
                ),
            )

        weights = tf.where(
            ignore_classes_tensor,
            tf.zeros_like(reshaped_mask_image),
            tf.ones_like(reshaped_mask_image),
        )

        metrics_dict = dict()

        if "DSC" in self.eval_metrics:
            metrics_dict["eval/dice_coefficient"] = dice_coefficient_metric(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "mIOU" in self.eval_metrics:
            metrics_dict["eval/mean_iou"] = tf.compat.v1.metrics.mean_iou(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "MPCA" in self.eval_metrics:
            metrics_dict[
                "eval/mean_per_class_accuracy"
            ] = tf.compat.v1.metrics.mean_per_class_accuracy(
                labels=reshaped_mask_image,
                predictions=pred,
                num_classes=self.num_classes,
                weights=weights,
            )

        if "Acc" in self.eval_metrics:
            metrics_dict["eval/accuracy"] = tf.compat.v1.metrics.accuracy(
                labels=reshaped_mask_image, predictions=pred, weights=weights,
            )

        return metrics_dict

    def get_evaluation_hooks(self, logits, labels, features):
        """ As a result of this TF issue, need to explicitly define summary
        hooks to able to log image summaries in eval mode
        https://github.com/tensorflow/tensorflow/issues/15332
        """
        if self.log_image_summaries:
            input_image = features
            reshaped_mask_image = labels
            reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)
            self._write_image_summaries(
                logits, input_image, reshaped_mask_image, is_training=False,
            )
            summary_hook = tf.estimator.SummarySaverHook(
                save_steps=1,
                output_dir=self.output_dir,
                summary_op=tf.compat.v1.summary.merge_all(),
            )
            return [summary_hook]
        else:
            return None
