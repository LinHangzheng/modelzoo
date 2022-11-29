from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.Conv2DLayer import Conv2DLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import LayerNormalizationLayer
class ConvBlock(BaseLayer):
    """MLP head of vision transformer.
    
    Parameters
    ----------
    classes_num : int 
        The number of classes to predict.
    """
    
    def __init__(self, 
                 output_n, 
                 data_format,
                 enable_bias,
                 initializer,
                 bias_initializer,
                 layer_norm_epsilon,
                 kernel_size=3,
                 boundary_casting = False,
                 tf_summary=False, 
                 **kwargs,
    ):
        super(ConvBlock,self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.conv =  Conv2DLayer(
                        filters=output_n,
                        kernel_size=kernel_size,
                        strides=(1, 1),
                        padding=((kernel_size - 1) // 2),
                        data_format=data_format,
                        use_bias=enable_bias,
                        kernel_initializer=initializer,
                        bias_initializer=bias_initializer,
                        boundary_casting=boundary_casting,
                        tf_summary=tf_summary,
                        dtype=self.dtype_policy,
                    )
    
        self.norm =  LayerNormalizationLayer(
                    dtype=self.dtype_policy,
                    epsilon=layer_norm_epsilon,
                    boundary_casting = boundary_casting,
                    tf_summary=tf_summary)
    
    def call(self, input):
        x = self.conv(input)
        x = self.norm(x)
        x = ActivationLayer.relu6(x)
        return x