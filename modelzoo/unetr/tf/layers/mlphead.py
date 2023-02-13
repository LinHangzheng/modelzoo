from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer

class MLPHead(BaseLayer):
    """MLP head of vision transformer.
    
    Parameters
    ----------
    classes_num : int 
        The number of classes to predict.
    """
    
    def __init__(self, classes_num, 
                 boundary_casting = False,
                 tf_summary=False, 
                 **kwargs,
    ):
        super(MLPHead,self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.dense = DenseLayer(
            classes_num, 
            boundary_casting = False,
            tf_summary=False, 
            dtype=self.dtype_policy)
    
    
    def call(self, input):
        return self.dense(input)