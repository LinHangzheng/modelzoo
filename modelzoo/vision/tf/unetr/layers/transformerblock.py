from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import LayerNormalizationLayer
from layers import EmbeddedPatches, TransformerEncoder
class TransformerBlock(BaseLayer):
    """MLP head of vision transformer.
    
    Parameters
    ----------
    classes_num : int 
        The number of classes to predict.
    """
    
    def __init__(self, 
                 hidden_size,
                 dropout_rate,
                 layer_norm_epsilon,
                 encoders_num,
                 heads_num,
                 mlp_dim,
                 ret_scores,
                 extract_layers,
                 boundary_casting = False,
                 tf_summary=False, 
                 **kwargs,
    ):
        super(TransformerBlock,self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.boundary_casting = boundary_casting
        self.tf_summary = tf_summary
        self.hidden_size = hidden_size
        self.ret_scores = ret_scores
        self.extract_layers = extract_layers
        self.emb_patches = EmbeddedPatches(latent_dim=hidden_size, 
                                           dropout_rate=dropout_rate, 
                                           boundary_casting=boundary_casting,
                                           tf_summary=tf_summary,
                                           dtype=self.dtype_policy)
    
        self.encoder_norm =  LayerNormalizationLayer(dtype=self.dtype_policy,
                            epsilon=layer_norm_epsilon,
                            boundary_casting = boundary_casting,
                            tf_summary=tf_summary )
        self.encoders = [
            TransformerEncoder(hidden_size, 
                               heads_num, 
                               mlp_dim, 
                               dropout_rate, 
                               layer_norm_epsilon, 
                               dtype=self.dtype_policy, 
                               boundary_casting=boundary_casting,
                               tf_summary=tf_summary)
            for _ in range(encoders_num)
        ]
        
    
    def call(self, input):
        layer_out = []
        x = self.emb_patches(input)
        for depth, encoder in enumerate(self.encoders):
            x = encoder(x, self.ret_scores)
            if self.ret_scores:
                x, scores = x
            if depth + 1  in self.extract_layers:
                layer_out.append(x)
        return layer_out