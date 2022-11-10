
from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import LayerNormalizationLayer
from modelzoo.common.tf.layers.AttentionLayer import AttentionLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
class TransformerEncoder(BaseLayer):
    """This class represents one encoder of the transformer.
    
    Parameters
    ----------
    latent_dim : int 
        The size of latent vectors in encoder layers.
    heads_num : int 
        The number of heads in MSA layers inside encoder layer.
    mlp_dim : int
        The size of one hidden layer in the MLP inside encoder layer.
    dropout_rate : float
        Dropout rate.
    """
    
    def __init__(self, 
                 latent_dim, 
                 heads_num, 
                 mlp_dim, 
                 dropout_rate,
                 layer_norm_epsilon,
                 boundary_casting = False,
                 tf_summary=False, 
                 **kwargs):
        super(TransformerEncoder, self).__init__(
            boundary_casting, tf_summary, **kwargs)
        self.ln1 = LayerNormalizationLayer(dtype=self.dtype_policy,
                                            epsilon=layer_norm_epsilon,
                                            boundary_casting = boundary_casting,
                                            tf_summary=tf_summary )
        self.ln2 = LayerNormalizationLayer(dtype=self.dtype_policy,
                                            epsilon=layer_norm_epsilon,
                                            boundary_casting = boundary_casting,
                                            tf_summary=tf_summary)
        

        self.MSA = MSA(
            latent_dim, 
            heads_num, 
            dropout_rate, 
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        self.MLP = MLP(
            latent_dim, 
            mlp_dim, 
            dropout_rate, 
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        
    def call(self, input, ret_scores=False):
       #print("encoder input: ", input)
        norm_input = self.ln1(input)
       #print("first norm out: ", norm_input)
        if ret_scores:
            msa, scores = self.MSA(norm_input, ret_scores)
        else:
            msa = self.MSA(norm_input)
       #print("MSA out: ", msa)
        x = msa + norm_input
        norm_msa = self.ln2(x)
       #print("second norm out: ", norm_msa)
        mlp = self.MLP(norm_msa)
        output = mlp + norm_msa
        if ret_scores:
            return output, scores
        else:
            return output


class MSA(BaseLayer):
    """
    Multihead self-attention.
    
    Args:
        latent_dim (int): The size of latent vectors.
        heads_num (int): The number of heads.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(
        self, 
        latent_dim, 
        heads_num, 
        dropout_rate, 
        boundary_casting = False, 
        tf_summary=False, **kwargs):
        super(MSA, self).__init__(
            boundary_casting=boundary_casting, tf_summary=tf_summary,**kwargs)
        
        if int(latent_dim / heads_num) == 0:
            raise ValueError("Incorrect number of heads."
                             "Try to take smaller number.")

        self.mha = AttentionLayer(
            hidden_size=latent_dim, 
            num_heads=heads_num, 
            dropout_rate=dropout_rate,
            boundary_casting = boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
        )
    
    
    def call(self, input, ret_scores=False):
        return self.mha(input, input)
    

class MLP(BaseLayer):
    """
    A simple two-layer perceptron used in encoders.
    
    Args:
        latent_dim (int): The size of latent vectors.
        mlp_dim (int): The size of a hidden layer.
        dropout_rate (float): Dropout rate.
    """
    
    def __init__(
        self, 
        latent_dim, 
        mlp_dim, 
        dropout_rate,
        boundary_casting = False,
        tf_summary=False,
        **kwargs):
        super(MLP, self).__init__(
            boundary_casting=boundary_casting, tf_summary=tf_summary, **kwargs)
        self.dense1 = DenseLayer(
            mlp_dim, 
            boundary_casting = boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        
        self.dropout1 = DropoutLayer(
            dropout_rate, 
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        
        self.dense2 = DenseLayer(
            latent_dim, 
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        
        self.dropout2 = DropoutLayer(
            dropout_rate, 
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
    
    
    def call(self, input):
        x = self.dense1(input)
        x = ActivationLayer.gelu(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        output = self.dropout2(x)
        
        return output