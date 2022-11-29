import tensorflow as tf
# from tensorflow.keras.initializers import RandomNormal
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
class EmbeddedPatches(BaseLayer):
    """This layers extracts patches of size P and 
    projects them to a latent space of size D;
    Add class and position embeddings.
    
    Parameters
    ----------
    patch_number : int
        The number of square patchs.
    latent_dim : int 
        The size of latent vectors in encoder layers.
    dropout_rate : float
        Dropout rate.
    """
    
    def __init__(self, 
                 latent_dim,
                 dropout_rate,    
                 boundary_casting=False,
                 tf_summary=False, 
                 **kwargs):
        super(EmbeddedPatches, self).__init__(boundary_casting, tf_summary, **kwargs)
        self.latent_dim = latent_dim
        self.projection = DenseLayer(self.latent_dim,use_bias=False, 
                                     boundary_casting = boundary_casting, 
                                     tf_summary = tf_summary,  
                                     dtype=self.dtype_policy)
        self.class_emb = tf.cast(self.add_weight(
            name="class_emb",
            shape=(1, 1, self.latent_dim),
            initializer="zeros",
            dtype=self.variable_dtype
        ),
        self.compute_dtype,)
        self.broadcast_class_emb = None
        
        self.position_emb = None
        
        self.dropout = DropoutLayer(rate=dropout_rate, 
                                    boundary_casting = False,
                                    tf_summary=False, 
                                    dtype=self.dtype_policy)
        

    
    def build(self, input_shape):
        (B, N, C) = input_shape
        self.pos_emb = tf.cast(self.add_weight(         # [1, 50, 32]
            name="pos_emb",
            shape=(N + 1, self.latent_dim),
            # initializer=RandomNormal(stddev=0.02), # FROM BERT
            initializer="zeros", # FROM BERT
            dtype=self.variable_dtype
        ),
        self.compute_dtype,)
        
    def call(self, input):
        # input size: [256, 49, 16]
        batch_size = tf.shape(input)[0] 
        patches = self.projection(input) # [256, 49, 32]
        
        # add embedding
        bc_class_emb = tf.tile(self.class_emb, [batch_size,1,1])          # [256, 1, 32]
        patches = tf.concat([bc_class_emb, patches], axis=1)              # [256, 50, 32] 

        # add position embeddings
        patches = patches + self.pos_emb  # [256, 50, 32] <- [256, 50, 32] + [50, 32]
        
        # dropout
        patches = self.dropout(patches) # [256, 50, 32]
        return patches

    