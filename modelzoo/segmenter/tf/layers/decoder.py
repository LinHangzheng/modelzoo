import tensorflow as tf
from modelzoo.common.tf.layers.ActivationLayer import ActivationLayer
from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import LayerNormalizationLayer
from modelzoo.common.tf.layers.AttentionLayer import AttentionLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from layers import EmbeddedPatches, TransformerEncoder
class MaskTransformer(BaseLayer):
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
                 class_num,
                 drop_path_rate,
                 patch_size,
                 hidden_size,
                 dropout_rate,
                 layer_norm_epsilon,
                 decoders_num,
                 heads_num,
                 mlp_dim,
                 ret_scores,
                 extract_layers,
                 boundary_casting = False,
                 tf_summary=False, 
                 **kwargs,
    ):
        super(MaskTransformer, self).__init__(
            boundary_casting, tf_summary, **kwargs)
        self.class_num = class_num
        self.boundary_casting = boundary_casting
        self.tf_summary = tf_summary
        self.hidden_size = hidden_size
        self.ret_scores = ret_scores
        self.extract_layers = extract_layers
        self.patch_size = patch_size
        self.scale = hidden_size**-0.5
        # dpr = [x for x in tf.linspace(0,drop_path_rate,decoders_num)]
        self.blocks = [
            TransformerEncoder(hidden_size, 
                               heads_num, 
                               mlp_dim, 
                               dropout_rate, 
                               layer_norm_epsilon, 
                               dtype=self.dtype_policy, 
                               boundary_casting=boundary_casting,
                               tf_summary=tf_summary)
            for _ in range(decoders_num)
        ]
        
        self.dense1 = DenseLayer(
            hidden_size, 
            boundary_casting = boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy)
        
        
        # self.emb_patches = EmbeddedPatches(latent_dim=hidden_size, 
        #                                    dropout_rate=dropout_rate, 
        #                                    boundary_casting=boundary_casting,
        #                                    tf_summary=tf_summary,
        #                                    dtype=self.dtype_policy)
    
        self.cls_emb = self.add_weight(name="class_embedding", 
                                       shape=[1, class_num, hidden_size],
                                       initializer='random_normal',
                                       trainable=True) 
        
        self.proj_patch= self.scale*self.add_weight(name="projected_patch", 
                                       shape=[hidden_size, hidden_size],
                                       initializer='random_normal',
                                       trainable=True) 
        
        self.proj_classes = self.scale*self.add_weight(name="projected_classes", 
                                       shape=[hidden_size, hidden_size],
                                       initializer='random_normal',
                                       trainable=True) 

        self.cls_emb = tf.cast(self.cls_emb, tf.float16)
        self.proj_patch = tf.cast(self.proj_patch, tf.float16)
        self.proj_classes = tf.cast(self.proj_classes, tf.float16)
        
        self.decoder_norm =  LayerNormalizationLayer(dtype=self.dtype_policy,
                            epsilon=layer_norm_epsilon,
                            boundary_casting = boundary_casting,
                            tf_summary=tf_summary )
        
        self.mask_norm =  LayerNormalizationLayer(dtype=self.dtype_policy,
                            epsilon=layer_norm_epsilon,
                            boundary_casting = boundary_casting,
                            tf_summary=tf_summary )
        
        
        
    def call(self, input, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.dense1(input)
        cls_emb = tf.tile(self.cls_emb,multiples=(x.shape[0],1,1))  
        x = tf.concat((x,cls_emb),1)
        for decoder in self.blocks:
            x = decoder(x, mask=None, ret_scores=self.ret_scores)
        x = self.decoder_norm(x)
        
        # [b,4,768], [b,7,768]
        patches, cls_seg_feat = x[:, : -self.class_num], x[:, -self.class_num :]
        patches =  patches @ self.proj_patch    # [b,4,768]
        cls_seg_feat = cls_seg_feat @ self.proj_classes # [b,7,768]
        
        patches = patches / tf.norm(patches,axis=-1, keepdims=True) # [b,4,768]
        cls_seg_feat = cls_seg_feat / tf.norm(cls_seg_feat,axis=-1, keepdims=True)  # [b,7,768]
        
        masks = tf.matmul(patches, cls_seg_feat, transpose_b=True) # [b, 4, 7]
        masks = self.mask_norm(masks)   # b (h w) n
        
        masks = tf.reshape(masks,[masks.shape[0],GS,masks.shape[1]//GS,masks.shape[2]]) # b h w n
        # masks = tf.transpose(masks, perm=[0,3,1,2])                                     # b n h w
        return masks