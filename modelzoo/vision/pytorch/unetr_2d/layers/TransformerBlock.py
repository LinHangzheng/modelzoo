import torch.nn as nn
import torch
import copy
from modelzoo.common.pytorch.model_utils.activations import get_activation
from modelzoo.vision.pytorch.layers.utils import ModuleWrapperClass
from modelzoo.common.pytorch.layers.TransformerEncoder import TransformerEncoder
from modelzoo.common.pytorch.layers.TransformerEncoderLayer import (
    TransformerEncoderLayer,
)
from modelzoo.vision.pytorch.layers import Tra
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1, act='relu'):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = get_activation(act)
        self.act  = ModuleWrapperClass(self.act, act)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))
    
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, image_size, patch_size, mlp_hidden):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((image_size[0] * image_size[1] ) / (patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, mlp_hidden)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.attn = TransformerEncoderLayer(
            embed_dim,
            num_heads,
            mlp_hidden,
            dropout
        )
        
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights
    
class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((image_size[0] * image_size[1] ) / (patch_size  * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv2d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, image_size, patch_size, num_heads, num_layers, dropout, extract_layers, mlp_hidden):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, image_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, image_size, patch_size, mlp_hidden)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers
