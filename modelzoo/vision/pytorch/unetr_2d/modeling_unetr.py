import torch
import torch.nn as nn
from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.vision.pytorch.unetr_2d.layers import Transformer
from modelzoo.vision.pytorch.layers.ConvNormActBlock import ConvNormActBlock

class DeConv2dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2),
            nn.Conv2d(out_planes, out_planes, kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class UNETR(nn.Module):
    
    def bce_loss(self, outputs, labels):
        neg_outputs = -1 * outputs
        zero_const = torch.tensor(
            0.0, dtype=outputs.dtype, device=outputs.device
        )
        max_val = torch.where(neg_outputs > zero_const, neg_outputs, zero_const)
        loss = (
            (1 - labels)
            .mul(outputs)
            .add(max_val)
            .add((-max_val).exp().add((neg_outputs - max_val).exp()).log())
        )
        mean = torch.mean(loss)
        # The return needs to be a dtype of FP16 for WS
        return mean.to(half_dtype_instance.half_dtype)
    
    def __init__(self, model_params):
        
        super(UNETR, self).__init__()
        self.img_shape = model_params["img_shape"]
        self.input_dim = model_params["input_dim"]
        self.output_dim = model_params["output_dim"]
        self.embed_dim = model_params["embed_dim"] 
        self.patch_size = model_params["patch_size"]
        self.num_heads = model_params["num_heads"]
        self.dropout = model_params["dropout"]
        self.mlp_hidden = model_params["mlp_hidden"]
        self.num_layers = model_params["num_layers"]
        self.ext_layers = model_params["ext_layers"]
        self.loss_type = model_params["loss"]
        if "bce" in self.loss_type:
            self.loss_fn = self.bce_loss
        self.patch_dim = [x // self.patch_size for x in self.img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                self.input_dim,
                self.embed_dim,
                self.img_shape,
                self.patch_size,
                self.num_heads,
                self.num_layers,
                self.dropout,
                self.ext_layers,
                self.mlp_hidden
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                ConvNormActBlock(self.input_dim, 32, 3, padding='same'),
                ConvNormActBlock(32, 64, 3, padding='same')
            )

        self.decoder3 = \
            nn.Sequential(
                DeConv2dBlock(self.embed_dim, 512, 3),
                DeConv2dBlock(512, 256, 3),
                DeConv2dBlock(256, 128, 3)
            )

        self.decoder6 = \
            nn.Sequential(
                DeConv2dBlock(self.embed_dim, 512, 3),
                DeConv2dBlock(512, 256, 3),
            )

        self.decoder9 = \
            DeConv2dBlock(self.embed_dim, 512, 3)

        self.decoder12_upsampler = \
            nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2)

        self.decoder9_upsampler = \
            nn.Sequential(
                ConvNormActBlock(1024, 512, 3, padding='same'),
                ConvNormActBlock(512, 512, 3, padding='same'),
                ConvNormActBlock(512, 512, 3, padding='same'),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                ConvNormActBlock(512, 256, 3, padding='same'),
                ConvNormActBlock(256, 256, 3, padding='same'),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                ConvNormActBlock(256, 128, 3, padding='same'),
                ConvNormActBlock(128, 128, 3, padding='same'),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            )

        self.decoder0_header = \
            nn.Sequential(
                ConvNormActBlock(128, 64, 3, padding='same'),
                ConvNormActBlock(64, 64, 3, padding='same'),
                nn.Conv2d(64, self.output_dim, kernel_size=1),
            )

    def forward(self, x):
        
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output