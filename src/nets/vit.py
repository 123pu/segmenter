# -*- coding: utf-8 -*-
'''
segmenter backbone vit implementation
'''
import mindspore as ms
from mindspore import Tensor, ops
import mindspore.nn as nn
from mindspore.nn import SequentialCell
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, TruncatedNormal, Constant
import numpy as np
from src.nets.blocks import Block


class PatchEmbedding(nn.Cell):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.proj = nn.Conv2d(
            channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=True,
            weight_init='xavier_uniform',
        )

    def construct(self, im):
        x = self.proj(im)
        B, C, H, W = x.shape
        x = self.reshape(x, (B, C, H*W))
        x = self.transpose(x, (0, 2, 1))
        return x

class VisionTransformer(nn.Cell):
    def __init__(self,
                 image_size,
                 patch_size,
                 n_layers,
                 d_model,
                 d_ff,
                 n_heads,
                 n_cls,
                 dropout=0.0,
                 drop_path_rate=0.1,
                 distilled=False,
                 channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, d_model, channels)
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(1. - dropout)
        self.n_cls = n_cls
        self.reshape = ops.Reshape()
        self.cat_1 = ops.Concat(axis=1)
        self.tile = ops.Tile()
        initialization = ms.common.initializer.Normal(sigma=1.0)
        init = ms.common.initializer.XavierUniform()
        self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)), name='cls_token', requires_grad=True)
        self.distlled = distilled
        if self.distlled:
            self.dist_token = Parameter(initializer(initialization, (1, 1, d_model)),name='dist_token', requires_grad=True)
            self.pos_embed = Parameter(initializer(initialization, (1, self.patch_embed.num_patches + 2, d_model)),
                                           name='pos_embed', requires_grad=True)
            self.head_dist =nn.Dense(d_model, n_cls)
            self.head_dist.weight.set_data(initializer(init, [n_cls, d_model]))
        else:
            self.pos_embed = Parameter(initializer(initialization, (1, self.patch_embed.num_patches + 1, d_model)),
                                          name='pos_embed', requires_grad=True)
        dpr = [x.item() for x in np.linspace(0.0, drop_path_rate, n_layers)]
        blocks = []
        for i in range(n_layers):
            blk = Block(d_model, n_heads, d_ff, dropout, dpr[i])
            blocks.append(blk)
        self.blocks = SequentialCell(blocks)
        # output head
        self.norm = nn.LayerNorm((d_model,))
        self.head = nn.Dense(d_model, n_cls)
        self.head.weight.set_data(initializer(init, [n_cls, d_model]))

    def construct(self, im, return_features=False):
        B, _, H, W = im.shape
        x = self.patch_embed(im)
        cls_tokens = self.tile(self.cls_token, (B, 1, 1))
        if self.distlled:
            dist_tokens = self.tile(self.dist_token, (B, 1, 1))
            x = self.cat_1((cls_tokens, dist_tokens, x))
        else:
            x = self.cat_1((cls_tokens, x))
        pos_embed = self.pos_embed
        if x.shape[1] != pos_embed.shape[1]:
            raise ValueError
        x = x + pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        if return_features:
            return x
        if self.distlled:
            x, x_dist = x[:, 0].astype('float32'), x[:, 1].astype('float32')
            x = self.head(x)
            x_dist = self.head_dist(x_dist)
            x = (x + x_dist) / 2
        else:
            x = x[:, 0].astype('float32')
            x = self.head(x)
        return x
