# -*- coding: utf-8 -*-

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.initializer import initializer, TruncatedNormal, Constant
from mindspore.common import initializer as init
from mindspore.nn import SequentialCell
from mindspore import Parameter, Tensor, ops
import numpy as np
from src.nets.blocks import Block


class DecoderLinear(nn.Cell):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.head = nn.Dense(self.d_encoder, n_cls)

    def construct(self, x):
        H, W = self.im_size
        GS = H // self.patch_size
        x = self.head(x)
        x_numpy = x.asnumpy()
        b, h, w, c = x_numpy.size()
        x_arry = x_numpy.reshape((b, GS, w, c)).transpose((0, 3, 1, 2))
        x = Tensor(x_arry, ms.float32)
        return x

class MaskTransformer(nn.Cell):
    def __init__(
            self,
            n_cls,
            patch_size,
            d_encoder,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            drop_path_rate,
            dropout,
            im_size,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.im_size = im_size
        self.cat_1 = ops.Concat(axis=1)
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        initialization = ms.common.initializer.Normal(sigma=1.0)
        init = ms.common.initializer.XavierUniform()
        dpr = [x.item() for x in np.linspace(0.0, drop_path_rate, n_layers)]
        blocks = []
        for i in range(n_layers):
            blk = Block(d_model, n_heads, d_ff, dropout, dpr[i])  # 1024, 16, 4096, 0.1, [0-0.1]
            blocks.append(blk)
        self.blocks = SequentialCell(blocks)
        self.cls_embed = Parameter(initializer(initialization, (1, n_cls, d_model)),
                                   name='cls_embed', requires_grad=True)
        self.proj_dec = nn.Dense(d_encoder, d_model)     # 1024, 1024
        self.proj_dec.weight.set_data(initializer(init, [d_model, d_encoder]))
        self.proj_patch = Parameter(initializer(initialization, (d_model, d_model)),
                                    name='proj_patch', requires_grad=True)
        self.proj_classes = Parameter(initializer(initialization, (d_model, d_model)),
                                      name='proj_classes', requires_grad=True)
        self.decoder_norm = nn.LayerNorm((self.d_model,))
        self.mask_norm = nn.LayerNorm((self.n_cls,))
        self.reshape = ops.Reshape()
        self.sqrt = ops.Sqrt()

    def construct(self, x):
        x = self.proj_dec(x)               # (1, 2304, 1024)
        cls_emb = self.tile(self.cls_embed, (x.shape[0], 1, 1)) # (1, 19, 1024)
        x = self.cat_1((x, cls_emb))   # (1, 2323, 1024)
        x = self.blocks(x)
        x = self.decoder_norm(x)
        H, W = self.im_size         # 768, 768   1, 2304, 1024 -- 1, 19, 1024
        patches, cls_seg_feat = x[:, : -self.n_cls],  x[:, -self.n_cls :]
        patches = ops.matmul(patches, self.proj_patch)
        b1, h1, w1 = patches.shape
        patchs_ones_tensor = Tensor(np.ones((b1, w1, 1)), ms.float32)
        patch_l2_norm = self.sqrt(ops.matmul(patches ** 2, patchs_ones_tensor))
        cls_seg_feat = ops.matmul(cls_seg_feat, self.proj_classes)
        b2, h2, w2 = cls_seg_feat.shape
        cls_ones_tensor = Tensor(np.ones((b2, w2, 1)), ms.float32)
        cls_l2_norm = self.sqrt(ops.matmul(cls_seg_feat ** 2, cls_ones_tensor))
        patches = patches / patch_l2_norm
        cls_seg_feat = cls_seg_feat / cls_l2_norm
        cls_seg_feat = self.transpose(cls_seg_feat, (0, 2, 1))   # (1, 1024, 19)
        masks = ops.matmul(patches, cls_seg_feat)  # (1, 2304, 1024)  -- (1, 1024, 19)
        masks = self.mask_norm(masks)                # (1, 2304, 19)
        b, hw, n = masks.shape
        masks_tensor = self.reshape(masks, (b, H // self.patch_size, hw // (H // self.patch_size), n))  # (1, 48, 48, 19)
        masks = self.transpose(masks_tensor, (0, 3, 1, 2))  # (1, 19, 48, 48)  # (n, c, h, w)
        return masks




