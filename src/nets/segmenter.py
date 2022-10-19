# -*- coding: utf-8 -*-

import mindspore.nn as nn
from mindspore import ops

class Segmenter(nn.Cell):
    def __init__(self, encoder, decoder, model_cfg):
        super().__init__()
        self.n_cls = model_cfg["n_cls"]
        self.patch_size = model_cfg['patch_size']
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, im):
        H_ori, W_ori = im.shape[2], im.shape[3]
        im = padding(im, self.patch_size)
        H, W = im.shape[2], im.shape[3]
        x = self.encoder(im, return_features=True)
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x)
        resize_bilinear = ops.ResizeBilinear((H, W))
        masks = resize_bilinear(masks)
        masks = uppadding(masks, (H_ori, W_ori))
        return masks

def padding(im, patch_size):
    """
     制作图像分割大小
    :param im:
    :param patch_size:
    :return:
    """
    H, W = im.shape[2], im.shape[3]
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W  % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    pad = nn.Pad(paddings=((0, 0), (0, 0), (0, pad_w), (0, pad_h)), mode="CONSTANT")
    if pad_h > 0 or pad_w > 0:
        im_padded = pad(im)
    return im_padded

def uppadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.shape[2], y.shape[3]

    extra_h = H_pad - H
    extra_w = W_pad -W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


