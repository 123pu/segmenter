# -*- coding: utf-8 -*-

from src.nets.vit import VisionTransformer
from src.nets.decoder import DecoderLinear, MaskTransformer
from src.nets.segmenter import Segmenter



def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")
    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]  # 4096
    model = VisionTransformer(**model_cfg)
    return model

def create_decoder(model_cfg, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = model_cfg['d_model']
    decoder_cfg["patch_size"] = model_cfg['patch_size']
    decoder_cfg['im_size'] = model_cfg['image_size']

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = model_cfg['d_model']
        n_heads = dim // 64   # 16
        decoder_cfg['n_heads'] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    encoder = create_vit(model_cfg)
    decoder = create_decoder(model_cfg, decoder_cfg)
    model = Segmenter(encoder, decoder, model_cfg)
    return model



