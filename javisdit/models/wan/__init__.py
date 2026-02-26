from . import configs, distributed, modules
from .first_last_frame2video import WanFLF2V
from .image2video import WanI2V
from .text2video import WanT2V
from .vace import WanVace, WanVaceMP

from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE

import os
import logging
from javisdit.registry import MODELS


@MODELS.register_module("Wan2_1_T2V_1_3B")
def Wan_T2V_1_3B(from_pretrained=None, **kwargs):
    if from_pretrained is not None:
        logging.info(f'loading {from_pretrained}')
        model = WanModel.from_pretrained(from_pretrained, **kwargs)
    else:
        model = WanModel(**kwargs)
    return model


@MODELS.register_module("Wan2_1_T2V_1_3B_VAE")
def Wan_T2V_1_3B_VAE(from_pretrained=None, **kwargs):
    vae_checkpoint = kwargs.pop("vae_checkpoint", "Wan2.1_VAE.pth")
    model = WanVAE(
        vae_pth=os.path.join(from_pretrained, vae_checkpoint), 
        **kwargs
    )
    return model


@MODELS.register_module("Wan2_1_T2V_1_3B_t5_umt5")
def Wan_T2V_1_3B_t5_umt5(from_pretrained=None, **kwargs):
    t5_checkpoint = kwargs.pop("t5_checkpoint", "models_t5_umt5-xxl-enc-bf16.pth")
    t5_tokenizer = kwargs.pop("t5_tokenizer", "google/umt5-xxl")
    model = T5EncoderModel(
        text_len=kwargs.pop("text_len", 512),
        checkpoint_path=os.path.join(from_pretrained, t5_checkpoint),
        tokenizer_path=os.path.join(from_pretrained, t5_tokenizer),
        **kwargs
    )
    return model