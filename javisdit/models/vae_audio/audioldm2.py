import os
import os.path as osp
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers.utils import logging
from diffusers import AudioLDM2Pipeline

from javisdit.registry import MODELS
from javisdit.utils.misc import requires_grad
from javisdit.utils.ckpt_utils import load_ckpt_state_dict

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@MODELS.register_module("AudioLDM2")
class AudioLDM2:
    def __init__(
        self,
        from_pretrained: Optional[str] = None,
        device='cuda',
        dtype=torch.float,
        init_to_device=True,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.pipe: AudioLDM2Pipeline = AudioLDM2Pipeline.from_pretrained(from_pretrained).to(dtype)
        if init_to_device:
            self.pipe.to(device)
        for m in [self.pipe.vae, self.pipe.vocoder]:
            m.eval()
            requires_grad(m, False)
        self.free_text()
        self.from_pretrained_path = from_pretrained

    def load_pretrained_ckpt(self):
        self.pipe.vae.load_state_dict(load_ckpt_state_dict(
            osp.join(self.from_pretrained_path, 'vae', 'diffusion_pytorch_model.safetensors')
        ))
        self.pipe.vocoder.load_state_dict(load_ckpt_state_dict(
            osp.join(self.from_pretrained_path, 'vocoder', 'model.safetensors')
        ))

    def free_text(self):
        del self.pipe.text_encoder; self.pipe.text_encoder = nn.Identity()
        del self.pipe.text_encoder_2; self.pipe.text_encoder_2 = nn.Identity()
        del self.pipe.projection_model; self.pipe.projection_model = nn.Identity()
        del self.pipe.language_model; self.pipe.language_model = nn.Identity()
        del self.pipe.feature_extractor; self.pipe.feature_extractor = nn.Identity()
        self.unet_config = deepcopy(self.pipe.unet.config)
        del self.pipe.unet; self.pipe.unet = nn.Identity()
        logger.info('AudioLDM2 text free.')

    def encode_audio(self, log_mel_spec: torch.Tensor):
        posterior = self.pipe.vae.encode(log_mel_spec, return_dict=False)[0]
        latents = posterior.sample() * self.pipe.vae.config.scaling_factor

        return latents

    def decode_audio(self, latents, original_waveform_length=None, return_np=False):
        latents = latents.to(self.dtype)
        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        mel_spectrogram = self.pipe.vae.decode(latents).sample

        audio = self.pipe.mel_spectrogram_to_waveform(mel_spectrogram)

        if original_waveform_length is not None:
            audio = audio[:, :original_waveform_length]

        if return_np:
            audio = audio.detach().cpu().numpy()

        return audio
    
    def prepare_latents(
        self, 
        audio_length_in_s: Optional[float] = None, 
        batch_size: int = 1,
        num_waveforms_per_prompt: Optional[int] = 1,
        device: torch.device = None,
        dtype: torch.dtype = None,
        latents: Optional[torch.FloatTensor] = None,
    ):
        # Convert audio input length from seconds to spectrogram height
        vocoder_upsample_factor = np.prod(self.pipe.vocoder.config.upsample_rates) / self.pipe.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = self.unet_config.sample_size * self.pipe.vae_scale_factor * vocoder_upsample_factor

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * self.pipe.vocoder.config.sampling_rate)
        vae_scale_factor = self.pipe.vae_scale_factor * 8  # TODO: fix audio pulse in saved .mp4 file
        if height % vae_scale_factor != 0:
            height = int(np.ceil(height / vae_scale_factor)) * vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )

        # Prepare latent variables
        num_channels_latents = self.unet_config.in_channels
        shape = (
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height // self.pipe.vae_scale_factor,
            self.pipe.vocoder.config.model_in_dim // self.pipe.vae_scale_factor,
        )
        latents = torch.randn(*shape, device=device, dtype=dtype)

        return latents, original_waveform_length
    
    @property
    def vae_out_channels(self):
        return self.pipe.vae.config.latent_channels
