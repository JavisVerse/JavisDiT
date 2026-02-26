import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
from torchvision.io.video import read_video as read_video_tv
import torchaudio
import soundfile as sf
import librosa
import av

from .utils import VID_EXTENSIONS, AUD_EXTENSIONS

MAX_NUM_FRAMES = 2500


def read_audio_from_video_with_pyav(video_path, sr=16000):
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]

    if sr is None or audio_stream.rate == sr:
        audio_frames = []
        for frame in container.decode(audio_stream):
            audio_frames.append(frame.to_ndarray())
        assert len(audio_frames)

        audio_data = np.concatenate(audio_frames, axis=1)
        sr = audio_stream.rate
    else:
        resampler = av.AudioResampler(
            format='s16', 
            layout=audio_stream.layout.name, 
            rate=sr
        )
        resampled_frames = []
        for frame in container.decode(audio_stream):
            resampled_frames.extend(resampler.resample(frame))
            
        resampled_frames.extend(resampler.resample(None))
        
        audio_data = np.concatenate([frame.to_ndarray() for frame in resampled_frames], axis=1)
    
    audio_data = audio_data.reshape((-1, audio_stream.channels))
    if np.issubdtype(audio_data.dtype, np.integer):  # Need normalization
        max_val = np.iinfo(audio_data.dtype).max + 1
        audio_data = audio_data.astype(np.float32) / max_val

    container.close()

    ainfo = {'audio_fps': float(sr)}

    return audio_data[:, 0], ainfo


def read_audio(
    audio_path, sr=None,
    backend: Literal["auto", "torch", "sf", "av", "librosa"] = "auto"
) -> Tuple[torch.Tensor, Dict[str, int]]:
    ext = os.path.splitext(audio_path)[-1].lower()
    if backend == 'auto': 
        if ext not in VID_EXTENSIONS and ext in ['.wav', '.aiff', '.flac', '.ogg']:
            backend = 'sf'
        else:
            backend = 'av'
    # normalized, (-1.0 ~ 1.0)
    if backend == "torch":
        if ext in VID_EXTENSIONS:
            _, aframes, ainfo = read_video_tv(filename=audio_path, pts_unit="sec", output_format="TCHW")
            del ainfo['video_fps']
            ainfo = {'audio_fps': float(ainfo['audio_fps'])}
        elif ext in AUD_EXTENSIONS:
            aframes, fs = torchaudio.load(audio_path)
            ainfo = {'audio_fps': float(fs)}
        else:
            raise ValueError(f"Unsupported audio format: {audio_path}")
        aframes = aframes[0]  # dual track
    elif backend == 'sf':
        if ext not in ['.wav', '.aiff', '.flac', '.ogg']:
            warnings.warn(f'Unsupported audio extension: {ext}')
        aframes, fs = sf.read(audio_path)
        ainfo = {'audio_fps': float(fs)}
        aframes = torch.from_numpy(aframes).to(torch.float32)
        if len(aframes.shape) > 1:  # TODO: check format
            if aframes.shape[1] < 0.1 * fs:
                aframes = aframes[:, 0]
            else:
                aframes = aframes[0, :]
    elif backend == 'librosa':
        aframes, sr = librosa.load(audio_path, sr=sr)
        if len(aframes.shape) == 2:
            aframes = aframes[0]
        ainfo = {'audio_fps': float(sr)}
        aframes = torch.from_numpy(aframes).to(torch.float32)
    elif backend == 'av':
        aframes, ainfo = read_audio_from_video_with_pyav(audio_path, sr=sr)
        aframes = torch.from_numpy(aframes).to(torch.float32)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return aframes, ainfo

