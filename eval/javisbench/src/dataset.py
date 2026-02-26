import os
import os.path as osp
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple, Union

import cv2
from moviepy.editor import VideoFileClip
from PIL import Image
from torio.io import StreamingMediaDecoder
from transformers import AutoProcessor

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2
import torchaudio
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torchvision.transforms._transforms_video import NormalizeVideo

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".", "ImageBind"))
from .ImageBind.imagebind import data as imagebind_data
from .ImageBind.imagebind.models.imagebind_model import ModalityType
from .wav2spec import get_spectrogram
from .av_align import (
    detect_audio_peaks, extract_frames, detect_video_peaks, 
    calc_intersection_over_union
)
from .VideoAlign.prompt_template import build_prompt
from .VideoAlign.vision_process import process_vision_info
from .utils import ResizeAndPad, read_video, read_audio, smart_pad


class FVDFADDataset(Dataset):
    def __init__(self, mm_list, video_fps, audio_sr, max_frames,
                 max_audio_len_s=None, transform=None, audio_only=False):
        super().__init__()
        self.video_list, self.audio_list = mm_list
        self.video_fps = video_fps  # unused
        self.audio_sr = audio_sr
        self.max_frames = max_frames
        self.max_audio_len_s = max_audio_len_s
        self.transform = transform
        self.audio_only = audio_only

    def __len__(self):
        return len(self.audio_list)
        
    def __getitem__(self, idx):
        if self.audio_only:
            audio_path = self.audio_list[idx]
            audio_data = read_audio(audio_path, sr=self.audio_sr, 
                                    max_audio_len_s=self.max_audio_len_s, padding=True)
            return {"audio":audio_data}

        # load video&audio pair from list
        video_path = self.video_list[idx]
        audio_path = self.audio_list[idx]

        # load video
        # shape: [T, H, W, C]
        video_data = read_video(video_path, self.max_frames, 
                                frame_transform=self.transform)

        # load audio
        # shape: [C, T]
        audio_data = read_audio(audio_path, sr=self.audio_sr,
                                max_audio_len_s=self.max_audio_len_s, padding=True)

        # return video&audio pair
        video_data = ((video_data + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        return {"video": video_data, "audio": audio_data}
    

class VideoFrameDataset(Dataset):
    """load video frames
    """
    def __init__(self, video_path_list, prompt_list, num_frames=48, mode='linspace',
                 frame_transform=None, **kwargs):
        super().__init__()
        assert len(video_path_list) == len(prompt_list)
        self.video_path_list = video_path_list
        self.prompt_list = prompt_list
        self.frame_transform = frame_transform

        self.num_frames = num_frames
        self.mode = mode
    
    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        video = read_video(video_path, self.num_frames, mode='linspace',
                           frame_transform=self.frame_transform)

        prompt = self.prompt_list[index]

        return video, prompt


class AudioDataset(Dataset):
    def __init__(self, audio_path_list, prompt_list, sr=16000, max_audio_len_s=None, padding=True, 
                 audio_transform=None, ta_processor=None, **kwargs):
        super().__init__()
        assert len(audio_path_list) == len(prompt_list)
        self.audio_path_list = audio_path_list
        self.prompt_list = prompt_list
        self.sr = sr
        self.max_audio_len_s = max_audio_len_s
        self.padding = padding

        self.audio_transform = audio_transform
        self.ta_processor = ta_processor
        self.load_kwargs = kwargs
    
    def __len__(self):
        return len(self.audio_path_list)
    
    def __getitem__(self, index):
        audio_path = self.audio_path_list[index]

        audio = read_audio(audio_path, self.sr, self.max_audio_len_s, self.padding, 
                           audio_transform=self.audio_transform, **self.load_kwargs)
        prompt = self.prompt_list[index]

        return audio, prompt


class VideoAlignDataset(Dataset):
    """load video frames
    """
    def __init__(self, video_path_list, prompt_list, data_config):
        super().__init__()
        assert len(video_path_list) == len(prompt_list)
        self.video_path_list = video_path_list
        self.prompt_list = prompt_list
        self.data_config = data_config
    
    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        prompt = self.prompt_list[index]

        fps = self.data_config.fps
        num_frames = self.data_config.num_frames
        max_pixels = self.data_config.max_frame_pixels

        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video", 
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ]
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{video_path}", 
                                "max_pixels": max_pixels, 
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {"type": "text", "text": build_prompt(prompt, self.data_config.eval_dim, self.data_config.prompt_template_type)},
                        ],
                    },
                ]
            ]
        image_inputs, video_inputs = process_vision_info(chat_data)
        assert image_inputs is None
        
        return chat_data, video_inputs


class CavpDataset(Dataset):
    def __init__(self, video_path_list, audio_path_list, sr=16000):
        super().__init__()
        assert len(video_path_list) == len(audio_path_list)
        self.video_path_list = video_path_list
        self.audio_path_list = audio_path_list
        self.sr = sr

    def __len__(self):
        return len(self.video_path_list)
    
    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        audio_path = self.audio_path_list[index]

        video = VideoFileClip(video_path)
        duration = video.duration

        truncate_second = int(duration)

        audio = get_spectrogram(audio_path, self.sr * truncate_second)[1]

        return video_path, audio, truncate_second


class ImageBindDataset(Dataset):
    def __init__(self, video_path_list, audio_path_list, prompt_list, audio_prompt_list=None):
        super().__init__()
        assert len(video_path_list) == len(audio_path_list) == len(prompt_list)
        self.video_path_list = video_path_list
        self.audio_path_list = audio_path_list
        self.prompt_list = prompt_list
        self.audio_prompt_list = audio_prompt_list or prompt_list

    def __len__(self):
        return len(self.video_path_list)
    
    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        audio_path = self.audio_path_list[index]
        prompt = self.prompt_list[index]
        audio_prompt = self.audio_prompt_list[index]

        inputs = {
            ModalityType.TEXT: imagebind_data.load_and_transform_text([prompt, audio_prompt], 'cpu'),
            ModalityType.VISION: imagebind_data.load_and_transform_video_data([video_path], 'cpu')[0],
            ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data([audio_path], 'cpu')[0],
        }

        return inputs


class AVAlignDataset(Dataset):
    def __init__(self, video_path_list, audio_path_list, size=None, max_length_s=None):
        super().__init__()
        assert len(video_path_list) == len(audio_path_list)
        self.video_path_list = video_path_list
        self.audio_path_list = audio_path_list
        self.size = size
        self.max_length_s = max_length_s
    
    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path, audio_path = self.video_path_list[index], self.audio_path_list[index]

        audio_peaks = detect_audio_peaks(audio_path, max_length_s=self.max_length_s)

        frames, fps = extract_frames(video_path, self.size, max_length_s=self.max_length_s)
        flow_trajectory, video_peaks = detect_video_peaks(frames, fps, use_tqdm=False)
                
        av_align_score = calc_intersection_over_union(audio_peaks, video_peaks, fps)

        return av_align_score


class AVScoreDataset(Dataset):
    def __init__(self, video_path_list, audio_path_list, prompt_list, sample_rate=16000,
                 window_size_s: float=0.5, window_overlap_s: float=0):
        super().__init__()
        assert len(video_path_list) == len(audio_path_list) == len(prompt_list)
        self.video_path_list = video_path_list
        self.audio_path_list = audio_path_list
        self.prompt_list = prompt_list
        self.sample_rate = sample_rate

        # for Javis-Score
        self.window_size_s = window_size_s
        self.window_overlap_s = window_overlap_s

        self.frame_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(224),
                NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    
    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path, audio_path = self.video_path_list[index], self.audio_path_list[index]
        prompt = self.prompt_list[index]  # TODO: how to use textual clues?

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  unsafe
        # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        # frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.frame_transform(frame)
            video_frames.append(frame)
        video_frames = torch.stack(video_frames, dim=0)
        
        avh_inputs = {
            ModalityType.VISION: video_frames,
            ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data([audio_path], 'cpu'),
        }

        waveform, sr = torchaudio.load(audio_path)
        if self.sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.sample_rate
            )
        if len(waveform.shape) == 1:
            waveform = waveform[None]  # shape(1,N)
        
        video_windows, audio_clips = self.segment_clip_transform(video_frames, waveform, fps)
        video_windows_indices = torch.stack(
            [torch.arange(*video_window) for video_window in video_windows], dim=0
        )

        javis_inputs = {
            ModalityType.AUDIO: audio_clips,
        }

        return avh_inputs, javis_inputs, video_windows_indices

    def segment_clip_transform(self, frames, wavform, fps):
        video_window_size = int(self.window_size_s * fps)
        video_window_overlap = int(self.window_overlap_s * fps)
        video_windows = []
        n_vframes = len(frames)

        audio_window_size = int(self.window_size_s * self.sample_rate)
        audio_window_overlap = int(self.window_overlap_s * self.sample_rate)
        audio_clips = []
        n_aframes = wavform.shape[1]

        if n_vframes <= video_window_size or n_aframes <= audio_window_size:
            video_windows.append([0, n_vframes])
            audio_clips = self.transform_audio_clip(wavform)[None]
            return video_windows, audio_clips

        for start in range(0, n_vframes, video_window_size-video_window_overlap):
            start = min(start, n_vframes-video_window_size)
            end = start + video_window_size
            video_windows.append([start, end])
            if end >= n_vframes:
                break

        for start in range(0, n_aframes, audio_window_size-audio_window_overlap):
            start = min(start, n_aframes-audio_window_size)
            end = start + audio_window_size
            audio_clips.append(wavform[:, start:end])
            if end >= n_aframes:
                break

        clip_num = min(len(video_windows), len(audio_clips))
        video_windows, audio_clips = video_windows[:clip_num], audio_clips[:clip_num]
        audio_clips = [self.transform_audio_clip(clip) for clip in audio_clips]
        audio_clips = torch.stack(audio_clips)

        return video_windows, audio_clips

    def transform_audio_clip(
        self, waveform, num_mel_bins=128, target_length=204,
        clip_duration=2, clips_per_video=3, mean=-4.268, std=9.138,
    ):
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=clips_per_video
        )
        all_clips_timepoints = imagebind_data.get_clip_timepoints(
            clip_sampler, waveform.size(1) / self.sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * self.sample_rate) : 
                int(clip_timepoints[1] * self.sample_rate)
            ]
            waveform_melspec = imagebind_data.waveform2melspec(
                waveform_clip, self.sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac) for ac in all_clips]
        all_clips = torch.stack(all_clips, dim=0)

        return all_clips


class DeSyncDataset(Dataset):
    def __init__(self, video_path_list, audio_path_list, 
                 size=224, video_fps=25.0, audio_sr=16000, max_length_s=8.0):
        super().__init__()
        assert len(video_path_list) == len(audio_path_list)
        self.video_path_list = video_path_list
        self.audio_path_list = audio_path_list
        self.size = size
        self.video_fps = video_fps
        self.audio_sr = audio_sr
        self.max_length_s = max_length_s

        self.expected_video_length = int(video_fps * max_length_s)
        self.expected_audio_length = int(audio_sr * max_length_s)

        self.video_transform = transforms_v2.Compose([
            transforms_v2.Resize(self.size, interpolation=transforms_v2.InterpolationMode.BICUBIC),
            transforms_v2.CenterCrop(self.size),
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        video_path, audio_path = self.video_path_list[index], self.audio_path_list[index]

        ## Load Video    
        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=self.expected_video_length,
            frame_rate=self.video_fps,
            format='rgb24',
        )
        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        video = data_chunk[0]

        video = video[:self.expected_video_length]
        video = smart_pad(video, self.expected_video_length-video.shape[0], dim=0)
        # if video.shape[0] < self.expected_video_length:
        #     raise RuntimeError(
        #         f'Sync video too short {video_path}, '
        #         f'expected {self.expected_video_length}, '
        #         f'got {video.shape[0]}'
        #     )
        
        video = self.video_transform(video)

        ## Load Audio
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # mono

        if sample_rate != self.audio_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.audio_sr
            )

        audio = waveform[:self.expected_audio_length]
        audio = smart_pad(audio, self.expected_audio_length-audio.shape[0], dim=0)
        # if audio.shape[0] < self.expected_audio_length:
        #     raise RuntimeError(
        #         f'Sync audio too short {audio_path}, '
        #         f'expected {self.expected_audio_length}, '
        #         f'got {audio.shape[0]}'
        #     )

        return video, audio


def create_dataloader_for_fvd_mmdiff(video_list, temp_dir, fps, sr, clip_duration=1.6):
    from .mmdiff_datasets import load_data as load_mmdiff_data

    VIDEO_SIZE=[16,3,224,224]
    AUDIO_RATE=44100
    AUDIO_SIZE=[1, int(AUDIO_RATE*1.6)]
    BATCH_SIZE=32

    os.makedirs(temp_dir, exist_ok=True)
    video_size = [int(fps * clip_duration)] + VIDEO_SIZE[1:]
    audio_size = AUDIO_SIZE[:1] + [int(sr * clip_duration)]
    audio_fps = sr
    return load_mmdiff_data(
        all_files=video_list,
        data_dir=temp_dir,
        batch_size=BATCH_SIZE,
        video_size=video_size,
        audio_size=audio_size,
        num_workers=8,
        frame_gap=fps//10,
        random_flip=False,
        deterministic=True,
        drop_last=False,
        audio_fps=audio_fps,
        video_fps=fps,
    )


def create_dataloader_for_fvd_vanilla(
    video_list, audio_list, 
    max_frames, image_size, video_fps, audio_sr, max_audio_len_s,
    audio_only=False, num_workers=4, **kwargs
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ResizeAndPad(image_size),
    ])

    batch_size = kwargs.get('batch_size', 8)
    if max_audio_len_s is None and batch_size != 1:
        print('Warning: set batch size to 1 since `max_audio_len_s` is not given.')
        batch_size = 1

    dataset = FVDFADDataset(
        (video_list, audio_list), video_fps, audio_sr, 
        max_frames, max_audio_len_s, transform=transform, audio_only=audio_only
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


def create_dataloader(metric, batch_size=8, num_workers=8, **kwargs):
    if metric == 'video-quality':
        collate_fn = DataCollatorForVideoAlignDataset(
            processor=kwargs.pop('processor')
        )
    else:
        collate_fn = None

    if metric == 'clip-score':
        dataset = VideoFrameDataset(**kwargs)
    elif metric == 'clap-score' or metric == 'audio-quality':
        dataset = AudioDataset(**kwargs)
    elif metric == 'video-quality':
        dataset = VideoAlignDataset(**kwargs)
    elif metric == 'cavp-score':
        dataset = CavpDataset(**kwargs)
    elif metric == 'imagebind-score':
        dataset = ImageBindDataset(**kwargs)
    elif metric == 'av-align':
        dataset = AVAlignDataset(**kwargs)
    elif metric == 'av-score':
        dataset = AVScoreDataset(**kwargs)
    elif metric == 'desync-score':
        dataset = DeSyncDataset(**kwargs)

    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, drop_last=False,
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    return dataloader


@dataclass
class DataCollatorForVideoAlignDataset(object):
    processor: AutoProcessor

    def __call__(self, batch: Sequence[List]) -> Dict[str, torch.Tensor]:

        chat_data, video_inputs = zip(*batch)
        # [[a], [b]] -> [a, b]
        chat_data = [x for sublist in chat_data for x in sublist]
        video_inputs = [x for sublist in video_inputs for x in sublist]

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        return batch
