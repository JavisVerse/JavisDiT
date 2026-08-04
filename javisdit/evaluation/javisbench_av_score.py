"""Standalone JavisBench AVHScore and JavisScore evaluation.

This module intentionally does not import the legacy ``eval.javisbench.main`` or
``eval.javisbench.src.metrics`` modules.  In particular, importing this module
does not change PyTorch's global gradient mode, mutate logging, initialize a
process group, or download model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


IMAGEBIND_CHECKPOINT_NAME = "imagebind_huge.pth"
IMAGEBIND_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
SAMPLE_RATE = 16000
_VISION = "vision"
_AUDIO = "audio"

__all__ = ["evaluate_av_score"]


def _validate_media_paths(paths: Sequence[str | os.PathLike[str]], name: str) -> list[str]:
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError(f"{name} must be a sequence of paths, not one path")

    try:
        path_list = list(paths)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of paths") from exc

    if not path_list:
        raise ValueError(f"{name} must not be empty")

    validated = []
    for index, value in enumerate(path_list):
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError(f"{name}[{index}] is not a path: {value!r}")
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{name}[{index}] does not exist: {path}")
        validated.append(str(path))
    return validated


def _resolve_checkpoint_path(checkpoint_or_cache: str | os.PathLike[str], allow_download: bool) -> Path:
    checkpoint = Path(checkpoint_or_cache).expanduser()
    if checkpoint.is_dir() or (not checkpoint.exists() and checkpoint.suffix == ""):
        checkpoint = checkpoint / IMAGEBIND_CHECKPOINT_NAME

    if checkpoint.is_file():
        return checkpoint
    if checkpoint.exists():
        raise ValueError(f"ImageBind checkpoint is not a file: {checkpoint}")
    if not allow_download:
        raise FileNotFoundError(
            f"ImageBind checkpoint not found: {checkpoint}. Pre-download it or set allow_download=True explicitly."
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(IMAGEBIND_CHECKPOINT_URL, str(checkpoint), progress=True)
    return checkpoint


def _load_imagebind_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    # Kept lazy so importing this API does not initialize the rest of JavisDiT's
    # model package or require ImageBind's optional runtime dependencies.
    from ._imagebind_runtime.models.imagebind_model import (
        imagebind_huge,
    )

    model_result = imagebind_huge(pretrained=False)
    model = model_result[0] if isinstance(model_result, tuple) else model_result
    try:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, Mapping) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def _waveform_to_melspec(
    waveform: torch.Tensor,
    sample_rate: int,
    num_mel_bins: int,
    target_length: int,
) -> torch.Tensor:
    import torchaudio

    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=10,
    ).transpose(0, 1)
    pad = target_length - fbank.size(1)
    if pad > 0:
        fbank = torch.nn.functional.pad(fbank, (0, pad))
    elif pad < 0:
        fbank = fbank[:, :target_length]
    return fbank.unsqueeze(0)


class _AVScoreDataset(Dataset):
    """Media preprocessing equivalent to JavisBench's AVScoreDataset."""

    def __init__(
        self,
        video_paths: Sequence[str],
        audio_paths: Sequence[str],
        window_size_s: float,
        window_overlap_s: float,
    ) -> None:
        from torchvision import transforms

        self.video_paths = video_paths
        self.audio_paths = audio_paths
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

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int):
        import cv2
        import torchaudio
        from PIL import Image

        video_path = self.video_paths[index]
        audio_path = self.audio_paths[index]

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = []
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.frame_transform(Image.fromarray(frame)))
        finally:
            capture.release()
        if not frames:
            raise ValueError(f"Video has no decodable frames: {video_path}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video has invalid FPS {fps}: {video_path}")
        video_frames = torch.stack(frames, dim=0)

        waveform, source_rate = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            raise ValueError(f"Audio has no samples: {audio_path}")
        if source_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=source_rate, new_freq=SAMPLE_RATE)

        video_windows, audio_clips = self._segment_clip_transform(video_frames, waveform, fps)
        video_window_indices = torch.stack([torch.arange(start, end) for start, end in video_windows], dim=0)
        avh_inputs = {
            _VISION: video_frames,
            _AUDIO: self._transform_audio_clip(waveform).unsqueeze(0),
        }
        javis_inputs = {_AUDIO: audio_clips}
        return avh_inputs, javis_inputs, video_window_indices

    def _segment_clip_transform(
        self, frames: torch.Tensor, waveform: torch.Tensor, fps: float
    ) -> tuple[list[tuple[int, int]], torch.Tensor]:
        video_window_size = int(self.window_size_s * fps)
        video_window_overlap = int(self.window_overlap_s * fps)
        audio_window_size = int(self.window_size_s * SAMPLE_RATE)
        audio_window_overlap = int(self.window_overlap_s * SAMPLE_RATE)
        if video_window_size <= 0 or audio_window_size <= 0:
            raise ValueError("Window size is too small for the media sampling rate")

        num_video_frames = len(frames)
        num_audio_frames = waveform.shape[1]
        if num_video_frames <= video_window_size or num_audio_frames <= audio_window_size:
            audio_clips = self._transform_audio_clip(waveform).unsqueeze(0)
            return [(0, num_video_frames)], audio_clips

        video_step = video_window_size - video_window_overlap
        audio_step = audio_window_size - audio_window_overlap
        if video_step <= 0 or audio_step <= 0:
            raise ValueError("Window overlap is too large after conversion to media samples")
        video_windows = []
        for start in range(0, num_video_frames, video_step):
            start = min(start, num_video_frames - video_window_size)
            video_windows.append((start, start + video_window_size))
            if start + video_window_size >= num_video_frames:
                break

        audio_windows = []
        for start in range(0, num_audio_frames, audio_step):
            start = min(start, num_audio_frames - audio_window_size)
            audio_windows.append(waveform[:, start : start + audio_window_size])
            if start + audio_window_size >= num_audio_frames:
                break

        clip_count = min(len(video_windows), len(audio_windows))
        video_windows = video_windows[:clip_count]
        audio_clips = torch.stack([self._transform_audio_clip(clip) for clip in audio_windows[:clip_count]])
        return video_windows, audio_clips

    def _transform_audio_clip(
        self,
        waveform: torch.Tensor,
        num_mel_bins: int = 128,
        target_length: int = 204,
        clip_duration: float = 2.0,
        clips_per_video: int = 3,
        mean: float = -4.268,
        std: float = 9.138,
    ) -> torch.Tensor:
        from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler

        sampler = ConstantClipsPerVideoSampler(clip_duration=clip_duration, clips_per_video=clips_per_video)
        timepoints = []
        last_end = 0.0
        is_last_clip = False
        duration = waveform.size(1) / SAMPLE_RATE
        while not is_last_clip:
            start, end, _, _, is_last_clip = sampler(last_end, duration, annotation=None)
            timepoints.append((start, end))
            last_end = end

        clips = []
        for start, end in timepoints:
            waveform_clip = waveform[:, int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
            melspec = _waveform_to_melspec(waveform_clip, SAMPLE_RATE, num_mel_bins, target_length)
            clips.append((melspec - mean) / std)
        return torch.stack(clips, dim=0)


@torch.no_grad()
def evaluate_av_score(
    video_paths: Sequence[str | os.PathLike[str]],
    audio_paths: Sequence[str | os.PathLike[str]],
    *,
    imagebind_checkpoint: str | os.PathLike[str],
    device: str | torch.device = "cuda",
    window_size_s: float = 2.0,
    window_overlap_s: float = 1.5,
    topk_min: int | float = 0.4,
    num_workers: int = 0,
    allow_download: bool = False,
) -> dict[str, float]:
    """Evaluate AVHScore and JavisScore for paired generated media.

    ``imagebind_checkpoint`` must point to ``imagebind_huge.pth`` or to an
    explicit cache directory containing it. Missing weights are never fetched
    unless ``allow_download=True`` is provided explicitly. Audio is resampled
    to JavisBench's fixed 16 kHz rate.
    """

    videos = _validate_media_paths(video_paths, "video_paths")
    audios = _validate_media_paths(audio_paths, "audio_paths")
    if len(videos) != len(audios):
        raise ValueError(f"video_paths and audio_paths must have equal length; got {len(videos)} and {len(audios)}")
    if not math.isfinite(window_size_s) or window_size_s <= 0:
        raise ValueError("window_size_s must be finite and greater than zero")
    if not math.isfinite(window_overlap_s) or window_overlap_s < 0 or window_overlap_s >= window_size_s:
        raise ValueError("window_overlap_s must be finite, non-negative, and smaller than window_size_s")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if isinstance(topk_min, bool) or not isinstance(topk_min, (int, float)):
        raise TypeError("topk_min must be an int or float")
    if isinstance(topk_min, int):
        if topk_min <= 0:
            raise ValueError("integer topk_min must be greater than zero")
    elif not math.isfinite(topk_min) or not 0 < topk_min <= 1:
        raise ValueError("float topk_min must be in the interval (0, 1]")

    checkpoint = _resolve_checkpoint_path(imagebind_checkpoint, allow_download)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    model = _load_imagebind_model(checkpoint, resolved_device)

    dataset = _AVScoreDataset(
        videos,
        audios,
        window_size_s=window_size_s,
        window_overlap_s=window_overlap_s,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    avh_scores = []
    javis_scores = []
    for avh_inputs, javis_inputs, video_window_indices in loader:
        avh_inputs = {key: value[0].to(resolved_device) for key, value in avh_inputs.items()}
        javis_inputs = {key: value[0].to(resolved_device) for key, value in javis_inputs.items()}
        video_window_indices = video_window_indices[0]

        embeddings = model(avh_inputs)
        frame_embeddings = embeddings[_VISION]
        audio_embedding = embeddings[_AUDIO]
        avh_scores.append(cosine(frame_embeddings, audio_embedding).mean().item())

        window_count, frames_per_window = video_window_indices.shape[:2]
        embeddings = model(javis_inputs)
        video_embeddings = frame_embeddings[video_window_indices.flatten()].view(window_count, frames_per_window, -1)
        audio_embeddings = embeddings[_AUDIO].unsqueeze(1)
        window_similarities = cosine(video_embeddings, audio_embeddings)
        k = topk_min if isinstance(topk_min, int) else max(1, int(frames_per_window * topk_min))
        if k < 1 or k > frames_per_window:
            raise ValueError(f"topk_min selects {k} frames, but each window has {frames_per_window} frames")
        lowest_similarities = torch.topk(
            window_similarities,
            k,
            dim=1,
            largest=False,
            sorted=False,
        ).values
        javis_scores.append(lowest_similarities.mean(dim=1).mean().item())

    result = {
        "avh_score": torch.tensor(avh_scores).mean().item(),
        "javis_score": torch.tensor(javis_scores).mean().item(),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError(f"AV-score evaluation produced non-finite results: {result}")
    return result


def _load_manifest(path: str | os.PathLike[str]) -> tuple[list[str], list[str]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping) and "video_paths" in payload:
            if "audio_paths" not in payload:
                raise ValueError("JSON manifest is missing audio_paths")
            return list(payload["video_paths"]), list(payload["audio_paths"])
        if not isinstance(payload, list):
            raise ValueError("JSON manifest must be a list of records or contain video_paths/audio_paths")
        rows = payload
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError("Manifest must have a .json or .csv extension")

    videos, audios = [], []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Manifest row {index} must be an object")
        video = row.get("video_path") or row.get("pred_video_path")
        audio = row.get("audio_path") or row.get("pred_audio_path")
        if not video or not audio:
            raise ValueError(f"Manifest row {index} needs video_path/audio_path or pred_video_path/pred_audio_path")
        videos.append(video)
        audios.append(audio)
    return videos, audios


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        required=True,
        help="CSV or JSON manifest containing paired video/audio paths",
    )
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--imagebind-checkpoint",
        required=True,
        help="ImageBind checkpoint file or explicit cache directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size-s", type=float, default=2.0)
    parser.add_argument("--window-overlap-s", type=float, default=1.5)
    parser.add_argument("--topk-min", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    video_paths, audio_paths = _load_manifest(args.input_file)
    result = evaluate_av_score(
        video_paths,
        audio_paths,
        imagebind_checkpoint=args.imagebind_checkpoint,
        device=args.device,
        window_size_s=args.window_size_s,
        window_overlap_s=args.window_overlap_s,
        topk_min=args.topk_min,
        num_workers=args.num_workers,
        allow_download=args.allow_download,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
