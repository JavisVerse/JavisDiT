import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Literal, Any, Dict, List, Optional, Tuple, Union

import av
import cv2
from decord import VideoReader, cpu

import numpy as np
import torch
from torchvision import get_video_backend, transforms
from torchvision.io.video import _check_av_available
from torchvision.transforms import InterpolationMode

from javisdit.datasets.utils import generate_temporal_window

MAX_NUM_FRAMES = 2500


def read_video_av(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    This method is modified from torchvision.io.video.read_video, with the following changes:

    1. will not extract audio frames and return empty for aframes
    2. remove checks and only support pyav
    3. add container.close() and gc.collect() to avoid thread leakage
    4. try our best to avoid memory leak

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    # format
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    # file existence
    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    # backend check
    assert get_video_backend() == "pyav", "pyav backend is required for read_video_av"
    _check_av_available()
    # end_pts check
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    # == get video info ==
    info = {}
    # TODO: creating an container leads to memory leak (1G for 8 workers 1 GPU)
    container = av.open(filename, metadata_errors="ignore")
    # fps
    video_fps = container.streams.video[0].average_rate
    # guard against potentially corrupted files
    if video_fps is not None:
        info["video_fps"] = float(video_fps)
    iter_video = container.decode(**{"video": 0})
    frame = next(iter_video).to_rgb().to_ndarray()
    height, width = frame.shape[:2]
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = MAX_NUM_FRAMES
        warnings.warn(f"total_frames is 0, using {MAX_NUM_FRAMES} as a fallback")
    container.close()
    del container

    # HACK: must create before iterating stream
    # use np.zeros will not actually allocate memory
    # use np.ones will lead to a little memory leak
    video_frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)

    # == read ==
    try:
        # TODO: The reading has memory leak (4G for 8 workers 1 GPU)
        container = av.open(filename, metadata_errors="ignore")
        assert container.streams.video is not None
        video_frames = _read_from_stream(
            video_frames,
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
            filename=filename,
        )
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    vframes = torch.from_numpy(video_frames).clone()
    del video_frames
    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    aframes = torch.empty((1, 0), dtype=torch.float32)
    return vframes, aframes, info


def _read_from_stream(
    video_frames,
    container: "av.container.Container",
    start_offset: float,
    end_offset: float,
    pts_unit: str,
    stream: "av.stream.Stream",
    stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
    filename: Optional[str] = None,
) -> List["av.frame.Frame"]:
    if pts_unit == "sec":
        # TODO: we should change all of this from ground up to simply take
        # sec and convert to MS in C++
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX-style packed B-frames can have out-of-order pts (2 frames in a single pkt)
        # so need to buffer some extra frames to sort everything
        # properly
        extradata = stream.codec_context.extradata
        # overly complicated way of finding if `divx_packed` is set, following
        # https://github.com/FFmpeg/FFmpeg/commit/d5a21172283572af587b3d939eba0091484d3263
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    # some files don't seek to the right location, so better be safe here
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        # FIXME this is kind of a hack, but we will jump to the previous keyframe
        # so this will be safe
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        # TODO check if stream needs to always be the video stream here or not
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError as e:
        print(f"[Warning] Error while seeking video {filename}: {e}")
        return []

    # == main ==
    buffer_count = 0
    frames_pts = []
    cnt = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames_pts.append(frame.pts)
            video_frames[cnt] = frame.to_rgb().to_ndarray()
            cnt += 1
            if cnt >= len(video_frames):
                break
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    # garbage collection for thread leakage
    container.close()
    del container
    # NOTE: manually garbage collect to close pyav threads
    gc.collect()

    # ensure that the results are sorted wrt the pts
    # NOTE: here we assert frames_pts is sorted
    start_ptr = 0
    end_ptr = cnt
    while start_ptr < end_ptr and frames_pts[start_ptr] < start_offset:
        start_ptr += 1
    while start_ptr < end_ptr and frames_pts[end_ptr - 1] > end_offset:
        end_ptr -= 1
    if start_offset > 0 and start_offset not in frames_pts[start_ptr:end_ptr]:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        if start_ptr > 0:
            start_ptr -= 1
    result = video_frames[start_ptr:end_ptr].copy()
    return result


def read_video_cv2(video_path, num_frames=None, frame_interval=1, fix_start_frame=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        # print("Error: Unable to open video")
        raise ValueError(f"Error: Unable to open video {video_path}")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        vinfo = {
            "video_fps": fps,
        }

        if num_frames is not None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # unsafe but faster
            # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            # total_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
            minimal_length = (num_frames-1) * frame_interval + 1
            if fix_start_frame is None:
                assert total_frames > minimal_length, f'Ensure video {video_path} has enough frames: {total_frames} < {minimal_length}'
                start_frame_ind, end_frame_ind = generate_temporal_window(total_frames, num_frames, frame_interval)
            else:
                start_frame_ind = min(fix_start_frame, max(total_frames-minimal_length, 0))
                end_frame_ind = min(start_frame_ind + minimal_length, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_ind)
            vinfo.update({'total_frames': total_frames, 
                          'start_frame_ind': start_frame_ind, 'end_frame_ind': end_frame_ind})

        frames = []
        while True:
            sucess, frame = cap.read()

            if not sucess:
                break

            frames.append(frame[:, :, ::-1])  # BGR to RGB

            if num_frames and len(frames) >= minimal_length:
                break

        cap.release()

        frames = frames[::frame_interval]
        cur_frames = len(frames)
        if fix_start_frame is None:
            assert cur_frames == num_frames, f'loaded {cur_frames}/{num_frames} frames from {video_path}'
        else:
            if cur_frames < num_frames:
                # padding the last frame
                for _ in range(num_frames - cur_frames):
                    frames.append(frames[-1])

        frames = np.stack(frames)
        frames = torch.from_numpy(frames)  # [T, H, W, C=3]
        frames = frames.permute(0, 3, 1, 2)
        return frames, vinfo


def read_video_decord(
    video_file, output_format:Literal["TCHW", "THWC"]="TCHW",
    max_length_sec_or_frame=None, start_sec_or_frame=0, end_sec_or_frame=None, 
    frame_width=-1, frame_height=-1, fast_resize=True,
    frames_sample_fps=None, frames_sample_num=None, frame_stride=2
):
    # TODO: support dynamic fps
    if not fast_resize:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1, width=frame_width, height=frame_height)

    raw_video_fps = vr.get_avg_fps()
    total_frame_num = len(vr)
    video_time = total_frame_num / raw_video_fps

    for flag in [max_length_sec_or_frame, start_sec_or_frame, end_sec_or_frame]:
        assert flag is None or type(flag) in [int, float], \
            f'Unrecognizable data format for {flag=}, type={type(end_sec_or_frame)}'
    # start sample frame
    if isinstance(start_sec_or_frame, float):  # second
        start_frame_idx = int(start_sec_or_frame * raw_video_fps)
    elif isinstance(start_sec_or_frame, int):  # frame
        assert start_sec_or_frame <= total_frame_num
        start_frame_idx = start_sec_or_frame
    # end sample frame
    if max_length_sec_or_frame is not None and max_length_sec_or_frame > 0:
        if isinstance(max_length_sec_or_frame, float):  # second
            max_length_frame = int(round(max_length_sec_or_frame * raw_video_fps))
        elif isinstance(max_length_sec_or_frame, int):  # frame
            max_length_frame = max_length_sec_or_frame
        total_frame_num = min(total_frame_num, start_frame_idx + max_length_frame)
    if isinstance(end_sec_or_frame, float):  # second
        total_frame_num = min(int(end_sec_or_frame * raw_video_fps), total_frame_num)
    elif isinstance(end_sec_or_frame, int):  # frame
        assert start_frame_idx <= end_sec_or_frame <= total_frame_num
        total_frame_num = min(end_sec_or_frame, total_frame_num)
    
    assert frames_sample_fps is None or frames_sample_num is None, \
        'Concurrently defining sampling frame number and fps is not supported'
    if frames_sample_fps is not None:
        avg_fps = round(raw_video_fps / frames_sample_fps)
        frame_idx = [i for i in range(start_frame_idx, total_frame_num, avg_fps)]
    elif frames_sample_num is not None:
        if frame_stride > 0 and frames_sample_num % frame_stride != 0:
            frames_sample_num = int(np.ceil(frames_sample_num / frame_stride)) * frame_stride
        frame_idx = torch.linspace(start_frame_idx, total_frame_num - 1, frames_sample_num).round().long().tolist()
    else:
        frame_idx = torch.arange(start_frame_idx, total_frame_num).tolist()
            
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = [i/raw_video_fps for i in frame_idx]
    # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)

    sample_fps = num_frames / (total_frame_num-start_frame_idx+1) * raw_video_fps

    if output_format == "TCHW":
        video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    if not fast_resize:
        assert output_format == "TCHW"
        video = transforms.functional.resize(
            video,
            [frame_height, frame_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

    vinfo = {
        'total_frames': num_frames_to_sample, 'video_fps': sample_fps,
        # TODO: check the semantic meaning
        # 'start_frame_ind': start_frame_idx, 'end_frame_ind': total_frame_num
    }

    return video, vinfo


def read_video(video_path, backend="av", **kwargs):
    if kwargs.get('num_frames'):
        assert backend == "cv2", "Currently only `cv2` mode support direct reading a clip"
    if backend == "cv2":
        vframes, vinfo = read_video_cv2(video_path, **kwargs)
    elif backend == "av":
        vframes, _, vinfo = read_video_av(filename=video_path, pts_unit="sec", 
                                          output_format="TCHW", **kwargs)
    elif backend == "decord":
        vframes, vinfo = read_video_decord(video_path, **kwargs)
    else:
        raise ValueError

    vinfo["video_fps"] = float(round(vinfo["video_fps"]))
    return vframes, vinfo
