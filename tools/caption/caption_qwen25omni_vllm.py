import argparse
import os
import os.path as osp
import pandas as pd
import numpy as np
import json
import csv
import time
import warnings
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Union
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import vllm.envs as envs
from vllm import LLM, SamplingParams
from qwen_omni_utils import fetch_video

from ..datasets.utils import IMG_EXTENSIONS, VID_EXTENSIONS
from javisdit.datasets.read_audio import read_audio
from .caption_qwen25omni import DEFAULT_SYSTEM, CAP_PROMPT


PROMPT_TMPL = (
    f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
    "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
    f"{CAP_PROMPT}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)


class AudioVideoTextVLLMDataset(Dataset):
    def __init__(self, samples, data_args):
        self.samples = samples
        self.data_args = data_args

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
            
        video_path = sample['video_path']

        ele = {
            "type": "video", 
            "video": video_path,
            "max_frames": self.data_args.frames_upbound,
            "max_pixels": self.data_args.frame_width * self.data_args.frame_height 
        }
        video = fetch_video(ele, return_video_sample_fps=False)
        video = video.permute(0, 2, 3, 1).contiguous().numpy()  # TCHW -> THWC

        try:
            audio = read_audio(video_path, backend='av', sr=self.data_args.audio_sr)[0]
            assert len(audio) > self.data_args.audio_sr * 0.1  # 0.1s
        except:  
            # TODO: record failed
            print('empty audio:', video_path, flush=True)
            audio = np.zeros(self.data_args.audio_sr * 1, )

        assert not envs.VLLM_USE_V1, (
            "V1 does not support use_audio_in_video. "
            "Please launch this example with "
            "`VLLM_USE_V1=0`."
        )

        return {
            'path':   video_path,
            'inputs': {
                "prompt": PROMPT_TMPL,
                "multi_modal_data": {
                    "video": video,
                    "audio": audio,
                },
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                },
            }
        }


@dataclass
class DataCollatorForQwen25OmniVLLM(object):
    def __call__(self, instances: Union[Sequence[Dict]]) -> Dict[str, torch.Tensor]:
        batch = {
            k: [instance[k] for instance in instances] for k in instances[0].keys()
        }

        return batch


def build_dataloader(samples, data_args=None, **kwargs):
    dataset = AudioVideoTextVLLMDataset(samples, data_args=data_args)
    data_collator = DataCollatorForQwen25OmniVLLM()
    dataloader = DataLoader(
        dataset, shuffle=False, 
        batch_size=kwargs.get("batch_size", 1), 
        num_workers=kwargs.get("num_workers", 8), 
        collate_fn=data_collator
    )

    return dataset, dataloader


def main(args):
    # Initialize the model
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=args.model_max_length,
        # max_num_seqs=5,
        # gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"video": args.batch_size, "audio": args.batch_size},
        seed=args.seed,
    )

    # We set temperature to 0.2 so that outputs can be different
    # even when all prompts are identical when running batch inference.
    sampling_params = SamplingParams(temperature=0.01, top_p=0.1, max_tokens=300)

    if osp.exists(args.output_file):
        df_exist = pd.read_csv(args.output_file)
        res_exist = set(df_exist['path'].tolist())
    else:
        df_exist = None
        res_exist = set([])

    df = pd.read_csv(args.input_file)
    df = df.loc[args.part_idx::args.part_num]  # partial
    samples = []
    for _, row in df.iterrows():
        if row['path'] in res_exist:
            continue
        samples.append({
            'video_path': row['path'], 'id': row['id'],
        })
    
    _, dataloader = build_dataloader(
        samples, data_args=args,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    ans_file = open(args.output_file, "w+")
    ans_writer = csv.writer(ans_file)
    ans_writer.writerow(["path", "text"])

    if df_exist is not None:
        for _, row in df_exist.iterrows():
            ans_writer.writerow([row["path"], row["text"]])

    for i, batch in enumerate(tqdm(dataloader)):
        outputs = llm.generate(
            batch.pop('inputs'), 
            sampling_params=sampling_params,
            use_tqdm=False
        )
        output_texts = [o.outputs[0].text for o in outputs]
        
        paths = batch.pop("path")
        for path, res in zip(paths, output_texts):
            ans_writer.writerow([path, res])
            
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Configuration
    parser.add_argument('--input-file', help='Path to the file containing inputs.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results in csv.', required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=1)
    parser.add_argument("--num-workers", type=int, required=False, default=None)
    # Model arguments
    parser.add_argument('--model_name_or_path', help='Huggingface model name or local path', required=True)
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=None)
    # Process arguments
    parser.add_argument("--frames_upbound", type=int, default=16)
    parser.add_argument("--frame_width", type=int, default=588)
    parser.add_argument("--frame_height", type=int, default=336)
    parser.add_argument("--audio_sr", type=int, default=16000)
    # Partial arguments
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=0)

    args = parser.parse_args()
    main(args)
