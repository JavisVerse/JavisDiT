import argparse
import os
import os.path as osp
import pandas as pd
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

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

from ..datasets.utils import IMG_EXTENSIONS, VID_EXTENSIONS
from javisdit.datasets.read_audio import read_audio

DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech."
)

CAP_PROMPT = (
    "Separately describe the visual and auditory content of the sounding video in detail. "
    "Your answer should be in a JSON format: {'visual content': 'xxx', 'auditory content': 'yyy'}. "
    "Note that the auditory content should NOT contain any visual elements, except for specific objects that make the sounds."
)

CONV_TMPL = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": DEFAULT_SYSTEM}
        ],
    },
    # {
    #     "role": "user",
    #     "content": [
    #         {"type": "image", "image": "/path/to/image.jpg"},
    #         {"type": "video", "video": "/path/to/video.mp4"},
    #         {"type": "audio", "audio": "/path/to/audio.wav"},
    #         {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
    #     ],
    # }
]

class AudioVideoTextDataset(Dataset):
    def __init__(self, samples, processor, data_args, use_audio_in_video=False):
        self.samples = samples
        self.processor = processor
        self.data_args = data_args
        self.use_audio_in_video = use_audio_in_video

    def __len__(self):
        return len(self.samples)

    def make_conversation(self, role="user", **content):
        conv = [{ "role": role, "content": [] }]
        for k, v in content.items():
            item = {"type": k, k: v}
            if k == 'video':
                item.update({
                    "max_frames": self.data_args.frames_upbound,
                    "max_pixels": self.data_args.frame_width * self.data_args.frame_height
                })
            conv[0]["content"].append(item)
        return conv

    def __getitem__(self, idx):
        sample = self.samples[idx]
            
        video_path  = sample['video_path']

        conversation = CONV_TMPL + self.make_conversation(video=video_path, text=CAP_PROMPT)

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        # assert self.use_audio_in_video is True
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)  # faster
        assert images is None and audios is None
        audios = [read_audio(video_path, backend='av', sr=16000)[0]]

        return {
            'path':        video_path,
            'text':        text,
            'audios':      audios, 
            'videos':      videos, 
        }


@dataclass
class DataCollatorForQwen25Omni(object):
    processor: Qwen2_5OmniProcessor
    use_audio_in_video: bool = False
    
    def __call__(self, instances: Union[Sequence[Dict]]) -> Dict[str, torch.Tensor]:
        content = {"text": [], "audios": [], "images": [], "videos": []}
        for instance in instances:
            for k in content.keys():
                content[k].extend(instance.pop(k, []))
        content['audio'] = content.pop('audios')
        invalid_keys = [k for k, v in content.items() if len(v) == 0]
        for k in invalid_keys:
            del content[k]

        inputs = self.processor(**content, return_tensors="pt", padding=True, use_audio_in_video=self.use_audio_in_video)

        batch = {"inputs": inputs}
        if len(instances[0]):
            assert len(set([len(ins) for ins in instances])) == 1  # all instances have the same keys
            for k in instances[0].keys():
                if k in batch:
                    continue
                batch[k] = [instance[k] for instance in instances]

        return batch


def build_dataloader(processor, samples, data_args=None, use_audio_in_video=False, **kwargs):
    dataset = AudioVideoTextDataset(samples, processor=processor, data_args=data_args, 
                                    use_audio_in_video=use_audio_in_video)
    data_collator = DataCollatorForQwen25Omni(processor=processor, use_audio_in_video=use_audio_in_video)
    dataloader = DataLoader(
        dataset, shuffle=False, 
        batch_size=kwargs.get("batch_size", 1), 
        num_workers=kwargs.get("num_workers", 8), 
        collate_fn=data_collator
    )

    return dataset, dataloader


# TODO: Data Parallel on Multi-GPU Inference
@torch.inference_mode()
def main(args):
    # Initialize the model
    dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name_or_path)

    setattr(args, 'processor', processor)
    
    df = pd.read_csv(args.input_file)
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'video_path': row['path'], 'id': row['id'],
        })
    
    use_audio_in_video = True
    _, dataloader = build_dataloader(
        args.dataset, processor, samples, data_args=args,
        use_audio_in_video=use_audio_in_video,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    ans_file = open(args.output_file, "w+")
    ans_writer = csv.writer(ans_file)
    ans_writer.writerow(["path", "text"])

    for i, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        inputs = batch.pop('inputs').to(model.device).to(model.dtype)
        paths = batch.pop("path")
        with torch.inference_mode():
            input_len = inputs['input_ids'].shape[1]
            text_ids = model.generate(
                **inputs, use_audio_in_video=use_audio_in_video, return_audio=False,
                do_sample=True, temperature=0.01, top_p=0.1, num_beams=1, 
                max_new_tokens=300,
            )
            output_texts = processor.batch_decode(text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
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
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--model_max_length", type=int, default=32768)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    # Process arguments
    parser.add_argument("--frames_upbound", type=int, default=16)
    parser.add_argument("--frame_width", type=int, default=588)
    parser.add_argument("--frame_height", type=int, default=336)

    args = parser.parse_args()
    main(args)
