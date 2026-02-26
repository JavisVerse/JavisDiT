from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import os.path as osp
import shutil
import random
import math
from typing import List, Dict, Optional

from PIL import Image
import cv2

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoProcessor, ClapModel
import torchaudio

from .fvd.fvd import get_fvd_logits, frechet_distance
from .fvd.download import load_i3d_pretrained
from .AudioCLIP.get_embedding import load_audioclip_pretrained, get_audioclip_embeddings_scores
from .utils import polynomial_mmd, Extract_CAVP_Features, pad_or_truncate, smart_pad
from .synchformer.synchformer import Synchformer, make_class_grid
from .VideoAlign.inference import VideoVLMRewardInference
from audiobox_aesthetics.infer import initialize_predictor as initialize_audio_aes_predictor

sys.path.append(os.path.join(os.path.dirname(__file__), "ImageBind"))
from .ImageBind.imagebind import data as imagebind_data
from .ImageBind.imagebind.models import imagebind_model
from .ImageBind.imagebind.models.imagebind_model import ModalityType

from .dataset import (
    create_dataloader, 
    create_dataloader_for_fvd_vanilla, create_dataloader_for_fvd_mmdiff
)

def calc_fvd_kvd_fad_direct(embed_dict, indices=None):
    fake_video_embed = embed_dict['fake']['video'].clone()
    real_video_embed = embed_dict['real']['video'].clone()
    fake_audio_embed = embed_dict['fake']['audio'].clone()
    real_audio_embed = embed_dict['real']['audio'].clone()

    if indices is not None:
        fake_video_embed = fake_video_embed[indices]
        real_video_embed = real_video_embed[indices]
        fake_audio_embed = fake_audio_embed[indices]
        real_audio_embed = real_audio_embed[indices]

    fvd = frechet_distance(fake_video_embed, real_video_embed).item()
    kvd = polynomial_mmd(fake_video_embed.cpu().numpy(), real_video_embed.cpu().numpy()).item()
    fad = frechet_distance(fake_audio_embed, real_audio_embed).item() * 10000

    return fvd, kvd, fad


def calc_fvd_kvd_fad(
    gt_video_list, pred_video_list, gt_audio_list, pred_audio_list, device="cuda:0", mode="vanilla", **kwargs
):
    # Original code from "https://github.com/researchmm/MM-Diffusion"
    
    fvd_avcache_path = kwargs.get('fvd_avcache_path', None)
    if fvd_avcache_path is not None:
        gt_embed_dict = torch.load(fvd_avcache_path)
        gt_embed_dict = {k: v.to(device) for k, v in gt_embed_dict.items()}
        use_gt_av_cache = True
        print(f'Ground-truth AV cache loaded from {fvd_avcache_path}')
    else:
        gt_embed_dict = None
        use_gt_av_cache = False

    if mode == 'mmdiffusion':
        cache_dir = kwargs.get('cache_dir')
        fps = kwargs.get('video_fps', 24)
        sr = kwargs.get('audio_sr', 16000)
        ## assume audios are integrated into videos
        if not use_gt_av_cache:
            real_loader = create_dataloader_for_fvd_mmdiff(gt_video_list, f"{cache_dir}/mmdiff/real", fps, sr)
        fake_loader = create_dataloader_for_fvd_mmdiff(pred_video_list, f"{cache_dir}/mmdiff/fake", fps, sr)
    else:
        if not use_gt_av_cache:
            real_loader = create_dataloader_for_fvd_vanilla(gt_video_list, gt_audio_list, **kwargs)
        fake_loader = create_dataloader_for_fvd_vanilla(pred_video_list, pred_audio_list, **kwargs)

    # load models
    i3d = load_i3d_pretrained(device)
    audioclip = load_audioclip_pretrained(device)

    if not use_gt_av_cache:
        loader_dict = {'real': real_loader, 'fake': fake_loader}
        embed_dict = {}
    else:
        loader_dict = {'fake': fake_loader}
        embed_dict = {'real': gt_embed_dict}

    for t, loader in loader_dict.items():
        video_embeds, audio_embeds = [], []
        for _, sample in enumerate(tqdm(loader, desc=f'fvd_kvd_fad - {t}')):
            # b t h w c
            video_sample = sample['video'].to(device)
            audio_sample = sample['audio'].to(device)

            video_embed = get_fvd_logits(video_sample, i3d, device=device)
            video_embeds.append(video_embed)

            _, audioclip_audio_embed, _ = get_audioclip_embeddings_scores(audioclip, video_sample, audio_sample)
            audio_embeds.append(audioclip_audio_embed)

        video_embeds = torch.cat(video_embeds)
        audio_embeds = torch.cat(audio_embeds)

        embed_dict[t] = {'video': video_embeds, 'audio': audio_embeds}
    
    sample_num = min(len(embed_dict['fake']['video']), len(embed_dict['real']['video']))
    fvd, kvd, fad = calc_fvd_kvd_fad_direct(embed_dict, list(range(sample_num)))

    return fvd, kvd, fad, embed_dict


def calc_video_quality_score(pred_video_list, prompt_list, device='cuda:0', bs=8, 
                             load_from_pretrained='./checkpoints/VideoReward'):
    # Original code from "https://github.com/KwaiVGI/VideoAlign"

    # weights will be automatically downloaded from huggingface
    predictor = VideoVLMRewardInference(
        load_from_pretrained=load_from_pretrained,
        device=device
    )

    pred_video_list = [osp.abspath(path) for path in pred_video_list]
    dataloader = create_dataloader(
        metric='video-quality', 
        video_path_list=pred_video_list,
        prompt_list=prompt_list,
        data_config=predictor.data_config,
        processor=predictor.processor,
        batch_size=bs,
    )

    visual_quality_score_list, motion_quality_score_list = [], []
    for batch in tqdm(dataloader, desc='video-quality'):
        batch = {k: v.to(predictor.device) if isinstance(v, torch.Tensor) else v \
                  for k, v in batch.items()}
        outputs = predictor.model(return_dict=True, **batch)["logits"]
        outputs = predictor.post_process(outputs, use_norm=False)

        visual_quality_scores = [r['VQ'] for r in outputs]
        motion_quality_scores = [r['MQ'] for r in outputs]

        visual_quality_score_list.extend(visual_quality_scores)
        motion_quality_score_list.extend(motion_quality_scores)

    visual_quality_scores = torch.tensor(visual_quality_score_list)
    motion_quality_scores = torch.tensor(motion_quality_score_list)

    return visual_quality_scores, motion_quality_scores


def calc_audio_quality_score(pred_audio_list, prompt_list, max_audio_len_s=8.0,
                             device='cuda:0', bs=8, audio_sr=16000):
    # Original code from "https://github.com/facebookresearch/audiobox-aesthetics"

    # weights will be automatically downloaded from huggingface
    predictor = initialize_audio_aes_predictor()
    predictor.model = predictor.model.to(device)
    predictor.device = device

    dataloader = create_dataloader(
        metric='audio-quality', 
        audio_path_list=pred_audio_list,
        prompt_list=prompt_list,
        sr=audio_sr,
        backend="torchaudio",
        mono=True,
        keepdim=True,
        norm=False,
        resample=True,
        max_audio_len_s=max_audio_len_s,
        batch_size=bs,
    )

    score_list = []
    for audios, prompts in tqdm(dataloader, desc='audio-quality'):
        batch = [{"path": wav, "sample_rate": audio_sr} for wav in audios]
        outputs = predictor.forward(batch)

        scores = torch.tensor([np.mean([list(o.values())]) for o in outputs])
        score_list.append(scores)
    
    audio_quality_scores = torch.cat(score_list)

    return audio_quality_scores


def calc_imagebind_score(video_list, audio_list, prompt_list, audio_prompt_list=None,
                         device='cuda:0', bs=1):
    # Original code from "https://github.com/sonyresearch/svg_baseline"

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    if audio_prompt_list is None:
        audio_prompt_list = prompt_list

    dataloader = create_dataloader(
        metric='imagebind-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        audio_prompt_list=audio_prompt_list,
        batch_size=8,
    )

    sim_tv_list, sim_ta_list, sim_av_list = [], [], []
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for inputs in tqdm(dataloader, desc='imagebind-score'):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # [B, v+a, 77] -> [v+a, B, 77] -> [v*B+a*B, 77]
        inputs[ModalityType.TEXT] = inputs[ModalityType.TEXT].transpose(0, 1).flatten(0, 1)

        with torch.no_grad():
            embeddings = model(inputs)

        text_embed, audio_text_embed = embeddings[ModalityType.TEXT].chunk(2, dim=0)
        video_embed, audio_embed = embeddings[ModalityType.VISION], embeddings[ModalityType.AUDIO]
        sim_tv_list.append(cos(text_embed, video_embed).cpu())
        sim_ta_list.append(cos(audio_text_embed, audio_embed).cpu())
        sim_av_list.append(cos(audio_embed, video_embed).cpu())

    sim_tv_scores, sim_ta_scores, sim_av_scores = \
        torch.cat(sim_tv_list), torch.cat(sim_ta_list), torch.cat(sim_av_list)

    return sim_tv_scores, sim_ta_scores, sim_av_scores


def calc_clip_score(video_list, prompt_list, device='cuda:0', num_frames=48):
    # load clip model
    device = device
    model, preprocess = clip.load("ViT-B/32", device=device)

    def _frame_transform(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        return frame

    dataloader = create_dataloader(
        metric='clip-score', 
        video_path_list=video_list,
        prompt_list=prompt_list,
        num_frames=num_frames,
        frame_transform=_frame_transform,
        batch_size=1
    )

    clip_score_list = []
    for frames, prompts in tqdm(dataloader, desc='clipscore'):
        assert frames.shape[0] == len(prompts) == 1  
        frames = frames.to(device)

        with torch.no_grad():
            text = clip.tokenize(prompts, truncate=True).to(device)
            text_features = model.encode_text(text)
            image_features = model.encode_image(frames[0])
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).mean()
            clip_score_list.append(score.item())
    
    clip_scores = torch.tensor(clip_score_list)

    return clip_scores


def calc_clap_score(audio_list, prompt_list, device='cuda:0'):
    # Original code from "https://github.com/sonyresearch/svg_baseline"
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.to(device=device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    dataloader = create_dataloader(
        metric='clap-score', 
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sr=48000,   # CLAP requires sample_rate=48000
        max_audio_len_s=None,
        batch_size=1
    )

    score_list = []
    for audios, prompts in tqdm(dataloader, desc='clapscore'):
        assert len(audios) == len(prompts) == 1
        inputs = processor(text=prompts[0], audios=audios[0].squeeze(), 
                           return_tensors="pt", padding=True, 
                           sampling_rate=48000)   # CLAP requires sample_rate=48000
        inputs.to(device=device)
        outputs = model(**inputs)
        scores = cos(outputs.text_embeds, outputs.audio_embeds).mean()
        score_list.append(scores)
    
    clap_scores = torch.tensor(score_list)

    return clap_scores


def calc_cavp_score(video_list, audio_list, device='cuda:0', sample_rate=16000,
                    cavp_ckpt_path='./checkpoints/cavp_epoch66.ckpt',
                    cavp_config_path='./configs/Stage1_CAVP.yaml'):
    # Original code from "https://github.com/sonyresearch/svg_baseline"

    fps = 4  #  CAVP default FPS=4, Don't change it.
    batch_size = 40  # Don't change it.

    # Initalize CAVP Model:
    extract_cavp = Extract_CAVP_Features(fps=fps,
                                        batch_size=batch_size,
                                        device=device,
                                        config_path=cavp_config_path,
                                        ckpt_path=cavp_ckpt_path)
    dataloader = create_dataloader(
        metric='cavp-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        sr=sample_rate,
        batch_size=1
    )

    tmp_path = "./tmp"
    score_list = []
    for video_paths, audios, truncate_seconds in tqdm(dataloader, desc='cavpscore'):
        assert len(video_paths) == len(audios) == len(truncate_seconds) == 1

        # Extract Video CAVP Features & New Video Path:
        try:  # TODO: debug
            cavp_feats, new_video_path = \
                extract_cavp(video_paths[0], 0, truncate_seconds[0].item(), tmp_path=tmp_path)
        except:
            score_list.append(0.0)
            continue

        spec = audios.unsqueeze(1).to(device).float()  # B x 1 x Mel x T
        spec = spec.permute(0, 1, 3, 2)  # B x 1 x T x Mel
        spec_feat = extract_cavp.stage1_model.spec_encoder(spec)  # B x T x C
        spec_feat = extract_cavp.stage1_model.spec_project_head(
            spec_feat).squeeze()
        spec_feat = F.normalize(spec_feat, dim=-1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        score = cos(torch.from_numpy(cavp_feats).to(device), spec_feat).mean().item()
        
        score_list.append(score)
    
    cavp_scores = torch.tensor(score_list)

    return cavp_scores


def calc_av_align(video_list, audio_list, size=None):
    # Original code from "https://yzxing87.github.io/Seeing-and-Hearing/"

    dataloader = create_dataloader(
        metric='av-align', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        size=size,
        batch_size=1
    )

    align_score_list = []
    for align_score in tqdm(dataloader, desc='av-align'):
        align_score_list.append(align_score)

    align_scores = torch.cat(align_score_list)

    return align_scores


def calc_av_score(video_list, audio_list, prompt_list, device='cuda:0',
                  sample_rate=16000, window_size_s=0.5, window_overlap_s=0, topk_min=0.4):
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    dataloader = create_dataloader(
        metric='av-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sample_rate=sample_rate,
        window_size_s=window_size_s,
        window_overlap_s=window_overlap_s,
        batch_size=1
    )

    avh_score_list, javis_score_list = [], []
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for avh_inputs, javis_inputs, video_windows_indices in tqdm(dataloader, desc='av-score'):
        assert video_windows_indices.shape[0] == 1

        # image shape: (B,C,H,W), video shape: (B,15,C,2,H,W), audio shape(B,3,C,T,S), 
        avh_inputs = {k: v[0].to(device) for k, v in avh_inputs.items()}
        javis_inputs = {k: v[0].to(device) for k, v in javis_inputs.items()}
        video_windows_indices = video_windows_indices[0]

        # for AVHScore
        with torch.no_grad():
            embeddings = model(avh_inputs)
        embed_frames = embeddings[ModalityType.VISION]  # shape(T,1024)
        embed_audio = embeddings[ModalityType.AUDIO]    # shape(1,1024)
        avh_score = cos(embed_frames, embed_audio).mean().item() #* 1000
        avh_score_list.append(avh_score)

        # for JavisScore
        M, N = video_windows_indices.shape[:2]
        with torch.no_grad():
            embeddings = model(javis_inputs)
        embed_video = embed_frames[video_windows_indices.flatten()].view(M, N, -1)  # shape(M,N,1024)
        embed_audio = embeddings[ModalityType.AUDIO].unsqueeze(1)    # shape(M,1,1024)

        javis_score_clip = cos(embed_video, embed_audio)  # shape(M,N)
        k = topk_min if isinstance(topk_min, int) else int(N * topk_min)
        topk_values, _ = torch.topk(javis_score_clip, k, dim=1, largest=False, sorted=False)
        javis_score_window = topk_values.mean(dim=1)
        javis_score = javis_score_window.mean(dim=0).item()

        # javis_score_clip = cos(embed_video, embed_audio).mean(dim=1)  # shape(M)
        # k = topk_min if isinstance(topk_min, int) else math.ceil(M * topk_min)
        # topk_values, _ = torch.topk(javis_score_clip, k, dim=0, largest=False, sorted=False)
        # javis_score = topk_values.mean().item()
        
        javis_score_list.append(javis_score)

    avh_scores = torch.tensor(avh_score_list)
    javis_scores = torch.tensor(javis_score_list)

    return avh_scores, javis_scores


def calc_desync_score(video_list, audio_list, max_length_s=8, device='cuda:0',
                      syncformer_ckpt_path='./checkpoints/synchformer_state_dict.pth'):
    # Original code from "https://github.com/hkchengrex/av-benchmark"

    if not osp.exists(syncformer_ckpt_path):
        print(f"Downloading SynchFormer weights to {syncformer_ckpt_path} ...")
        os.makedirs(osp.dirname(syncformer_ckpt_path), exist_ok=True)
        torch.hub.download_url_to_file(
            "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth",
            syncformer_ckpt_path,
            progress=True,
        )

    synchformer = Synchformer().to(device).eval()
    sd = torch.load(syncformer_ckpt_path, weights_only=True)
    synchformer.load_state_dict(sd)

    dataloader = create_dataloader(
        metric='desync-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        batch_size=8,
        max_length_s=max_length_s
    )

    # TODO: compatible with Torch 2.4.0
    sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
        wkwargs={'device': device}
    )
    mel_scale_fb = sync_mel_spectrogram.mel_scale.fb.to(device)
    sync_mel_spectrogram.mel_scale.register_buffer('fb', mel_scale_fb)

    # [-2.0, -1.8, ..., 0.0, ..., 1.8, 2.0], equals to `torch.linspace(-2, 2, 21)`
    sync_grid = make_class_grid(-2, 2, 21)

    desync_score_list = []
    for video, audio in tqdm(dataloader, desc='desync'):
        video, audio = video.to(device), audio.to(device)

        ## Step1. Encode Video
        # x: (B, T, C, H, W) H/W: 224
        b, t, c, h, w = video.shape
        assert c == 3 and h == 224 and w == 224

        # partition the video
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(video[:, i * step_size:i * step_size + segment_size])
        vx = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        vx = rearrange(vx, 'b s t c h w -> (b s) 1 t c h w')
        vx = synchformer.extract_vfeats(vx)
        vx: torch.Tensor = rearrange(vx, '(b s) 1 t d -> b s t d', b=b)
        
        ## Step2. Encode Audio
        assert audio.shape[0] == b
        _, t = audio.shape

        # partition the video
        segment_size = 10240
        step_size = 10240 // 2
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(audio[:, i * step_size:i * step_size + segment_size])
        ax = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        ax = torch.log(sync_mel_spectrogram(ax) + 1e-6)
        ax = pad_or_truncate(ax, 66)

        mean = -4.2677393
        std = 4.5689974
        ax = (ax - mean) / (2 * std)
        # ax: B * S * 128 * 66
        ax: torch.Tensor = synchformer.extract_afeats(ax.unsqueeze(2))

        ## Step3. Calculate DeSync Score
        batch_sync_scores = []
        assert (frame_num := vx.shape[1]) == ax.shape[1]
        segment_size = 14

        segment_num = math.ceil(frame_num / segment_size)
        for si in range(segment_num):
            fstart, fend = si * segment_size, min((si + 1) * segment_size, frame_num)
            vx_segment, ax_segment = vx[:, fstart:fend], ax[:, fstart:fend]
            flen = fend - fstart

            delta_frame = segment_size - flen
            if delta_frame > 0:
                if si == 0:
                    repeat_ = math.ceil(delta_frame / flen)
                    video_pad = vx_segment.repeat(1, repeat_, *([1] * (vx_segment.dim() - 2)))
                    video_pad = video_pad[:, :delta_frame, ...]
                    vx_segment = torch.cat((vx_segment, video_pad), dim=1)
                    audio_pad = ax_segment.repeat(1, repeat_, *([1] * (ax_segment.dim() - 2)))
                    audio_pad = audio_pad[:, :delta_frame, ...]
                    ax_segment = torch.cat((ax_segment, audio_pad), dim=1)
                else:
                    assert si == segment_num - 1
                    vx_segment, ax_segment = vx[:, -segment_size:], ax[:, -segment_size:]
            
            # shape (B, 21)
            logits = synchformer.compare_v_a(vx_segment, ax_segment)
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            # shape (B, )
            for j in range(vx_segment.shape[0]):
                batch_sync_scores.append(abs(sync_grid[top_id[j]].item()))
        
        batch_sync_scores = torch.tensor(batch_sync_scores)
        batch_sync_scores = batch_sync_scores.reshape(b, -1).mean(dim=1)
        desync_score_list.append(batch_sync_scores)

    desync_scores = torch.cat(desync_score_list)

    return desync_scores


def calc_audio_score(gt_audio_list, pred_audio_list, prompt_list, device='cuda:0', 
                     exist_metrics={}, bs=8, **kwargs):
    # calculate "fad", "quality", "ib_ta", "clap" for audio_score

    # Part I
    if "fad" not in exist_metrics or kwargs.get('force_eval', False):
        from .AudioCLIP.get_embedding import preprocess_audio

        audioclip = load_audioclip_pretrained(device)

        real_loader = create_dataloader_for_fvd_vanilla(gt_audio_list, gt_audio_list, audio_only=True, **kwargs)
        fake_loader = create_dataloader_for_fvd_vanilla(pred_audio_list, pred_audio_list, audio_only=True, **kwargs)

        loader_dict = {'real': real_loader, 'fake': fake_loader}
        embed_dict = {}
        for t, loader in loader_dict.items():
            audio_embeds = []
            for _, sample in enumerate(tqdm(loader, desc=f'fad: {t}')):
                audio_sample = sample['audio'].to(device)

                audios = preprocess_audio(audio_sample).to(device)

                with torch.no_grad():
                    audioclip_audio_embed = audioclip(audio=audios, video=None)[0][0][0]
                assert audio_sample.shape[0] == audioclip_audio_embed.shape[0]
                
                audio_embeds.append(audioclip_audio_embed)

            embed_dict[t] = torch.cat(audio_embeds)
        
        sample_num = min(len(embed_dict['fake']), len(embed_dict['real']))
        fad = frechet_distance(
            embed_dict['fake'][:sample_num], embed_dict['real'][:sample_num]
        ).item() * 10000
        exist_metrics["fad"] = fad
    
    # Part II
    if "quality" not in exist_metrics or kwargs.get('force_eval', False):
        max_length_s = kwargs.get('max_audio_len_s', 8.0)
        audio_quality = calc_audio_quality_score(pred_audio_list, prompt_list, max_length_s, device, bs=bs)
        exist_metrics["quality"] = audio_quality.mean().item()
    
    # Part III
    if "ib_ta" not in exist_metrics or kwargs.get('force_eval', False):
        model = imagebind_model.imagebind_huge(pretrained=True)
        model.eval()
        model.to(device)

        text_embeds, audio_embeds = [], []
        # fast enough in a for-loop
        for i in tqdm(range(0, len(pred_audio_list), bs), desc='ib_score'):
            prompts, audios = prompt_list[i:i+bs], pred_audio_list[i:i+bs]
            inputs = {
                ModalityType.TEXT: imagebind_data.load_and_transform_text(prompts, device),
                ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(audios, device),
            }

            with torch.no_grad():
                embeddings = model(inputs)

            text_embeds.append(embeddings[ModalityType.TEXT])
            audio_embeds.append(embeddings[ModalityType.AUDIO])

        text_embeds, audio_embeds = torch.cat(text_embeds), torch.cat(audio_embeds)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_ta = cos(text_embeds, audio_embeds).mean().item()
        exist_metrics["ib_ta"] = sim_ta
    
    # Part IV
    if "clap" not in exist_metrics or kwargs.get('force_eval', False):
        clap_score = calc_clap_score(pred_audio_list, prompt_list, device)
        exist_metrics["clap"] = clap_score.mean().item()
    
    return 