import argparse
import os
import os.path as osp
import pandas as pd
import json
from typing import List, Dict, Literal
from glob import glob
from copy import deepcopy
import math

import logging
logging.warning = lambda *args, **kwargs: None

import torch
import torch.distributed as dist
DEVICE_COUNT = torch.cuda.device_count()

from .src.metrics import (
    # quality
    calc_fvd_kvd_fad, calc_audio_quality_score, calc_video_quality_score,
    # alignment
    calc_imagebind_score, calc_clip_score, calc_clap_score, calc_cavp_score,
    # synchrony
    calc_av_align, calc_av_score, calc_desync_score,
    # audio-only
    calc_audio_score,
    # interface
    calc_fvd_kvd_fad_direct
)


class JavisBenchCategory(object):
    def __init__(self, cfg: str):
        self.cfg = cfg
        
        with open(cfg, 'r') as f:
            data = json.load(f)
        
        category_matrix = []
        for aspect in data:
            category_list = []
            for category in aspect['categories']:
                category_list.append(category['title'])
            category_matrix.append(category_list)

        self.category_cfg = data
        self.aspect_list = [aspect['aspect'] for aspect in data]
        self.category_matrix = category_matrix
        self.aspect_num = len(self.category_matrix)


class JavisEvaluator(object):
    def __init__(self, input_file: str, category_cfg: str, metrics: List[str], output_file: str, **kwargs):
        self.input_file = input_file
        self.df = pd.read_csv(input_file)
        eval_num = kwargs.pop('eval_num', None)
        if eval_num:
            print(f'Evaluate the first {eval_num} samples.')
            self.df = self.df.iloc[:eval_num]

        if category_cfg and osp.isfile(category_cfg) and kwargs.get('verbose'):
            self.cat2indices = self.parse_aspect_dict(category_cfg)
        else:
            self.cat2indices = None

        self.output_file = output_file

        self.total_metrics = [
            'fvd+kvd+fad',   # quality
            'video-quality',  # visual quality and motion quality
            'audio-quality',  # audio quality
            'imagebind-score', 'cxxp-score',  # semantic consistency
            'av-align',  # av alignment
            'av-score',  #'avh-score', 'javis-score'
            'desync',  # av synchrony
            # 'audio-score', 
        ]
        if metrics == ['all']:
            metrics = self.total_metrics
        self.metrics = metrics
        self.metric2items = {
            # for general audio-video evaluation
            'fvd+kvd+fad': ['fvd', 'kvd', 'fad'],
            'video-quality': ['visual_quality', 'motion_quality'],
            'audio-quality': ['audio_quality'],
            'imagebind-score': ['ib_tv', 'ib_ta', 'ib_av'],
            'cxxp-score': ['clip_score', 'clap_score', 'cavp_score'],
            'av-align': ['av_align'],
            'av-score': ['avh_score', 'javis_score'],
            'desync': ['desync'],
            # for audio evaluation only
            'audio-score': ['fad', 'quality', 'ib_ta', 'clap'],
            # for audio-video reward calculation
            'av-reward': [
                'visual_quality', 'motion_quality', 'audio_quality',
                'ib_tv', 'ib_ta', 'ib_av',
                'desync_scores',
            ]
        }
        self.exclude = kwargs.pop('exclude', [])
        self.eval_kwargs = kwargs

        self.gather_audio_video_pred()

        if DEVICE_COUNT > 1:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank() % DEVICE_COUNT
        else:
            self.world_size = 1
            self.rank = -1
        self.is_main_process = self.rank in [0, -1]
        if self.world_size > 1:
            nums_per_rank = math.ceil(len(self.df) / self.world_size)
            start = self.rank * nums_per_rank
            self.df = self.df.iloc[start:start+nums_per_rank]
    
    def parse_aspect_dict(self, category_cfg:str):
        self.category = JavisBenchCategory(category_cfg)
        cat2indices: List[List[List[int]]] = []
        for ai in range(self.category.aspect_num):
            index_list = [[] for _ in range(len(self.category.category_matrix[ai]))]
            for pi, cat_str in enumerate(self.df[f'cat{ai}_ind'].tolist()):
                for ci in (cat_str.split(',') if isinstance(cat_str, str) else [cat_str]):
                    index_list[int(ci)].append(pi)
            cat2indices.append(index_list)
        
        return cat2indices
    
    @torch.no_grad()
    def __call__(self, *args, **kwds):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exist_metrics = self.load_metric()

        prompt_list = self.df['text'].tolist()
        # video_prompt_list = self.df.get('video_text', self.df['text']).tolist()
        video_prompt_list = prompt_list
        audio_prompt_list = self.df.get('audio_text', self.df['text']).tolist()
        # audio_prompt_list = prompt_list

        gt_video_list = self.df['path'].tolist()
        gt_audio_list = self.df.get('audio_path', self.df['path']).tolist()

        pred_video_list = self.df['pred_video_path'].tolist()
        pred_audio_list = self.df['pred_audio_path'].tolist()

        save_avalign_scores = self.eval_kwargs.get('save_avalign_scores', False)
        save_av_reward = False
        max_length_s = self.eval_kwargs.get('max_audio_len_s', 8.0)
        for metric in self.metrics:
            if not self.eval_kwargs.get('force_eval', False) and \
                    all(item in exist_metrics for item in self.metric2items[metric]):
                print(f'{metric} calculated. skip.')
                continue
            if metric in self.exclude:
                print(f'{metric} excluded. skip.')
                continue
            
            if metric == 'fvd+kvd+fad':
                mode = self.eval_kwargs.get('fvd_mode', 'vanilla')
                fvd, kvd, fad, embed_dict = calc_fvd_kvd_fad(
                    gt_video_list, pred_video_list, gt_audio_list, pred_audio_list, 
                    device, mode=mode, **self.eval_kwargs
                )
                if self.world_size > 1:
                    embed_dict = self.all_gather_object(
                        embed_dict, 
                        gather_func=lambda x_list: {
                            t: {
                                m: torch.cat([x[t][m] for x in x_list]) for m in ['video', 'audio']
                            } for t in ['real', 'fake']
                        }
                    )
                    sample_num = min(len(embed_dict['fake']['video']), len(embed_dict['real']['video']))
                    fvd, kvd, fad = calc_fvd_kvd_fad_direct(embed_dict, list(range(sample_num)))
                exist_metrics['fvd'], exist_metrics['kvd'], exist_metrics['fad'] = \
                    self.calc_category_distribution(metric, (fvd, kvd, fad), embed_dict=embed_dict)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'video-quality':
                visual_quality_scores, motion_quality_scores = \
                    calc_video_quality_score(pred_video_list, video_prompt_list, device)
                visual_quality_scores = self.all_gather_object(visual_quality_scores)
                motion_quality_scores = self.all_gather_object(motion_quality_scores)
                exist_metrics["visual_quality"] = self.calc_category_distribution(metric, visual_quality_scores)
                exist_metrics["motion_quality"] = self.calc_category_distribution(metric, motion_quality_scores)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'audio-quality':
                audio_sr = self.eval_kwargs.get('audio_sr', 16000)
                audio_quality_scores = calc_audio_quality_score(
                    pred_audio_list, audio_prompt_list, max_length_s, device, audio_sr=audio_sr)
                audio_quality_scores = self.all_gather_object(audio_quality_scores)
                exist_metrics["audio_quality"] = self.calc_category_distribution(metric, audio_quality_scores)
                self.write_metric(exist_metrics, metric)

            elif metric == 'imagebind-score':
                sim_tv_scores, sim_ta_scores, sim_av_scores = calc_imagebind_score(
                    pred_video_list, pred_audio_list, video_prompt_list, audio_prompt_list, device)
                sim_tv_scores = self.all_gather_object(sim_tv_scores)
                sim_ta_scores = self.all_gather_object(sim_ta_scores)
                sim_av_scores = self.all_gather_object(sim_av_scores)
                exist_metrics["ib_tv"] = self.calc_category_distribution(metric, sim_tv_scores)
                exist_metrics["ib_ta"] = self.calc_category_distribution(metric, sim_ta_scores)
                exist_metrics["ib_av"] = self.calc_category_distribution(metric, sim_av_scores)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'cxxp-score':
                if "clip_score" not in self.exclude:
                    clip_scores = calc_clip_score(pred_video_list, video_prompt_list, device)
                    clip_scores = self.all_gather_object(clip_scores)
                    exist_metrics["clip_score"] = self.calc_category_distribution(metric, clip_scores)
                if "clap_score" not in self.exclude:
                    clap_scores = calc_clap_score(pred_audio_list, audio_prompt_list, device)
                    clap_scores = self.all_gather_object(clap_scores)
                    exist_metrics["clap_score"] = self.calc_category_distribution(metric, clap_scores)
                if 'cavp_score' not in self.exclude:
                    cavp_scores = calc_cavp_score(pred_video_list, pred_audio_list, device,
                                                  cavp_config_path=self.eval_kwargs['cavp_config_path'])
                    cavp_scores = self.all_gather_object(cavp_scores)
                    exist_metrics["cavp_score"] = self.calc_category_distribution(metric, cavp_scores)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'av-align':
                av_align_scores = calc_av_align(pred_video_list, pred_audio_list)
                if save_avalign_scores:
                    assert len(av_align_scores) == len(self.df)
                    self.df['av_align_scores'] = av_align_scores.tolist()
                av_align_scores = self.all_gather_object(av_align_scores)
                exist_metrics["av_align"] = self.calc_category_distribution(metric, av_align_scores)
                self.write_metric(exist_metrics, metric)

            elif metric == 'av-score':
                avh_scores, javis_scores = calc_av_score(pred_video_list, pred_audio_list, prompt_list, device,
                                                        window_size_s=self.eval_kwargs.get("window_size_s", 2.0),
                                                        window_overlap_s=self.eval_kwargs.get("window_overlap_s", 1.5))
                if save_avalign_scores:
                    assert len(avh_scores) == len(javis_scores) == len(self.df)
                    self.df['avh_scores'] = avh_scores.tolist()
                    self.df['javis_scores'] = javis_scores.tolist()
                avh_scores = self.all_gather_object(avh_scores)
                javis_scores = self.all_gather_object(javis_scores)
                exist_metrics["avh_score"] = self.calc_category_distribution(metric, avh_scores)
                exist_metrics["javis_score"] = self.calc_category_distribution(metric, javis_scores)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'desync':
                desync_scores = calc_desync_score(pred_video_list, pred_audio_list, max_length_s, device)
                if save_avalign_scores:
                    assert len(desync_scores) == len(self.df)
                    self.df['desync_scores'] = desync_scores
                desync_scores = self.all_gather_object(desync_scores)
                exist_metrics["desync"] = self.calc_category_distribution(metric, desync_scores) 
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'audio-score':
                assert self.world_size == 1, 'Distributional evaluation on audio-score is not supported!'
                calc_audio_score(gt_audio_list, pred_audio_list, audio_prompt_list, device, 
                                 exist_metrics=exist_metrics, **self.eval_kwargs)
                self.write_metric(exist_metrics, metric)
            
            elif metric == 'av-reward':
                if 'visual_quality' not in self.exclude:
                    visual_quality_scores, motion_quality_scores = \
                        calc_video_quality_score(pred_video_list, video_prompt_list, device)
                    self.df['visual_quality'] = visual_quality_scores.tolist()
                    self.df['motion_quality'] = motion_quality_scores.tolist()
                    save_av_reward = True
                    visual_quality_scores = self.all_gather_object(visual_quality_scores)
                    motion_quality_scores = self.all_gather_object(motion_quality_scores)
                    exist_metrics['visual_quality'] = visual_quality_scores.mean().item()
                    exist_metrics['motion_quality'] = motion_quality_scores.mean().item()
                if 'audio_quality' not in self.exclude:
                    audio_sr = self.eval_kwargs.get('audio_sr', 16000)
                    audio_quality_scores = calc_audio_quality_score(
                        pred_audio_list, audio_prompt_list, max_length_s, device, audio_sr=audio_sr)
                    self.df['audio_quality'] = audio_quality_scores.tolist()
                    save_av_reward = True
                    audio_quality_scores = self.all_gather_object(audio_quality_scores)
                    exist_metrics['audio_quality'] = audio_quality_scores.mean().item()
                if 'ib_tv' not in self.exclude:
                    sim_tv_scores, sim_ta_scores, sim_av_scores = calc_imagebind_score(
                        pred_video_list, pred_audio_list, video_prompt_list, audio_prompt_list, device)
                    self.df["ib_tv"] = sim_tv_scores.tolist()
                    self.df["ib_ta"] = sim_ta_scores.tolist()
                    self.df["ib_av"] = sim_av_scores.tolist()
                    save_av_reward = True
                    sim_tv_scores = self.all_gather_object(sim_tv_scores)
                    sim_ta_scores = self.all_gather_object(sim_ta_scores)
                    sim_av_scores = self.all_gather_object(sim_av_scores)
                    exist_metrics['ib_tv'] = sim_tv_scores.mean().item()
                    exist_metrics['ib_ta'] = sim_ta_scores.mean().item()
                    exist_metrics['ib_av'] = sim_av_scores.mean().item()
                if 'desync' not in self.exclude:
                    desync_scores = calc_desync_score(pred_video_list, pred_audio_list, max_length_s, device)
                    self.df['desync_scores'] = desync_scores.tolist()
                    save_av_reward = True
                    desync_scores = self.all_gather_object(desync_scores)
                    exist_metrics['desync_scores'] = desync_scores.mean().item()
                self.write_metric(exist_metrics, metric)

        if self.world_size > 1:
            self.df = self.all_gather_object(self.df, pd.concat)

        if self.is_main_process:
            if save_avalign_scores:
                save_path = osp.splitext(self.output_file)[0] + '_avalign.csv'
                self.df.to_csv(save_path, index=False)

            if save_av_reward:
                save_path = osp.splitext(self.output_file)[0] + '_avreward.csv'
                self.df.to_csv(save_path, index=False)

    def all_gather_object(self, obj, gather_func=torch.cat):
        if self.world_size > 1:
            obj_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(obj_list, obj)
            obj = gather_func(obj_list)

        return obj

    def calc_category_distribution(self, metric, score_array, **kwargs):
        if self.cat2indices is None:
            if 'fvd' in metric:
                return score_array 
            else:
                return torch.mean(score_array).item()

        if 'fvd' in metric:
            fvd, kvd, fad = {'overall': score_array[0]}, {'overall': score_array[1]}, {'overall': score_array[2]}
            embed_dict = kwargs.get('embed_dict', {})
            for ai, index_list in enumerate(self.cat2indices):
                fvd[ai], kvd[ai], fad[ai] = [], [], []
                for ci, indices in enumerate(index_list):
                    ret = calc_fvd_kvd_fad_direct(embed_dict, indices)
                    fvd[ai].append(ret[0])
                    kvd[ai].append(ret[1])
                    fad[ai].append(ret[2])
            score_dict = fvd, kvd, fad
        else:
            score_dict = {'overall': score_array.mean().item(), 'all': score_array.tolist()}
            for ci, sub_cat_indices in enumerate(self.cat2indices):
                score_dict[ci] = []
                for sci, indices in enumerate(sub_cat_indices):
                    score_dict[ci].append(torch.mean(score_array[indices]).item())

        return score_dict

    def write_metric(self, metric:dict, metric_type:str):
        if self.is_main_process:
            os.makedirs(osp.dirname(self.output_file), exist_ok=True)
            for item in self.metric2items[metric_type]:
                if item not in metric:
                    print(f'{item}: NOT FOUND', end='; ')
                    continue
                score = metric[item]
                if isinstance(score, dict):
                    score = score['overall']
                print(f'{item}: {score:.4f}', end='; ')
            print()
            with open(self.output_file, 'w+') as f:
                json.dump(metric, f, indent=4, ensure_ascii=False)

        if self.world_size > 1:
            dist.barrier()

    def load_metric(self):
        metric = {}
        if osp.exists(self.output_file) and osp.getsize(self.output_file) > 0:
            with open(self.output_file, 'r') as f:
                metric = json.load(f)
        return metric

    def gather_audio_video_pred(self):
        infer_data_dir = self.eval_kwargs['infer_data_dir']
        if not infer_data_dir:
            if not self.eval_kwargs['eval_gt']:
                assert 'pred_video_path' in self.df and 'pred_audio_path' in self.df
            else:
                assert 'fvd+kvd+fad' not in self.metrics
                self.df['pred_video_path'] = self.df['path']
                self.df['pred_audio_path'] = self.df['audio_path']
            return
        assert osp.isdir(infer_data_dir), infer_data_dir
        audio_only = self.metrics == ['audio-score']
        sample_num = len(self.df)
        if audio_only:
            pred_audio_list = [f'{infer_data_dir}/sample_{i:04d}.wav' for i in range(sample_num)]
            pred_video_list = [''] * sample_num
            assert all(osp.exists(path) for path in pred_audio_list)
            self.df['text'] = self.df['audio_text']
            self.df['path'] = self.df['audio_path']
        else:
            pred_audio_list = [f'{infer_data_dir}/sample_{i:04d}.wav' for i in range(sample_num)]
            pred_video_list = [f'{infer_data_dir}/sample_{i:04d}.mp4' for i in range(sample_num)]
            assert all(osp.exists(path) for path in pred_audio_list)
            assert all(osp.exists(path) for path in pred_video_list)

        self.df['pred_video_path'] = pred_video_list
        self.df['pred_audio_path'] = pred_audio_list

def run_eval(args):
    if DEVICE_COUNT > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"]) % DEVICE_COUNT
        torch.cuda.set_device(local_rank)
    else:
        local_rank = -1

    if local_rank in [-1, 0]:
        print(f"Start evaluation on {args.infer_data_dir}")

    evaluator = JavisEvaluator(**vars(args))
    evaluator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="path to input csv file", required=True)
    parser.add_argument("--infer_data_dir", type=str, default=None, help="directory to audio-video inference results")
    parser.add_argument("--output_file", type=str, default=None, help="path to output json file", required=True)
    parser.add_argument("--category_cfg", type=str, default='./eval/javisbench/configs/category.json')
    parser.add_argument("--metrics", type=str, nargs='+', default='all', help="metrics to calculate, default as `all`")
    parser.add_argument("--exclude", type=str, nargs='+', default=[], help="skipping specific metric calculation")
    parser.add_argument("--verbose", action='store_true', default=False, help="whether to present category-specific score list")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loading")
    # parameters for evaluation
    parser.add_argument("--max_frames", type=int, default=24, help="size of the input video")
    parser.add_argument("--max_audio_len_s", type=float, default=None, help="maximum length of the audio")
    parser.add_argument("--video_fps", type=int, default=24, help="frame rate of the input video")
    parser.add_argument("--audio_sr", type=int, default=16000, help="sampling rate of the audio")
    parser.add_argument("--image_size", type=int, default=224, help="size of the input image")
    parser.add_argument("--eval_num", type=int, default=None, help="number of videos to evaluate")
    parser.add_argument("--fvd_avcache_path", type=str, default=None, help="path to the audio-video cache file for FVD/KVD/FAD evaluation")
    parser.add_argument("--fvd_mode", type=str, default='vanilla', choices=['vanilla', 'mmdiffusion'], help="mode of fvd calculation, `video` or `audio`")
    parser.add_argument("--force_eval", action='store_true', default=False, help="whether to evaluate scores even if existing")
    parser.add_argument("--eval_gt", action='store_true', default=False, help="whether to evaluate ground-truth audio-video pairs")
    # hyper-parameters for metrics
    parser.add_argument("--window_size_s", type=float, default=2.0, help="JavisScore window size")
    parser.add_argument("--window_overlap_s", type=float, default=1.5, help="JavisScore overlap size")
    parser.add_argument("--cavp_config_path", type=str, default='./eval/javisbench/configs/Stage1_CAVP.yaml', help="JavisScore overlap size")
    parser.add_argument("--save_avalign_scores", action='store_true', default=False, help="whether to return score list for AV-Align evaluation")
    args = parser.parse_args()

    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    cache_dir = f'{osp.dirname(args.output_file)}/cache/{osp.basename(osp.splitext(args.output_file)[0])}'
    setattr(args, "cache_dir", cache_dir)

    run_eval(args)
