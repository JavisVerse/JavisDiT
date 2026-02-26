import argparse
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import pandas as pd
from copy import deepcopy


def gather_dpo_gen(args):
    attr_keys = ['num_frames', 'height', 'width', 'aspect_ratio', 'resolution', 'fps', 'audio_path', 'audio_fps']
    unused_keys = 'aes,ocr,speech,flow,aes_rank,flow_rank,ocr_rank,mean_rank'.split(',')

    df_src = pd.read_csv(args.src_meta_path)
    for key in unused_keys:
        if key in df_src.columns:
            del df_src[key]
    # make dummy GT pairs
    df_src['pred_video_path'] = df_src['path'].tolist()
    for key in attr_keys:
        df_src[f'pred_{key}'] = df_src[key].tolist()

    df_gen = pd.read_csv(args.tgt_meta_path)
    # sort
    pattern = rf'^(.*?)sample_(\d+)-(\d+)'
    df_gen[["prefix", "d1", "d2"]] = df_gen['path'].str.extract(pattern, expand=True)
    df_gen["d1"] = df_gen["d1"].astype(int)
    df_gen["d2"] = df_gen["d2"].astype(int)
    df_gen = df_gen.sort_values(by=["prefix", "d1", "d2"], ascending=[True, True, True])
    df_gen = df_gen.drop(columns=["prefix", "d1", "d2"]).reset_index(drop=True)
    df_gen.to_csv(args.tgt_meta_path.replace('.csv', '_check.csv'), index=False)

    assert len(df_src.columns) == len(df_gen.columns) + 1 + len(attr_keys) + 1
    num_src, num_gen = len(df_src), len(df_gen)
    assert num_gen % num_src == 0
    num_gen_per_sample = num_gen // num_src
    df_list = []
    for i in range(num_gen_per_sample):
        df_gen_cur = df_gen.iloc[i::num_gen_per_sample]
        assert len(df_gen_cur) == num_src
        df_src_cur = deepcopy(df_src)
        df_src_cur['pred_video_path'] = df_gen_cur['path'].tolist()
        for key in attr_keys:
            assert key in df_gen_cur.columns
            df_src_cur[f'pred_{key}'] = df_gen_cur[key].tolist()
        df_list.append(df_src_cur)
        
    df_pair_total = pd.concat([df_src] + df_list)

    df_pair_total.to_csv(args.out_meta_path, index=False)


def rank_dpo_pair(args, del_gt=False, norm=True, mtype='micro'):
    df = pd.read_csv(args.src_meta_path)

    reward_aspects = {
        'video': ['visual_quality', 'motion_quality', 'ib_tv'],
        'audio': ['audio_quality', 'ib_ta'],
        'audio_video': ['ib_av', 'desync_scores'],
    }
    all_metrics = sum(reward_aspects.values(), [])

    for m in all_metrics:
        if norm:
            df[m] = (df[m] - df[m].mean()) / df[m].std()
    df['desync_scores'] = -df['desync_scores']

    if del_gt:
        df = df[df['pred_video_path'].str.contains('_gen/')]

    all_modality_metrics = []
    for modality, metrics in reward_aspects.items():
        k = f'{modality}_total_score'
        if mtype == 'micro':
            df[k] = df[metrics].mean(axis=1)  # higher is better
        elif mtype == 'macro':
            for m in metrics:
                df[f'{m}_rank'] = df[m].rank(ascending=True)  # higher is better
            df[k] = df[[f'{m}_rank' for m in metrics]].mean(axis=1)
        all_modality_metrics.append(k)

    df_path_group = df.groupby("path")
    data_dict = {k: [] for k in 'path,id,relpath,num_frames,height,width,aspect_ratio,fps,resolution,audio_path,audio_id,audio_fps,text,path_reject,audio_path_reject'.split(',')}
    for path, group in tqdm(df_path_group, total=len(df_path_group), desc='grouping'):
        group = group.reset_index(drop=True)
        # shape (n_sample, n_metrics)
        # g = group[all_metrics].to_numpy()
        g = group[all_modality_metrics].to_numpy()
        # shape (n_sample, n_sample)
        mask = (g[:, None, :] > g[None, :, :]).all(axis=-1)
        if not np.any(mask):
            continue

        dom_pairs = np.argwhere(mask)
        if len(dom_pairs) > 1:
            diffs = (g[dom_pairs[:, 0]] - g[dom_pairs[:, 1]]).sum(axis=1)
            max_idx = diffs.argmax()
            dom_pairs = dom_pairs[max_idx:max_idx+1]
        
        chosen_idx, reject_idx = int(dom_pairs[0][0]), int(dom_pairs[0][1])
        min_res_idx = chosen_idx if group["pred_resolution"][chosen_idx] < group["pred_resolution"][reject_idx] else reject_idx
        if not 0.5 < group["pred_aspect_ratio"][min_res_idx] < 0.6:  # specified for 0.56
            continue
        for key in ['height', 'width', 'aspect_ratio', 'resolution']:
            data_dict[key].append(group[f"pred_{key}"][min_res_idx])
        min_fnum_idx = chosen_idx if group["pred_num_frames"][chosen_idx] < group["pred_num_frames"][reject_idx] else reject_idx
        data_dict['num_frames'].append(group["pred_num_frames"][min_fnum_idx])
        for k in ['id', 'relpath', 'fps', 'text', 'audio_id', 'audio_fps']:
            data_dict[k].append(group[k][chosen_idx])
        data_dict['path'].append(group['pred_video_path'][chosen_idx])
        data_dict['audio_path'].append(group['pred_audio_path'][chosen_idx])
        data_dict['path_reject'].append(group['pred_video_path'][reject_idx])
        data_dict['audio_path_reject'].append(group['pred_audio_path'][reject_idx])
    assert all(len(data_dict[k]) == len(data_dict['path']) for k in data_dict.keys())

    df_dpo = pd.DataFrame(data_dict)
    
    # if norm: mtype += '_norm'
    # if not del_gt: mtype += '_wgt'
    # save_path = args.out_meta_path.replace('.csv', f'_{mtype}.csv')
    save_path = args.out_meta_path

    print(f'save {len(df_dpo)} pairs to {save_path}')
    df_dpo.to_csv(save_path, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task', type=str, choices=['gather_dpo_gen', 'gather_dpo_reward'], help='which task to run')
    argparser.add_argument('--src_meta_path', type=str, help='root meta path for source data')
    argparser.add_argument('--tgt_meta_path', type=str, help='root meta path for target data')
    argparser.add_argument('--out_meta_path', type=str, help='path to save processed .csv file')
    args = argparser.parse_args()

    eval(args.task)(args)
