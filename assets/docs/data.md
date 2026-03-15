
## Data Preparation for JavisDiT++


### Stage1 - Audio Pretrain

In this stage, we only need audio files to initialize the audio generation capability. Our pre-processed data is released at [HuggingFace](https://huggingface.co/datasets/JavisVerse/JavisData-Audio), and the data-curation pipeline is described as follows:

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text| audio_text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| placeholder.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | placeholder | yyy |

Download the audios (including [AudioCaps](https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link), [VGGSound](https://huggingface.co/datasets/Loie/VGGSound), [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet), [WavCaps](ttps://huggingface.co/datasets/cvssp/WavCaps), [Clotho](https://zenodo.org/records/3490684), [ESC50](https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download), [MACS](https://zenodo.org/records/2589280), [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html), [MusicInstrument](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset), [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), etc.), and put them into the same folder `/path/to/audios`. Follow the commands to automatically generate a `train_audio.csv` for configuration:

```bash
ROOT_AUDIO="/path/to/audios"
ROOT_META="./data/meta/audio"

# 1.1 Create a meta file from a unified audio folder. This should output ${ROOT_META}/meta.csv
python -m tools.datasets.convert audio ${ROOT_AUDIO} --output ${ROOT_META}/meta.csv

# 1.2 Get audio information. This should output ${ROOT_META}/meta_ainfo.csv
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --audio-info

# 2.1 Trim audios within 30 seconds. This should overwrite the raw audios by default and output ${ROOT_META}/meta_ainfo_trim30s.csv
python -m tools.datasets.datautil ${ROOT_META}/audio_meta.csv --trim-audio 30

# 2.2 Unify the sample rate to 16k Hz for all audios. This should output ${ROOT_META}/audio_meta_trim30s_sr16000.csv
python -m tools.datasets.datautil ${ROOT_META}/meta_ainfo_trim30s.csv --resample-audio --audio-sr 16000

# 3.1 Set dummy videos. This should output ${ROOT_META}/audio_meta_trim30s_sr16000_dummy_videos.csv
python -m tools.datasets.datautil ${ROOT_META}/audio_meta_trim30s_sr16000.csv --dummy-video

# 3.2 Get training meta csv. This should output ${ROOT_META}/train_audio.csv
python -m tools.datasets.find_audio_ds all \
    --data_root ${ROOT_AUDIO} \
    --meta_file ${ROOT_META}/audio_meta_trim30s_sr16000_dummy_videos.csv \
    --save_file ${ROOT_META}/train_audio.csv
```

### Stage2 - Audio-Video SFT

Here we provide an example with [TAVGBench](https://github.com/OpenNLPLab/TAVGBench) to prepare video-audio-text triplets for training. The `video_id` list used in JavisDiT++ is provided at [assets/meta/AV_SFT_330K_video_ids.txt](assets/meta/AV_SFT_330K_video_ids.txt), and you can easily transfer to your own datasets.

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 160 | 480 | 640 | 0.75 | 16 | 307200 | /path/to/xxx.wav | 16000 | yyy |

The following script will automatically generate a `train_av_sft.csv` for configuration:

```bash
ROOT_VIDEO="/path/to/videos"
ROOT_META="./data/meta/video"

fmin=10  # minial frames for each video
fps=16  # for Wan2.1-1.3B

# 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

# 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin${fmin}.csv
python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin ${fmin}

# 2.1 Unify FPS to 16 Hz for all videos. This will change the raw videos, and output ${ROOT_META}/meta_info_fmin${fmin}_fps${fps}.csv
python -m tools.datasets.datautil ${ROOT_META}/meta_info_fmin${fmin}.csv --uni-fps ${fps} --overwrite

# 3.1 Get training meta csv. This should output ${ROOT_META}/train_jav.csv
python -m tools.datasets.find_jav_ds tavgbench \
    --meta_src /path/to/TAVGBench/release_captions.txt \
    --meta_file ${ROOT_META}/meta_info_fmin${fmin}_fps${fps}.csv \
    --save_file ${ROOT_META}/train_av_sft.csv
```

If you get multiple data sources, just merge the csv files to a single one:

```bash
python -m tools.datasets.datautil ds1.csv ds2.csv ... --output /path/to/output.csv
```

### Stage3 - Audio-Video DPO

To run DPO, you need to prepare a data pool isolated from the SFT training data, organized into a `train_av_dpo_raw.csv`:

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 160 | 480 | 640 | 0.75 | 16 | 307200 | /path/to/xxx.wav | 16000 | yyy |

Then, run inference to generate N (e.g., 3) audio-video samples for each prompt input:

```bash
model_path="/path/to/av_sft_ckpt"
src_meta_path="/path/to/train_av_dpo_raw.csv"
save_dir="/path/to/dpo_gen"

resolution=480p # or 240p
num_frames=81  # 5s
aspect_ratio="9:16"
num_sample=3

cfg_file="configs/wan2.1/inference/sample.py"

torchrun --standalone --nproc_per_node 8 \
    scripts/inference.py \
    ${cfg_file} \
    --resolution ${resolution} --num-frames ${num_frames} --aspect-ratio ${aspect_ratio} \
    --prompt-path ${src_meta_path} --model-path ${model_path} --num-sample ${num_sample}$ \
    --save-dir ${save_dir} --verbose 1
```

Next, gather 1 ground-truth audio-video pair with 3 generated audio-video pair for each prompt input, and :

```bash
src_meta_path="/path/to/train_av_dpo_raw.csv"
data_dir="/path/to/dpo_gen"
gen_meta_path="/path/to/train_av_dpo_gen.csv"
ROOT_META="./data/meta/avdpo"

# first, create a meta file and extract audios from videos
python -m tools.datasets.convert video ${data_dir} --output ${data_dir}/meta.csv
python -m tools.datasets.datautil ${data_dir}/meta.csv --info --fmin 1
python -m tools.datasets.datautil ${ROOT_META}/meta_info_fmin1.csv --extract-audio --audio-sr 16000

# second, gather the generated audio-video pairs
python -m tools.datasets.process_dpo \
    --task gather_dpo_gen \
    --src_meta_path ${src_meta_path}$ \
    --tgt_meta_path ${ROOT_META}/meta_info_fmin1_au_sr16000.csv \
    --out_meta_path ${gen_meta_path}$
```

Then, score the 1+3 candidates for modality-aware rewarding, and the results will be saved at `./evaluation_results/audio_video_dpo_reward/avdpo_gen_avreward.csv`.

```bash
gen_meta_path="/path/to/train_av_dpo_gen.csv"
res_dir="./evaluation_results/audio_video_dpo_reward"

METRICS="av-reward"
MAX_AUDIO_LEN_S=5.0 

export CUDA_VISIBLE_DEVICES="0"
torchrun --nproc_per_node=1 -m eval.javisbench.main \
    --input_file ${gen_meta_path} \
    --output_file "${res_dir}/avdpo_gen.json" \
    --max_audio_len_s ${MAX_AUDIO_LEN_S} \
    --metrics ${METRICS}
```

Finally, ranking the generated samples and select the chosen-reject (or win-lose) sample pairs for DPO training:

```bash
src_meta_path="./evaluation_results/audio_video_dpo_reward/avdpo_gen_avreward.csv"
out_meta_path="./data/meta/avdpo/train_av_dpo.csv"

python -m tools.datasets.process_dpo \
    --task rank_dpo_pair \
    --src_meta_path ${src_meta_path}$ \
    --out_meta_path ${out_meta_path}$
```