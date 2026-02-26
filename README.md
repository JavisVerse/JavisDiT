## <div align="center"> JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation</div>

<div align="center">

[[`HomePage`](https://JavisVerse.github.io/JavisDiT2-page/)] 
[[`ArXiv Paper`](https://arxiv.org/abs/2602.19163)] 
[[`HF Paper`](https://huggingface.co/papers/2602.19163)]
[[`Model`](https://huggingface.co/collections/JavisVerse/javisdit-v1.0)]

</div>


Under the [JavisVerse](https://javisverse.github.io/) project, this repo presents two versions of the JavisDiT series:

- [JavisDiT](https://arxiv.org/abs/2503.23377): Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization (a.k.a. **JavisDiT-v0.1**)
- [JavisDiT++](https://arxiv.org/abs/2602.19163): Unified Modeling and Optimization for Joint Audio-Video Generation (a.k.a. **JavisDiT-v1.0**)

The main branch maintains **JavisDiT++**, a more concise yet powerful DiT model to generate semantically and temporally aligned sounding videos with textual conditions. Auxiliary support of JavisDiT (v0.1) can be found at [assets/docs/JavisDiT.md](assets/docs/JavisDiT.md).

https://github.com/user-attachments/assets/c7a0d9dc-71d3-4fb6-8c97-00553062de51
<!-- [![cover](https://img.youtube.com/vi/PUSGQU6ZpdE/maxresdefault.jpg)](https://www.youtube.com/watch?v=PUSGQU6ZpdE) -->

<!-- <video controls width="100%">
  <source src="https://raw.githubusercontent.com/JavisVerse/JavisDiT/main/assets/video/teaser-video-JavisDiT++.mp4" type="video/mp4">
</video> -->

## 📰 News

- **[2026.02.26]** 🔥🔥 This repository has upgraded from [JavisDiT](https://arxiv.org/abs/2503.23377) to [JavisDiT++](https://arxiv.org/abs/2602.19163), both of which are accepted at **ICLR 2026**. Fore the previous version, please refer to [assets/docs/JavisDiT.md](assets/docs/JavisDiT.md).
- **[2025.12.26]** 🚀 JavisDiT and JavisGPT are integrated into the [JavisVerse](https://javisverse.github.io/) project. We hope to contribute to the _Joint Audio-Video Intelligence Symphony (Javis)_ in the community.
- **[2025.12.26]** 🚀 We released [JavisGPT](https://openreview.net/forum?id=MZoOpD9NHV), a unified multi-modal LLM for sounding-video comprehension and generation. For more details refer to this [repo](https://github.com/JavisVerse/JavisGPT). 
- **[2025.08.11]** 🔥 We released the data and code for JAVG evaluation. For more details refer to [eval/javisbench/README.md](eval/javisbench/README.md).
- **[2025.04.15]** 🔥 We released the data preparation and model training instructions. You can train JavisDiT with your own dataset!
- **[2025.04.07]** 🔥 We released the inference code and a preview model of **JavisDiT-v0.1** at [HuggingFace](https://huggingface.co/JavisDiT). <!-- , which includes **JavisDiT-v0.1-audio**, **JavisDiT-v0.1-prior**, and **JavisDiT-v0.1-jav** (with a [low-resolution version](https://huggingface.co/JavisVerse/JavisDiT-v0.1-jav-240p4s) and a [full-resolution version](https://huggingface.co/JavisVerse/JavisDiT-v0.1-jav)). -->
- **[2025.04.03]** We release the repository of [JavisDiT](https://arxiv.org/abs/2503.23377). Code, model, and data are coming soon.

### 👉 TODO 
- [ ] Release the data and evaluation code for JavisScore.

## Brief Introduction

**JavisDiT++** addresses the key bottleneck of JAVG with a unified perspective of modeling and optimization.

<!-- <p align="center">
  <img src="./assets/image/JavisDiT-intro-resized.png" width="550"/>
</p> -->

![framework](./assets/image/JavisDiT++-framework.jpg)

- We model JAVG via joint self-attention to enable dense inter-modal interaction, with modality-specific MoE (**MS-MoE**) design to refine intra-modal representation.
- We propose a temporally aligned rotary position encoding (**TA-RoPE**) scheme to ensure explicit and fine-grained audio-video token synchronization.
- We devise the **AV-DPO** technique to consistently improve audio-video quality and synchronization by aligning generation with human preferences.

We hope to set a new standard for the JAVG community. For more technical details, kindly refer to the original [paper](https://arxiv.org/abs/2602.19163). 


## Installation

### Install from Source

For CUDA 12.1, you can install the dependencies with the following commands.

```bash
# create a virtual env and activate (conda as an example)
conda create -n javisdit python=3.10
conda activate javisdit

# download the repo
git clone https://github.com/JavisVerse/JavisDiT
cd JavisDiT

# install torch, torchvision and xformers
pip install -r requirements/requirements-cu121.txt

# install ffpmeg
conda install -c conda-forge ffmpeg -y

# the default installation is for inference only
pip install -v .
# for development mode, `pip install -v -e .`
# to skip dependencies, `pip install -v -e . --no-deps`
pip install flash-attn --no-build-isolation

# replace
PYTHON_SITE_PACKAGES=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
cp assets/src/pytorchvideo_augmentations.py ${PYTHON_SITE_PACKAGES}/pytorchvideo/transforms/augmentations.py
cp assets/src/funasr_utils_load_utils.py ${PYTHON_SITE_PACKAGES}/funasr/utils/load_utils.py
```



### Pre-trained Weights


| Version   | Base Model | Resolution | Duration | Model Size |
| --------- | ---------- | ---------- | -------- | ---------- |
| [JavisDiT-v1.0](https://huggingface.co/JavisVerse/JavisDiT-v1.0-jav)    | [Wan2.1-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)    | 240P-480P | 2s-5s | 2.1B |

Run the following command to download the pre-trained weights.

```bash
# download JavisDiT weights
hf download JavisVerse/JavisDiT-v1.0-jav --local-dir ./checkpoints/JavisDiT-v1.0-jav

# download VAEs
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./checkpoints/Wan2.1-T2V-1.3B
hf download cvssp/audioldm2 --local-dir ./checkpoints/audioldm2
```


## Inference


The following command will generate a standard 480P sounding video for 5 seconds at 16 FPS:

```bash
python scripts/inference.py \
  configs/javisdit-v1-0/inference/sample.py \
  --num-frames 81 --resolution 480p --aspect-ratio 9:16 \
  --model-path ./checkpoints/JavisDiT-v1.0-jav \
  --prompt "A brown bear is walking towards the camera, growling in a natural setting with greenery in the background." \
  --verbose 2
```

`--verbose 2` will display the progress of a single diffusion.
If you want to generate 240P, 4 second videos on a given prompt list (organized with a `.txt` for `.csv` file):

```bash
python scripts/inference.py \
  configs/javisdit-v1-0/inference/sample.py \
  --num-frames 65 --resolution 240p --aspect-ratio 9:16 \
  --model-path ./checkpoints/JavisDiT-v1.0-jav \
  --prompt-path data/meta/JavisBench.csv --verbose 1
```

`--verbose 1` will display the progress of the whole generation list.


To enable multi-device inference, you need to use `torchrun` to run the inference script. The following command will run the inference with 2 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/inference.py \
  configs/javisdit-v1-0/inference/sample.py \
  --num-frames 65 --resolution 240p --aspect-ratio 9:16 \
  --model-path ./checkpoints/JavisDiT-v1.0-jav \
  --prompt-path data/meta/JavisBench.csv --verbose 1
```


## Training 

### Data Preparation

We follow [OpenSora](https://github.com/hpcaitech/Open-Sora) to use a `.csv` file to manage all the training entries and their attributes for efficient training:

| path | id | relpath | num_frames | height | width | aspect_ratio | fps | resolution | audio_path | audio_fps | text |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---|
| /path/to/xxx.mp4 | xxx | xxx.mp4 | 240 | 480 | 640 | 0.75 | 24 | 307200 | /path/to/xxx.wav | 16000 | yyy |

The content of columns may vary in different training stages. The detailed instructions for each training stage can be found in [assets/docs/data.md](assets/docs/data.md).


### Stage1 - Audio Pretrain

This stage performs audio pretraining to intialize the text-to-audio generation. 
Following the [instructions](assets/docs/data.md#stage1---audio-pretrain) to prepare the audio data, or just download the preprocessed data from [HuggingFace](https://huggingface.co/datasets/JavisVerse/JavisData-Audio):

```bash
hf download --repo-type dataset JavisVerse/JavisData-Audio --local-dir /path/to/audio
```

Then, run the command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train.py \
    configs/javisdit-v1-0/train/stage1_audio.py \
    --data-path data/meta/audio/train_audio.csv
```

The resulting checkpoints will be saved at `runs/0aa-Wan2_1_T2V_1_3B/epoch0bb-global_stepccc/model`, and you can move the weights to `outputs/stage1_audio_pt/model`.

### Stage2 - Audio-Video SFT

This stage finetunes the T2V+T2A backbone for preliminary T2AV generation. 
Following the [instructions](assets/docs/data.md#stage2---audio-video-sft) to prepare the audio-video data. Due to copyright issues, we cannot release the raw YouTube videos used by [TAVGBench](https://github.com/OpenNLPLab/TAVGBench).

Then, run the command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train.py \
    configs/javisdit-v1-0/train/stage2_audio_video.py \
    --data-path data/meta/video/train_av_sft.csv
```

The resulting checkpoints will be saved at `runs/0xx-Wan2_1_T2V_1_3B/epoch0yy-global_stepzzz/*`, and you can move the weights to `outputs/stage2_av_sft/*`.

### Stage3 - Audio-Video DPO

This stage deploy DPO to improve human-preference alignment of T2AV generation.
Following the [instructions](assets/docs/data.md#stage3---audio-video-dpo) to prepare the audio-video preference data. We are working on releasing our generated data. 

Then, run the command:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node 8 \
    scripts/train.py \
    configs/javisdit-v1-0/train/stage3_audio_video_dpo.py \
    --data-path data/meta/avdpo/train_av_dpo.csv
```

The resulting checkpoints will be saved at `runs/0aa-Wan2_1_T2V_1_3B/epoch0bb-global_stepccc/model`.

```bash
mv runs/0aa-Wan2_1_T2V_1_3B/epoch0bb-global_stepccc ./outputs/JavisDiT-v1.0-jav
```

## Evaluation

### Installation

Install necessary packages:

```bash
pip install -r requirements/requirements-eval.txt
```

### Data Preparation

Download the meta file and data of [JavisBench](https://huggingface.co/datasets/JavisVerse/JavisBench), and put them into `data/eval/`:

```bash
mkdir -p data/eval
hf download --repo-type dataset JavisVerse/JavisBench --local-dir data/eval/JavisBench
```

### Inference on JavisBench/JavisBench-mini

Run the joint audio-video generation (JAVG) inference to generate sounding videos in 240P for 4 seconds:

```bash
DATASET="JavisBench"  # or JavisBench-mini
prompt_path="data/eval/JavisBench/${DATASET}.csv"

cfg_file="configs/javisdit-v1-0/inference/sample.py"
model_path="./checkpoints/JavisDiT-v1.0-jav"
save_dir="samples/${DATASET}"

resolution=240p
num_frames=4s
aspect_ratio="9:16"

rm -rf ${save_dir}
python scripts/inference.py ${cfg_file} \
    --resolution ${resolution} --num-frames ${num_frames} --aspect-ratio ${aspect_ratio} \
    --prompt-path ${prompt_path} --model-path ${model_path} \
    --save-dir ${save_dir} --verbose 1

# (Optional, for evaluation) Extract audios from generated videos
python -m tools.datasets.convert video ${save_dir} --output ${save_dir}/meta.csv
python -m tools.datasets.datautil ${save_dir}/meta.csv --extract-audio --audio-sr 16000
rm -f ${save_dir}/meta*.csv
```


### Evaluation on JavisBench/JavisBench-mini

Run the following code and the results will be saved in `./evaluation_results`.

```bash
MAX_FRAMES=16
IMAGE_SIZE=224
MAX_AUDIO_LEN_S=4.0

# Params to calculate JavisScore
WINDOW_SIZE_S=2.0
WINDOW_OVERLAP_S=1.5

METRICS="all" 
RESULTS_DIR="./evaluation_results"

DATASET="JavisBench"  # or JavisBench-mini
INPUT_FILE="data/eval/JavisBench/${DATASET}.csv"
FVD_AVCACHE_PATH="data/eval/JavisBench/cache/fvd_fad/${DATASET}-vanilla-max4s.pt"
INFER_DATA_DIR="samples/${DATASET}"

python -m eval.javisbench.main \
  --input_file "${INPUT_FILE}" \
  --infer_data_dir "${INFER_DATA_DIR}" \
  --output_file "${RESULTS_DIR}/${DATASET}.json" \
  --max_frames ${MAX_FRAMES} \
  --image_size ${IMAGE_SIZE} \
  --max_audio_len_s ${MAX_AUDIO_LEN_S} \
  --window_size_s ${WINDOW_SIZE_S} \
  --window_overlap_s ${WINDOW_OVERLAP_S} \
  --fvd_avcache_path ${FVD_AVCACHE_PATH} \
  --metrics ${METRICS}
```

## Acknowledgement

Below we show our appreciation for the exceptional work and generous contribution to open source. Special thanks go to the authors of [Open-Sora](https://github.com/hpcaitech/Open-Sora), [Wan-Video](https://github.com/Wan-Video/Wan2.1), [AudioLDM2](https://github.com/haoheliu/AudioLDM2), and [TAVGBench](https://github.com/OpenNLPLab/TAVGBench) for their valuable codebase and dataset. For other works and datasets, please refer to our paper.


## Citation

If you find JavisDiT is useful and use it in your project, please kindly cite:

```bibtex
@inproceedings{liu2025javisdit,
  title       = {JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization}, 
  author      = {Liu, Kai and Li, Wei and Chen, Lai and Wu, Shengqiong and Zheng, Yanhao and Ji, Jiayi and Zhou, Fan and Luo, Jiebo and Liu, Ziwei and Fei, Hao and Chua, Tat-Seng},
  conference  = {The Fourteenth International Conference on Learning Representations},
  year        = {2026},
}

@inproceedings{liu2026javisdit++,
  title       = {JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation},
  author      = {Liu, Kai and Zheng, Yanhao and Wang, Kai and Wu, Shengqiong and Zhang, Rongjunchen and Luo, Jiebo and Hatzinakos, Dimitrios and Liu, Ziwei and Fei, Hao and Chua, Tat-Seng},
  conference  = {The Fourteenth International Conference on Learning Representations},
  year        = {2026},
}
```

<!-- ---

# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=JavisVerse/JavisDiT&type=Date)](https://star-history.com/#JavisVerse/JavisDiT&Date) -->

