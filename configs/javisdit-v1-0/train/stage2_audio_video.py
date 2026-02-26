# Dataset settings

dataset = dict(
    type="VariableVideoAudioTextDataset",
    direct_load_video_cli=True,
    transform_name="resize_crop",
    audio_transform_name="mel_spec_audioldm2",
    default_video_fps=16,
    scale_factor=16,  # video scale factor
    use_audio_in_video=True,
)
load_text_features = False

# webvid
bucket_config = {  # 7.5s/it, randomly assigning raw videos to pre-defined and proper buckets
    "240p": {33: ((1.0, 1.0), 12), 49: ((1.0, 0.4), 10), 65: ((1.0, 0.3), 10), 81: ((1.0, 0.2),  8)},
    "360p": {33: ((0.5, 0.5),  6), 49: ((0.5, 0.3),  5), 65: ((0.5, 0.2),  5), 81: ((0.5, 0.2),  4)},
    "480p": {33: ((0.5, 0.3),  4), 49: ((1.0, 0.2),  3), 65: ((1.0, 0.2),  3), 81: ((1.0, 0.1),  2)},
}
grad_checkpoint = True

# Acceleration settings
num_workers = 16
num_bucket_build_workers = 8
dtype = 'bf16'
plugin = 'zero2'

# Model settings
video_weight_path = "./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
audio_weight_path = "./outputs/stage1_audio_pt"

model = dict(
    type="Wan2_1_T2V_1_3B",
    weight_init_from=[
        video_weight_path,
        audio_weight_path,
    ],
    model_type='t2av',
    patch_size=(1, 2, 2),
    dim=1536,
    ffn_dim=8960,
    freq_dim=256,
    num_heads=12,
    num_layers=30,
    window_size=(-1, -1),
    qk_norm=True,
    cross_attn_norm=True,
    audio_patch_size=(2, 2),
    audio_in_dim=8,
    audio_out_dim=8,
    audio_special_token=False,
    train_audio_specific_blocks=False,  # do not train alone
    dual_ffn=True,
    init_from_video_branch=False,
    class_drop_prob=0.1,
    audio_pe_type='interleave_offset',
)
vae = dict(
    type="Wan2_1_T2V_1_3B_VAE",
    from_pretrained="./checkpoints/Wan2.1-T2V-1.3B",
    vae_checkpoint='Wan2.1_VAE.pth',
    vae_stride=(4, 8, 8),
)
audio_vae = dict(
    type="AudioLDM2",
    from_pretrained="./checkpoints/audioldm2",
)
text_encoder = dict(
    type="Wan2_1_T2V_1_3B_t5_umt5",
    from_pretrained="./checkpoints/Wan2.1-T2V-1.3B",
    t5_checkpoint='models_t5_umt5-xxl-enc-bf16.pth',
    t5_tokenizer='google/umt5-xxl',
    text_len=512,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
    num_sampling_steps=50,
    transform_scale=5.0,
)

aes = None   # aesthetic score
flow = None  # motion score
neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，低音质，差音质，最差音质，噪音，失真的，破音，削波失真，数字瑕疵，声音故障，不自然的，刺耳的，尖锐的，底噪，过多混响，过多回声，突兀的剪辑，不自然的淡出，录音质量差，业余录音'

# lora settings
lora_enabled = True
lora_r = 64
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = [
    'self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
    'cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o',
    'ffn.0', 'ffn.2', 'audio_ffn.0', 'audio_ffn.2'
]
## NOTE: if the lora config remains unchanged, use `lora_pretrained_dir` to load config and weights;
# lora_pretrained_dir = audio_weight_path + '/lora'
## NOTE: otherwise, use `lora_pretrained_path` to only load the lora weights
# lora_pretrained_path = audio_weight_path + '/lora/adapter_model.bin'

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 2
log_every = 10
ckpt_every = 1000
save_total_limit = 20

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

# audio settins
sampling_rate = 16000
mel_bins = 64
audio_cfg = {
    "preprocessing": {
        "audio": {
            "sampling_rate": sampling_rate,
            "max_wav_value": 32768.0,
            "duration": 10.24,
            "scale_factor": 8 # pad 1 token at most.
        },
        "stft": {
            "filter_length": 1024,
            "hop_length": 160,
            "win_length": 1024,
        },
        "mel": {
            "n_mel_channels": mel_bins,
            "mel_fmin": 0,
            "mel_fmax": 8000,
        }
    },
    "augmentation": {
        "mixup": 0.0,
    }
}
