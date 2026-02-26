# Dataset settings
audio_only = True

dataset = dict(
    type="VariableVideoAudioTextDataset",
    transform_name="resize_crop",
    audio_transform_name="mel_spec_audioldm2",
    audio_only=audio_only,
    default_video_fps=24
)

bucket_config = {
    "144p": {
        # 67GB, 6.4 s/iter
        33: (1.0, 128), # 2s
        # 66GB, 5.2 s/iter
        65: ((1.0, 0.5), 96), # 4s
        # 53GB, 4.7 s/iter
        97: ((1.0, 0.3), 80), # 6s
        # 55GB, 5.1 s/iter
        129: ((1.0, 0.2), 64) # 8s
    },
}

grad_checkpoint = True

# Acceleration settings
num_workers = 16
num_bucket_build_workers = 8
dtype = 'bf16'
plugin = 'zero2'

# Model settings
model = dict(
    type="Wan2_1_T2V_1_3B",
    weight_init_from="./checkpoints/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    model_type='t2a',
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
    train_audio_specific_blocks=True,
    dual_ffn=True,
    init_from_video_branch=True,
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
lora_enabled = False
# lora_r = 128
# lora_alpha = 256
# lora_target_modules = ['self_attn.q', 'self_attn.k', 'self_attn.v', 'self_attn.o',
#                         'cross_attn.q', 'cross_attn.k', 'cross_attn.v', 'cross_attn.o']
# lora_dropout = 0

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 50
log_every = 10
ckpt_every = 1000
save_total_limit = 2

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
