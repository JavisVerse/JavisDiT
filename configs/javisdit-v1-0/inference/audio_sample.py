resolution = "240p"
aspect_ratio = "9:16"
num_frames = "4s"
fps = 16
audio_fps = 16000
frame_interval = 1
save_fps = 16

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "OpenSora"
dtype = "bf16"
loop = 1  # loop for video extension
condition_frame_length = 5  # used for video extension conditioning
align = 5  # TODO: unknown mechanism, maybe for conditional frame alignment?
verbose = 2

audio_only = True
prompt_key = 'audio_text'

# Model settings
model = dict(
    type="Wan2_1_T2V_1_3B",
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
    train_audio_specific_blocks=False,
    dual_ffn=True,
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
    num_sampling_steps=50,
    transform_scale=5.0,
    cfg_scale=5.0
)

aes = None   # aesthetic score
flow = None  # motion score
neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，低音质，差音质，最差音质，噪音，失真的，破音，削波失真，数字瑕疵，声音故障，不自然的，刺耳的，尖锐的，底噪，过多混响，过多回声，突兀的剪辑，不自然的淡出，录音质量差，业余录音'
use_text_preprocessing = False

# lora settings
lora_enabled = False
# lora_pretrained_dir = weight_dir + '/lora'