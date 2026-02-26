# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import numpy as np
from typing import Optional

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention, attnetion_with_mask, flash_attention_varlen
from javisdit.utils.misc import get_logger, requires_grad
from javisdit.utils.ckpt_utils import load_checkpoint
from javisdit.acceleration.checkpoint import auto_grad_checkpoint
from javisdit.models.layers.blocks import CaptionEmbedder, approx_gelu

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2

TOKEN_VIDEO = 0
TOKEN_AUDIO = 1
TOKEN_SOA = 2    # Start of Audio
TOKEN_EOA = 3    # End of Audio
TOKEN_AUDIO_PAD = 4

logger = get_logger()


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).to(x.dtype) #.float()


@amp.autocast(enabled=False)
def inv_freq(dims, base=10000):
    inv_freq = [1.0 / (base ** (torch.arange(0, dim, 2).to(dtype=torch.float64) / dim)) for dim in dims]
    return inv_freq


@amp.autocast(enabled=False)
def rope_cos_sin(inv_freqs, position_ids):
    """
    inv_freqs = List[torch.Tensor]
    """
    freqs_cis = []
    for idx, inv_freq in enumerate(inv_freqs):
        freqs = torch.outer(position_ids[:, idx], inv_freq)
        freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs))
    freqs_cis = torch.cat(freqs_cis, dim=-1) # seq_len, dim
    return freqs_cis


@amp.autocast(enabled=False)
def mmrope_apply(x, freqs):
    """
    x: (bs, seq_len, n, dim)
    freqs: (seq_len, dim),
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2)) # (bs, seq_len, n, dim/2)
    x_out = torch.view_as_real(x * freqs[None, :, None, :]).flatten(3)
    return x_out.to(dtype)


def get_video_position_ids(T_0, T_1, H_0, H_1, W_0, W_1, device):
    """video position ids
    """
    t_coords, h_coords, w_coords = torch.meshgrid(
        torch.arange(T_0, T_1, device=device),
        torch.arange(H_0, H_1, device=device),
        torch.arange(W_0, W_1, device=device),
        indexing='ij',
    )
    return torch.stack([t_coords, h_coords, w_coords], dim=-1).reshape(-1, 3)


def get_audio_position_ids(audio_shape, video_shape, audio_pe_type="interleave_offset", device="cuda:0"):
    _, _, audio_steps, M = audio_shape
    _, _, video_steps, H, W = video_shape
    if audio_pe_type == 'vanilla':
        t_coords, m_coords = torch.meshgrid(
            torch.arange(0, audio_steps, device=device, dtype=torch.float64),
            torch.arange(0, M, device=device, dtype=torch.float64),
            indexing='ij'
        )
        return torch.stack([t_coords, t_coords, m_coords], dim=-1).reshape(-1, 3)
    elif audio_pe_type == 'interpolate':
        t_arange = torch.linspace(0, video_steps-1, audio_steps, device=device, dtype=torch.float64)
        t_coords, m_coords = torch.meshgrid(
            t_arange,
            torch.arange(0, M, device=device, dtype=torch.float64),
            indexing='ij'
        )
        return torch.stack([t_coords, t_coords, m_coords], dim=-1).reshape(-1, 3)
    elif audio_pe_type == 'interleave':
        t_arange = torch.ceil(torch.linspace(0, video_steps-1, audio_steps, device=device, dtype=torch.float64)).long()
        t_coords, m_coords = torch.meshgrid(
            t_arange,
            torch.arange(0, M, device=device),
            indexing='ij'
        )
        w_arange = torch.arange(0, audio_steps, device=device)
        w_coords = w_arange[:, None].repeat(1, M)
        return torch.stack([t_coords, w_coords, m_coords], dim=-1).reshape(-1, 3)
    elif audio_pe_type == 'interleave_offset':
        t_arange = torch.ceil(torch.linspace(0, video_steps-1, audio_steps, device=device, dtype=torch.float64)).long()
        t_coords, m_coords = torch.meshgrid(
            t_arange,
            torch.arange(W, M+W, device=device),
            indexing='ij'
        )
        w_arange = torch.arange(H, H+audio_steps, device=device)
        w_coords = w_arange[:, None].repeat(1, M)
        return torch.stack([t_coords, w_coords, m_coords], dim=-1).reshape(-1, 3)
    else:
        raise ValueError(f'Unknown audio position encoding type: {audio_pe_type}')


def get_video_audio_position_ids(video_shape, audio_shape, audio_pe_type="interleave_offset", device="cuda:0"):
    _, _, Tv, H, W = video_shape
    # _, _, Ta, M = audio_shape
    video_pos_ids = get_video_position_ids(0, Tv, 0, H, 0, W, device=device).to(torch.float64)
    audio_pos_ids = get_audio_position_ids(audio_shape, video_shape, audio_pe_type=audio_pe_type, 
                                           device=device).to(torch.float64)
    return video_pos_ids, audio_pos_ids


def scatter_audio_to_video_by_timesteps(
    audio_shape, 
    video_shape, 
    add_special_token, 
    audio_step_length, 
    video_step_length,
    audio_pe_type="interleave_offset", 
    device="cuda:0"
):
    _, _, Ta, M = audio_shape
    _, _, Tv, H, W = video_shape

    audio_timestamps = np.linspace(0, (Ta - 1) * audio_step_length / video_step_length, Ta)
    video_token_types: list = [TOKEN_VIDEO] * (H * W) 
    audio_token_types: list = [TOKEN_AUDIO] * M

    token_types = []
    audio_ptr = 0
    audio_win_num = 0
    for v_idx in range(Tv):
        token_types.extend(video_token_types)
        if audio_ptr >= Ta:
            continue
        audio_win_num += 1
        if add_special_token:
            token_types.extend(TOKEN_SOA)
        while audio_ptr < Ta and audio_timestamps[audio_ptr] <= v_idx + 1:
            token_types.extend(audio_token_types)
            audio_ptr += 1
        if add_special_token:
            token_types.extend(TOKEN_EOA)
    token_types = torch.tensor(token_types, device=device, dtype=torch.long)
    video_pos_ids, audio_pos_ids = get_video_audio_position_ids(
        video_shape, audio_shape, audio_pe_type=audio_pe_type, device=device
    )
    position_ids = torch.zeros((len(token_types), 3), dtype=torch.float64, device=device)
    position_ids[token_types == TOKEN_VIDEO] = video_pos_ids
    position_ids[token_types == TOKEN_AUDIO] = audio_pos_ids
    if add_special_token:
        soa_pos_ids = torch.tensor([[i, i, 0]  for i in range(audio_win_num)], device=device)
        eoa_pos_ids = torch.tensor([[i, i, 1] for i in range(audio_win_num)], device=device)
        position_ids[token_types == TOKEN_SOA] = soa_pos_ids
        position_ids[token_types == TOKEN_EOA] = eoa_pos_ids
    return token_types, position_ids


def video_audio_interleave(
    audio_tokens:Optional[torch.Tensor], # [B,C,L,M]
    video_tokens:Optional[torch.Tensor] = None, # [B,C,T,H,W]
    num_frames:int = 9,
    audio_step_length: float = 80.0,
    video_step_length: float = 250.0,
    use_audio:bool = False,
    use_video:bool = False,
    audio_special_token:bool = False,
    audio_start_token:Optional[torch.Tensor] = None, # [C]
    audio_end_token:Optional[torch.Tensor] = None, # [C]
    audio_pe_type: Optional[str] = 'interleave',
):
    device = audio_tokens.device if audio_tokens is not None else video_tokens.device
    dtype = audio_tokens.dtype if audio_tokens is not None else video_tokens.dtype
    batch_size, dim = audio_tokens.shape[:2] if use_audio else video_tokens.shape[:2]

    audio_shape = audio_tokens.shape if use_audio else (1, 1, 0, 0)
    video_shape = video_tokens.shape if use_video else (1, 1, num_frames, 0, 0)
    
    token_types, position_ids = scatter_audio_to_video_by_timesteps(
        audio_shape, video_shape, audio_special_token, audio_step_length, video_step_length, 
        audio_pe_type=audio_pe_type, device=device
    )
    
    interleave_tokens = torch.zeros((batch_size, token_types.shape[0], dim), device=device, dtype=dtype)
    if use_audio:
        interleave_tokens[:, token_types == TOKEN_AUDIO] = audio_tokens.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, dim)
        if audio_special_token:
            interleave_tokens[:, token_types == TOKEN_SOA] = audio_start_token
            interleave_tokens[:, token_types == TOKEN_EOA] = audio_end_token
    if use_video:
        interleave_tokens[:, token_types == TOKEN_VIDEO] = video_tokens.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, dim)
    
    return interleave_tokens, token_types, position_ids, audio_shape, video_shape


def video_audio_deinterleave(tokens, tokens_type, use_audio, use_video):
    r"""
    Args:
        tokens(Tensor): Shape [B, L, C]
        tokens_type(Tensor): Shape [L]
    """
    video_index = (tokens_type == TOKEN_VIDEO)
    audio_index = (tokens_type == TOKEN_AUDIO)
    video_tokens = tokens[:, video_index] if use_video else None
    audio_tokens = tokens[:, audio_index] if use_audio else None

    return video_tokens, audio_tokens


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return super().forward(x.float()).type_as(x)
        ##### Compatible with Open-Sora #####
        return super().forward(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=mmrope_apply(q, freqs),
            k=mmrope_apply(k, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        cross_attn_cls = WAN_CROSSATTENTION_CLASSES[cross_attn_type]
        self.cross_attn = cross_attn_cls(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        **kwargs
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        #     e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32
        e = e.to(x.dtype)
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,  # .float()
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])  # .float()
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class ParameterWrapper(nn.Module):
    def __init__(self, parameter: nn.Parameter):
        super().__init__()
        self.param = parameter 
    
    def forward(self) -> torch.Tensor:
        return self.param
    

class AudioVideoAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_audio=False,
        use_video=False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.audio_ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.use_audio = use_audio
        self.use_video = use_video

        # common layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        cross_attn_cls = WAN_CROSSATTENTION_CLASSES[cross_attn_type]
        self.cross_attn = cross_attn_cls(dim, num_heads, (-1, -1), qk_norm, eps)
        # video layers
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        # audio layers
        self.audio_norm2 = WanLayerNorm(dim, eps)
        self.audio_ffn = nn.Sequential(
            nn.Linear(dim, self.audio_ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(self.audio_ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.audio_modulation = ParameterWrapper(nn.Parameter(torch.randn(1, 3, dim) / dim**0.5))

    def forward(
        self,
        x,
        e,
        audio_e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        video_token_num,
        audio_token_num,
        video_shape=None,
        audio_shape=None,
        position_ids=None,
    ):

        assert e.dtype == torch.float32
        e = e.to(x.dtype)
        e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,  # .float()
            freqs)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        # split audio video tokens
        vx, ax = torch.split(x, (video_token_num, audio_token_num), dim=1)
        
        # audio video ffn
        if self.use_video:
            vy = self.ffn(self.norm2(vx) * (1 + e[4]) + e[3])
            vx = vx + vy * e[5]

        if self.use_audio:
            audio_e = audio_e.to(x.dtype)
            audio_e = (self.audio_modulation() + audio_e).chunk(3, dim=1)
            ay = self.audio_ffn(self.audio_norm2(ax) * (1 + audio_e[1]) + audio_e[0])
            ax = ax + ay * audio_e[2]
        
        x = torch.cat((vx, ax), dim=1)
        return x

    def init_audio_from_video_branch(self):
        with torch.no_grad():
            slice_from_modulation = self.modulation[:, 3:, :].clone()
            self.audio_modulation.param.copy_(slice_from_modulation)

            ### ffn, use pca-based method
            W1_v = self.ffn[0].weight.data.T 
            W2_v = self.ffn[2].weight.data.T
            b1_v = self.ffn[0].bias.data
            b2_v = self.ffn[2].bias.data

            self.audio_ffn[0].weight.copy_(W1_v.T)
            self.audio_ffn[2].weight.copy_(W2_v.T)
            self.audio_ffn[0].bias.copy_(b1_v)
            self.audio_ffn[2].bias.copy_(b2_v)

    def free_unused_modules(self):
        if not self.use_video:
            # Delete video-specific modules
            del self.norm2
            del self.ffn
            self.norm2 = None
            self.ffn = None
    
        if not self.use_audio:
            # Delete audio-specific modules
            del self.audio_norm2
            del self.audio_ffn
            del self.audio_modulation
            self.audio_norm2 = None
            self.audio_ffn = None
            self.audio_modulation = None


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock', 'AudioVideoAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 # audio config
                 audio_patch_size=(2, 2), # 80ms per token for 16kHz
                 audio_in_dim=8,
                 audio_out_dim=8,
                 audio_special_token=False,
                 dual_ffn=False, # audio ffn branch
                 init_from_video_branch=False,
                 audio_specific_text_embedding=False,
                 audio_pe_type='interleave_offset',
                 weight_init_from=None,
                 # traing config
                 train_audio_specific_blocks=False,
                 full_finetuning=False,
                 class_drop_prob=0.0,
                 **kwargs
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward nedualrk
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            audio_patch_size (`tuple`, *optional*, defaults to (2, 2)):
                2D patch dimensions for audio embedding (t_patch, f_patch)
            audio_in_dim (`int`, *optional*, defaults to 8):
                Input audio channels (C_in)
            audio_out_dim (`int`, *optional*, defaults to 8):
                Output audio channels (C_out)
            audio_special_token (`bool`, *optional*, defaults to False):
                add special token between video tokens and audio tokens
            weight_init_from (`str`, *optional*, defaults to None):
                Init from weight files
            full_finetuning (`bool`, *optional*, defaults to False):
               fine tuning all transformer blocks.
            class_drop_prob (`float`, *optional*, defaults to 0.0):
                for classifier-free gudiance training
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace', 't2a', 't2av'] # TODO: x-conditional
        self.model_type = model_type

        ########## config output modality #############
        self.use_audio, self.use_video = False, False
        if model_type in ['t2a', 't2av']:
            self.use_audio = True
        if model_type in ['t2v', 'i2v', 'flf2v', 'vace', 't2av']:
            self.use_video = True
        ###############################################

        ##### setup args ##############################
        self.patch_size = patch_size
        self.audio_patch_size = audio_patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.audio_in_dim = audio_in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.audio_out_dim = audio_out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.audio_special_token = audio_special_token
        self.start_of_audio = None # special tokens
        self.end_of_audio = None
        self.audio_pe_type = audio_pe_type
        self.class_drop_prob = class_drop_prob
        self.dual_ffn = dual_ffn
        self.audio_specific_text_embedding = audio_specific_text_embedding
        #########################################

        ###### modality-specific modulars ######
        if self.use_audio:
            self.audio_patch_embedding = nn.Conv2d(
                audio_in_dim, dim, kernel_size=audio_patch_size, stride=audio_patch_size
            )
            self.audio_head = Head(dim, audio_out_dim, audio_patch_size, eps)

            if audio_special_token:
                self.start_of_audio = nn.Parameter(
                    torch.randn((audio_in_dim,), dtype=torch.float32), requires_grad=True)
                self.end_of_audio = nn.Parameter(
                    torch.randn((audio_in_dim, ), dtype=torch.float32), requires_grad=True)
        if self.use_video:
            self.patch_embedding = nn.Conv3d(
                in_dim, dim, kernel_size=patch_size, stride=patch_size)
            self.head = Head(dim, out_dim, patch_size, eps)

            if model_type == 'i2v' or model_type == 'flf2v':
                self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')
        ########################################

        ################ common modules #########
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        if self.dual_ffn and self.audio_specific_text_embedding:
            self.audio_text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
                nn.Linear(dim, dim)
            )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if self.use_audio and dual_ffn:
            self.audio_time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 3))

        ######## Compatible with Open-Sora framework ########
        self.y_embedder = None
        #####################################################

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type in ['t2v', 't2a', 't2av'] else 'i2v_cross_attn'
        if dual_ffn:
            self.blocks = nn.ModuleList([
                AudioVideoAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads,
                    window_size, qk_norm, cross_attn_norm, eps, 
                    use_audio=self.use_audio, use_video=self.use_video, 
                ) for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                WanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads,
                    window_size, qk_norm, cross_attn_norm, eps
                ) for _ in range(num_layers)
            ])
        
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.inv_freqs = inv_freq([d - 4 * (d // 6), 2 * (d  // 6), 2 * (d // 6)])

        # initialize weights
        self.init_weights()

        self.weight_init_from = weight_init_from
        self.load_pretrained_ckpt()

        if dual_ffn:
            if init_from_video_branch:
                self.init_from_video_branch()
            for block in self.blocks:
                block.free_unused_modules()

        # freeze original parameters
        trainable_modules = []
        if self.use_audio and train_audio_specific_blocks:
            trainable_modules.append(self.audio_patch_embedding)
            trainable_modules.append(self.audio_head)
            if self.audio_specific_text_embedding:
                trainable_modules.append(self.audio_text_embedding)
            if self.audio_special_token:
                trainable_modules.append(self.start_of_audio)
                trainable_modules.append(self.end_of_audio)
            if self.dual_ffn:
                for block in self.blocks:
                    trainable_modules.append(block.audio_norm2)
                    trainable_modules.append(block.audio_ffn)
                    trainable_modules.append(block.audio_modulation)
                # print(block)
                trainable_modules.append(self.audio_time_projection)

        requires_grad(self, False)
        for module in trainable_modules:
            requires_grad(module, True)
        
        if full_finetuning:
            requires_grad(self, True)

    def load_pretrained_ckpt(self):
        if self.weight_init_from is not None:
            init_paths = self.weight_init_from
            if isinstance(init_paths, str):
                init_paths = [init_paths]
            for pretrain_path in init_paths:
                load_checkpoint(self, pretrain_path, verbose=False)

    def forward(
        self,
        x,
        t,
        y,   ##### Compatible with Open-Sora #####
        seq_len=None,
        mask=None,
        x_mask=None, #fps=None, height=None, width=None,
        ax_mask=None,
        clip_fea=None,
        x_cond=None,   ##### Compatible with Open-Sora #####
        **kwargs,
    ):
        dtype = self.time_embedding[0].weight.dtype
        device = self.time_embedding[0].weight.device

        if isinstance(x, dict): # multimodal
            vx = None if not self.use_video else x.pop('video', None)
            ax = None if not self.use_audio else x.pop('audio', None)
        elif isinstance(x, torch.Tensor): # compatible for singal modal input
            if self.use_audio:
                vx, ax = None, x
            elif self.use_video:
                vx, ax = x, None

        batch_size = t.shape[0]

        t, y =  t.to(dtype), y.to(dtype) # t: timesteps, shape(bs, 1), y: text conditions, shape(bs, ctx_len, text_dim)
        drop_ = None
        if self.training and self.class_drop_prob > 0:
            # class_drop for classifier-free guidance
            drop_ = torch.rand((t.size(0),), device=device) < self.class_drop_prob
            # y: bs,512,dim, mask: bs, 512
            y[drop_] = kwargs.get("y_null", torch.zeros_like(y)[drop_])
            mask[drop_] = kwargs.get("mask_null", torch.zeros_like(mask)[drop_])

        if self.use_audio:
            assert ax is not None
            ax = ax.to(dtype)
            ax = self.audio_patch_embedding(ax) # shape(bs,d,l,m)
            audio_grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in ax]) # shape(bs, 2), same

        if self.use_video:
            assert vx is not None
            vx = vx.to(dtype)

            if self.model_type == 'i2v' or self.model_type == 'flf2v':
                assert clip_fea is not None and x_cond is not None
            vx = self.patch_embedding(vx) # shape(bs,d,t/1,h/2,w/2)
            video_grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in vx]) # shape(bs, 3), same

            if x_cond is not None:
                vx = [torch.cat([u.unsqueeze(0), v], dim=0) for u, v in zip(vx, x_cond)]

        # prepare input embeddings
        num_frames = (int(kwargs.pop('num_frames', [33])[0]) - 1) // 4 + 1

        # compatible with single-modality training
        audio_shape = ax.shape if self.use_audio else (1, 1, 0, 0)
        video_shape = vx.shape if self.use_video else (1, 1, num_frames, 0, 0)
        # audio_step_length = 1000.0 / float(kwargs.get('audio_fps', [16000])[0]) * 160 * 4 * self.audio_patch_size[0]
        # video_step_length = 1000.0 / float(kwargs.get('fps', [16])[0]) * 4 * self.patch_size[0]

        video_position_ids, audio_position_ids = get_video_audio_position_ids(
            video_shape, audio_shape, audio_pe_type=self.audio_pe_type, device=device
        )
        position_ids = torch.cat((video_position_ids, audio_position_ids), dim=0)
        video_token_num = math.prod(video_shape[2:])
        audio_token_num = math.prod(audio_shape[2:])
        if self.use_audio and self.use_video:
            avx = torch.cat((
                vx.permute(0, 2, 3, 4, 1).view(batch_size, video_token_num, -1),
                ax.permute(0, 2, 3, 1).view(batch_size, audio_token_num, -1)), dim=1)
        elif self.use_audio:
            avx = ax.permute(0, 2, 3, 1).view(batch_size, audio_token_num, -1)
        else:
            avx = vx.permute(0, 2, 3, 4, 1).view(batch_size, video_token_num, -1)

        if self.inv_freqs[0].device != device:
            self.inv_freqs = [inv_freq.to(device) for inv_freq in self.inv_freqs]

        freqs = rope_cos_sin(self.inv_freqs, position_ids)

        ##### Compatible with Open-Sora #####
        if seq_len is None:
            # sp_size = kwargs.pop('sp_size', 1)
            seq_len = avx.size(1) # bs,n,d
        #####################################
        seq_lens = torch.tensor([u.size(0)  for u in avx], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        avx = torch.cat([avx, avx.new_zeros(avx.size(0), seq_len - avx.size(1), avx.size(2))], dim=1)

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
            if self.use_audio:
                audio_e0 = self.audio_time_projection(e).unflatten(1, (3, self.dim))
            else:
                audio_e0 = None

        # context
        context_lens = None
        ##### Compatible with Open-Sora #####
        # context_lens = mask.gt(0).sum(dim=1).long()
        y[mask == 0] = 0.
        audio_context = None
        context = self.text_embedding(y)
        if self.use_audio and self.audio_specific_text_embedding:
            audio_context = self.audio_text_embedding(y)
            context = torch.concat([context, audio_context], dim=1) if self.use_video else audio_context

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            audio_e=audio_e0,
            seq_lens=seq_lens,
            grid_sizes=None, # useless
            freqs=freqs,
            context=context,
            context_lens=context_lens,
            video_token_num=video_token_num,
            audio_token_num=audio_token_num,
            video_shape=video_shape,
            audio_shape=audio_shape,
        )

        for block in self.blocks:
            if self.training:
                avx = auto_grad_checkpoint(block, avx, **kwargs)
            else:
                avx = block(avx, **kwargs)

        # interleave -> video, audio
        vx, ax = torch.split(avx, (video_token_num, audio_token_num), dim=1)

        # head
        if self.use_video:
            vx = self.head(vx, e)
            vx = self.unpatchify(vx, video_grid_sizes)
            # cast to float32 for better accuracy
            vx = torch.stack(vx).to(torch.float32)
            # reverse the velocity for compatibility
            vx = -vx

        if self.use_audio:
            ax = self.audio_head(ax, e)
            ax = self.unpatchify_audio(ax, audio_grid_sizes)
            ax = torch.stack(ax).to(torch.float32)
            ax = -ax

        ret = {
            'video': vx if self.use_video else None,
            'audio': ax if self.use_audio else None,
        }

        if isinstance(x, torch.Tensor):
            if self.use_video: ret = ret['video']
            elif self.use_audio: ret = ret['audio']

        return ret

    @torch.no_grad()
    def init_from_video_branch(self):
        for block in self.blocks:
            block.init_audio_from_video_branch()
        if self.use_audio:
            self.audio_time_projection[1].weight.copy_(self.time_projection[1].weight[self.dim * 3:, :].clone())
            self.audio_time_projection[1].bias.copy_(self.time_projection[1].bias[self.dim * 3:].clone())
        if self.audio_specific_text_embedding:
            self.audio_text_embedding[0].weight.copy_(self.text_embedding[0].weight.clone())
            self.audio_text_embedding[0].bias.copy_(self.text_embedding[0].bias.clone())
            self.audio_text_embedding[2].weight.copy_(self.text_embedding[2].weight.clone())
            self.audio_text_embedding[2].bias.copy_(self.text_embedding[2].bias.clone())

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def unpatchify_audio(self, x, grid_sizes):
        c = self.audio_out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.audio_patch_size, c)
            u = torch.einsum('fmpqc->cfpmq', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.audio_patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings & heads
        if self.use_video:
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
            nn.init.zeros_(self.head.head.weight)
        if self.use_audio:
            nn.init.xavier_uniform_(self.audio_patch_embedding.weight.flatten(1))
            nn.init.zeros_(self.audio_head.head.weight) 

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
