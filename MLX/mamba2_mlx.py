import math
from dataclasses import dataclass
from typing import Union

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Mamba2Config:
    d_model: int  # D
    n_layers: int
    expand_factor: int = 2  # E in paper/comments
    d_state: int = 64  # N in paper/comments (Larger in Mamba2)
    headdim: int = 64  # Head dimension for SSD
    ngroups: int = 1   # Number of groups for GQA-like B/C
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = False # Parameter kept for compatibility, we always use pure SSD
    d_inner: int = 0

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        assert self.nheads % self.ngroups == 0, "nheads must be divisible by ngroups"

def pure_ssd(x, B, C, dt, A_log):
    """
    Pure MLX implementation of Structured State Space Duality (SSD).
    x: (batch, seqlen, nheads, headdim)
    B: (batch, seqlen, ngroups, d_state)
    C: (batch, seqlen, ngroups, d_state)
    dt: (batch, seqlen, nheads)
    A_log: (nheads,)
    """
    batch, seqlen, nheads, headdim = x.shape
    ngroups = B.shape[2]
    d_state = B.shape[3]
    
    A = -mx.exp(A_log) # (nheads,)
    
    # Broadcast B and C to nheads if ngroups < nheads
    if ngroups < nheads:
        heads_per_group = nheads // ngroups
        # reshape to (batch, seqlen, ngroups, 1, d_state)
        # repeat to (batch, seqlen, ngroups, heads_per_group, d_state)
        # flatten to (batch, seqlen, nheads, d_state)
        B = mx.repeat(mx.expand_dims(B, 3), repeats=heads_per_group, axis=3).reshape(batch, seqlen, nheads, d_state)
        C = mx.repeat(mx.expand_dims(C, 3), repeats=heads_per_group, axis=3).reshape(batch, seqlen, nheads, d_state)

    # Decay mask deltaA: (batch, seqlen, nheads)
    deltaA = dt * A.reshape(1, 1, -1)
    cum_deltaA = mx.cumsum(deltaA, axis=1)
    
    cum_deltaA_t = mx.expand_dims(cum_deltaA, 2) # (batch, seqlen, 1, nheads)
    cum_deltaA_s = mx.expand_dims(cum_deltaA, 1) # (batch, 1, seqlen, nheads)
    
    diff = cum_deltaA_t - cum_deltaA_s     # sum_{k=s+1}^{t} deltaA[k]
    decay = mx.exp(diff)
    
    tril_mask = mx.tril(mx.ones((seqlen, seqlen))).reshape(1, seqlen, seqlen, 1)
    decay = decay * tril_mask
    decay = decay.transpose(0, 3, 1, 2) # (batch, nheads, t_seqlen, s_seqlen)
    
    B_scaled = B * mx.expand_dims(dt, -1)
    
    C_trans = C.transpose(0, 2, 1, 3) # (batch, nheads, seqlen, d_state)
    B_trans = B_scaled.transpose(0, 2, 1, 3)
    
    scores = C_trans @ B_trans.transpose(0, 1, 3, 2)
    attn = scores * decay
    
    x_trans = x.transpose(0, 2, 1, 3)
    y = attn @ x_trans
    
    y = y.transpose(0, 2, 1, 3) # (batch, seqlen, nheads, headdim)
    return y


class Mamba2Block(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config

        self.d_in_proj = 2 * config.d_inner + 2 * config.ngroups * config.d_state + config.nheads
        self.in_proj = nn.Linear(config.d_model, self.d_in_proj, bias=config.bias)

        conv_dim = config.d_inner + 2 * config.ngroups * config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            padding=config.d_conv - 1,
            groups=conv_dim,
        )
        
        # dt bias
        dt = mx.random.uniform(shape=[config.nheads]) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        dt = mx.clip(mx.exp(dt), a_min=config.dt_init_floor, a_max=None)
        inv_dt = dt + mx.log1p(-mx.exp(-dt))
        self.dt_bias = inv_dt

        # A log parameter
        A = mx.random.uniform(1.0, 16.0, shape=[config.nheads])
        self.A_log = mx.log(A)

        self.D = mx.ones([config.nheads])
        self.norm = nn.RMSNorm(config.d_inner)
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def __call__(self, u):
        # u : (B, L, D)
        batch, seqlen, _ = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        
        split_sizes = [
            self.config.d_inner, 
            self.config.d_inner + 2 * self.config.ngroups * self.config.d_state, 
            self.config.nheads
        ]
        z, xBC, dt = mx.split(zxbcdt, indices_or_sections=[split_sizes[0], split_sizes[0]+split_sizes[1]], axis=-1)
        
        dt = nn.softplus(dt + self.dt_bias)  # (B, L, nheads)
        
        # 1D conv branch
        xBC = nn.silu(self.conv1d(xBC)[:, :seqlen, :])
        
        split_sizes2 = [
            self.config.d_inner,
            self.config.ngroups * self.config.d_state,
        ]
        x, B, C = mx.split(xBC, indices_or_sections=[split_sizes2[0], split_sizes2[0]+split_sizes2[1]], axis=-1)
        
        x = x.reshape(batch, seqlen, self.config.nheads, self.config.headdim)
        B = B.reshape(batch, seqlen, self.config.ngroups, self.config.d_state)
        C = C.reshape(batch, seqlen, self.config.ngroups, self.config.d_state)
        
        y_ssm = pure_ssd(x, B, C, dt, self.A_log)
        
        # Multiply by D skip inside SSM logic
        # D is (nheads,), x is (B, L, nheads, headdim)
        y_ssm = y_ssm + x * self.D.reshape(1, 1, -1, 1)
        
        y_ssm = y_ssm.reshape(batch, seqlen, self.config.d_inner)
        
        y = self.norm(y_ssm) * nn.silu(z)
        out = self.out_proj(y)
        
        return out


class ResidualBlock(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.mixer = Mamba2Block(config)
        self.norm = nn.RMSNorm(config.d_model)

    def __call__(self, x):
        return self.mixer(self.norm(x)) + x


class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.layers = [ResidualBlock(config) for _ in range(config.n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
