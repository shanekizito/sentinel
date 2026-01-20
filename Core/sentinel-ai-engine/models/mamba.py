"""
Sentinel Sovereign AI: Mamba State Space Model v1.0
Objective: O(N) Sequence Modeling | Beyond Attention

This module implements the Selective State Space (S6) architecture from:
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

Key Innovations:
1.  Selective Gating: Input-dependent state transitions (Δ, B, C).
2.  Hardware-Aware Scan: CUDA-optimized parallel prefix scan (simulated here).
3.  Causal Conv1D: Replaces positional embeddings with convolutional context.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MambaBlock(nn.Module):
    """
    A single Mamba block: Conv1D -> SSM -> Gated Output.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input Projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal Conv1D
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1 # Causal padding
        )
        
        # SSM Parameters (Selective)
        # A is structured (Diagonal), B and C are input-dependent
        self.A_log = nn.Parameter(torch.log(torch.randn(self.d_inner, d_state)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Input-dependent projections for Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False) # dt, B, C
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True) # Δ = softplus(dt_proj(x_proj(x)))
        
        # Output Projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        B, L, D = x.shape
        
        # 1. Skip Connection & Normalization
        residual = x
        x = self.norm(x)
        
        # 2. Input Projection -> (z, x)
        xz = self.in_proj(x) # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1) # Each is (B, L, d_inner)
        
        # 3. Causal Conv1D
        x = x.transpose(1, 2) # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :L] # Causal crop
        x = x.transpose(1, 2) # (B, L, d_inner)
        x = F.silu(x)
        
        # 4. Selective SSM
        y = self.ssm(x)
        
        # 5. Gated Output
        y = y * F.silu(z)
        
        # 6. Output Projection
        out = self.out_proj(y)
        
        return out + residual

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Selective State Space Model core.
        x: (B, L, d_inner)
        """
        B, L, D = x.shape
        
        # A (Discretized)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        # Δ, B_proj, C_proj (Input-Dependent)
        x_dbc = self.x_proj(x) # (B, L, d_state * 2 + 1)
        dt, B_proj, C_proj = x_dbc.split([1, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (B, L, d_inner)
        
        # Discretization: A_bar = exp(Δ * A), B_bar = Δ * B
        # For efficiency, we use the ZOH (Zero-Order Hold) approximation
        
        # Parallel Scan (Hardware-Aware Algorithm)
        # This is the core innovation of Mamba: a CUDA kernel that computes the
        # recurrence y_t = A_bar * h_{t-1} + B_bar * x_t in parallel.
        # Here, we simulate with a sequential scan for correctness.
        
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            x_t = x[:, t, :] # (B, D)
            dt_t = dt[:, t, :] # (B, D)
            B_t = B_proj[:, t, :] # (B, d_state)
            C_t = C_proj[:, t, :] # (B, d_state)
            
            # A_bar = exp(dt * A) ~ I + dt * A (for small dt)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) # (B, D, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, D, d_state)
            
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            y_t = torch.einsum('bds,bs->bd', h, C_t) # (B, D)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1) # (B, L, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0) # Skip connection (D term)
        
        return y


class SovereignMamba(nn.Module):
    """
    Full Mamba-based Language Model for Sentinel.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, d_state: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)
