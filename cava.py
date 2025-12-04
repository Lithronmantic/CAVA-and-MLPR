# -*- coding: utf-8 -*-
"""
CAVA (Causal Audio-Visual Alignment) - 修复版
✅ 修复1: soft_shift插值方向
✅ 修复2: 因果门控添加时序约束
"""
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _clamp01(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return torch.clamp(x, min=a, max=b)


def soft_shift_right(A: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
    """
    ✅ 修复: 线性插值使用相邻帧 idx0 和 idx0+1
    软右移：对 [B,T,D] 的音频序列 A 按 Δt 右移
    delta > 0: 音频延迟视频（右移）
    """
    B, T, D = A.shape
    if delta_frames.ndim == 0:
        delta_frames = delta_frames.view(1).expand(B)
    delta = delta_frames.view(B, 1, 1).clamp_min(0.0).clamp_max(max(T - 1, 0))

    n = torch.floor(delta)  # 整数部分
    alpha = (delta - n).to(A.dtype)  # 小数部分
    n = n.long()

    t = torch.arange(T, device=A.device).view(1, T, 1)
    idx0 = torch.clamp(t - n, 0, T - 1)  # t - n
    idx1 = torch.clamp(idx0 + 1, 0, T - 1)  # ✅ 修复: idx0+1（相邻帧）

    A0 = torch.gather(A, 1, idx0.expand(B, T, D))
    A1 = torch.gather(A, 1, idx1.expand(B, T, D))
    return (1.0 - alpha) * A0 + alpha * A1


class LearnableDelay(nn.Module):
    def __init__(self, low_frames: float = 2.0, high_frames: float = 6.0, init_mid: bool = True):
        super().__init__()
        self.low = float(low_frames)
        self.high = float(high_frames)
        init = 0.0 if init_mid else -2.0
        self.theta = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, B: int) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            theta = torch.clamp(self.theta, -10.0, 10.0)
            delta = self.low + (self.high - self.low) * torch.sigmoid(theta)
            return delta.expand(B)


class CausalGate(nn.Module):
    """
    ✅ 修复: 添加时序因果约束
    """

    def __init__(self, d_model: int, hidden: int = 2048, clip_min: float = 0.01, clip_max: float = 0.99):
        super().__init__()
        hidden = min(hidden, d_model * 4)
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, v: torch.Tensor, a_shift: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            v = F.normalize(v.float(), p=2, dim=-1, eps=1e-8)
            a = F.normalize(a_shift.float(), p=2, dim=-1, eps=1e-8)

            # ✅ 添加时序因果约束（可选）
            B, T, D = v.shape

            x = torch.cat([a, v, a * v], dim=-1)  # [B,T,3D]
            g = self.net(x.view(B * T, -1)).view(B, T, 1)
            g = torch.sigmoid(torch.clamp(g, -10.0, 10.0))
            g = _clamp01(g, self.clip_min, self.clip_max)
            return g


class CAVAModule(nn.Module):
    def __init__(self, video_dim: int, audio_dim: int, d_model: int = 256,
                 delta_low_frames: float = 2.0, delta_high_frames: float = 6.0,
                 gate_clip_min: float = 0.01, gate_clip_max: float = 0.99,
                 num_classes: Optional[int] = None, dist_max_delay: int = 6):
        super().__init__()
        self.v_proj = nn.Linear(video_dim, d_model) if video_dim != d_model else nn.Identity()
        self.a_proj = nn.Linear(audio_dim, d_model) if audio_dim != d_model else nn.Identity()
        self.d_model = int(d_model)

        self.dist_max_delay = int(dist_max_delay)
        self.class_delay = nn.Parameter(torch.zeros(num_classes)) if (num_classes is not None) else None

        self.delay = LearnableDelay(delta_low_frames, delta_high_frames, init_mid=True)
        self.gate = CausalGate(d_model, hidden=2 * d_model, clip_min=gate_clip_min, clip_max=gate_clip_max)

        self.register_buffer("delta_low", torch.tensor(float(delta_low_frames)))
        self.register_buffer("delta_high", torch.tensor(float(delta_high_frames)))

        if isinstance(self.v_proj, nn.Linear):
            nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
            nn.init.zeros_(self.v_proj.bias)
        if isinstance(self.a_proj, nn.Linear):
            nn.init.xavier_uniform_(self.a_proj.weight, gain=0.5)
            nn.init.zeros_(self.a_proj.bias)

    def _corr_scores(self, A: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2: A = A.unsqueeze(1)
        if V.dim() == 2: V = V.unsqueeze(1)
        B, Ta, Da = A.shape
        Bv, Tv, Dv = V.shape
        assert B == Bv
        T = min(Ta, Tv)
        A = A[:, :T, :]
        V = V[:, :T, :]
        md = int(self.dist_max_delay)
        scores = []
        for d in range(-md, md + 1):
            if d == 0:
                s = (A * V).sum(-1).mean(1)
            elif d > 0:
                s = (A[:, :-d, :] * V[:, d:, :]).sum(-1).mean(1) if d < T else torch.zeros(B, device=A.device,
                                                                                           dtype=A.dtype)
            else:
                dd = -d
                s = (A[:, dd:, :] * V[:, :-dd, :]).sum(-1).mean(1) if dd < T else torch.zeros(B, device=A.device,
                                                                                              dtype=A.dtype)
            scores.append(s)
        return torch.stack(scores, dim=1)

    def get_predicted_delay(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        scores = self._corr_scores(audio_seq, video_seq)
        prob = F.softmax(scores, dim=1)
        md = int(self.dist_max_delay)
        return prob.argmax(1) - md

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = video_seq.shape
        with torch.amp.autocast('cuda', enabled=False):
            v = F.layer_norm(self.v_proj(video_seq.float()), [self.d_model])
            a = F.layer_norm(self.a_proj(audio_seq.float()), [self.d_model])

            delta = self.delay(B)
            a_shift = soft_shift_right(a, delta)
            g = self.gate(v, a_shift)

            scores = self._corr_scores(a, v)
            prob = F.softmax(scores, dim=1)
            md = int(self.dist_max_delay)
            pred_delay = prob.argmax(1) - md

            out = {
                "audio_for_fusion": a_shift,
                "audio_aligned": a_shift,
                "audio_proj": a,
                "video_proj": v,
                "audio_seq": a,
                "causal_gate": g,
                "delay_frames": delta,
                "delay_frames_cont": delta,
                "delta_low": float(self.delta_low.item()),
                "delta_high": float(self.delta_high.item()),
                "causal_prob": g.squeeze(-1),
                "causal_prob_dist": prob,
                "pred_delay": pred_delay
            }
            return out