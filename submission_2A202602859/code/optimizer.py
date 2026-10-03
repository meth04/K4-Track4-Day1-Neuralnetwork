"""optimizer.py — HOÀN THIỆN. Gom việc chọn bộ tối ưu và cắt gradient.

Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import math

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer không hợp lệ: {name} (chọn trong {OPTIMIZERS})")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    # adamw
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học. Trả về None nếu name là None.

    Hỗ trợ "cosine" (CosineAnnealingLR) và "step" (StepLR). Nếu dùng, ghi vào cột notes của bảng.
    """
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    if name == "step":
        step_size = kwargs.pop("step_size", max(1, total_steps // 3))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, **kwargs)
    raise ValueError(f"scheduler không hỗ trợ: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Giá trị trả về chính là `grad_norm` phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    params = [p for p in params if p.grad is not None]
    if max_norm is None:
        # tính chuẩn toàn cục mà KHÔNG cắt (max_norm=inf)
        total_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if isinstance(total_norm, torch.Tensor):
        total_norm = float(total_norm)
    if not math.isfinite(total_norm):
        total_norm = float("inf")
    return float(total_norm)
