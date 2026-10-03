"""experiments.py — Định nghĩa TẤT CẢ thí nghiệm của lab trong một chỗ.

Được notebook (code/lab.ipynb) và script chạy hàng loạt import để bảo đảm công bằng:
mọi thí nghiệm chỉ đổi MỘT yếu tố so với baseline, cùng split, cùng số epoch (trừ
thí nghiệm đang xét), cùng seed.

Baseline: M-base (256,128), He init, CE, SGD+momentum 0.9, lr=0.05, batch 512, 20 epoch, fp32, seed 1.
"""
from __future__ import annotations

from train import DEFAULT_CFG

BASE = dict(DEFAULT_CFG, lr=0.05, epochs=20, train_eval_subsample=50_000)


def cfg(**kw) -> dict:
    c = dict(BASE)
    c.update(kw)
    return c


def build_experiments() -> list[dict]:
    exps: list[dict] = []

    # --- baseline, 3 seed (đo độ nhiễu) ---
    for s in (1, 2, 3):
        exps.append(cfg(exp_id=f"base-s{s}", group="baseline", seed=s,
                        description=f"Baseline M-base, seed {s}"))

    # --- chủ đề optimizer: mỗi bộ >= 3 lr, so ở lr tốt nhất của nó ---
    for lr in (0.05, 0.1, 0.2):
        exps.append(cfg(exp_id=f"opt-sgd-lr{lr:g}", group="optimizer", optimizer="sgd", lr=lr,
                        description=f"SGD thuần, lr={lr}"))
    for lr in (0.05, 0.1, 0.2):
        exps.append(cfg(exp_id=f"opt-sgdm-lr{lr:g}", group="optimizer", optimizer="sgd_momentum", lr=lr,
                        description=f"SGD+momentum 0.9, lr={lr}"))
    for lr in (1e-3, 3e-3, 1e-2):
        exps.append(cfg(exp_id=f"opt-adam-lr{lr:g}", group="optimizer", optimizer="adam", lr=lr,
                        description=f"Adam, lr={lr}"))
    for lr in (1e-3, 3e-3, 1e-2):
        exps.append(cfg(exp_id=f"opt-adamw-lr{lr:g}", group="optimizer", optimizer="adamw", lr=lr,
                        weight_decay=0.01, description=f"AdamW (wd=0.01), lr={lr}"))

    # --- chủ đề loss: CE vs MSE (cùng lr/epoch/seed) ---
    exps.append(cfg(exp_id="loss-mse-lr0.05", group="loss", loss="mse", lr=0.05,
                    description="MSE trên one-hot (thay CE), lr=0.05"))

    # --- chủ đề hparam ---
    exps.append(cfg(exp_id="hp-batch128", group="hparam", batch=128, description="batch=128"))
    exps.append(cfg(exp_id="hp-batch2048", group="hparam", batch=2048, description="batch=2048"))
    exps.append(cfg(exp_id="hp-wide", group="hparam", hidden=(512, 256), description="M-wide 512-256"))
    exps.append(cfg(exp_id="hp-deep", group="hparam", hidden=(256, 128, 64), description="M-deep 256-128-64"))
    exps.append(cfg(exp_id="hp-wd1e-2", group="hparam", weight_decay=0.01,
                    description="weight_decay=0.01 (SGD+momentum)"))

    # --- chủ đề dropout ---
    for q in (0.1, 0.3, 0.5):
        exps.append(cfg(exp_id=f"drop-{q:g}", group="dropout", dropout=q,
                        description=f"Dropout q={q} sau mỗi ReLU lớp ẩn"))

    # --- chủ đề clipping ---
    exps.append(cfg(exp_id="clip-lr0.05-c1.0", group="clipping", lr=0.05, clip_norm=1.0,
                    description="lr=0.05, clip c=1.0 (dự kiến không kích hoạt)"))
    exps.append(cfg(exp_id="clip-lr1.0-noc", group="clipping", lr=1.0, clip_norm=None,
                    description="lr=1.0, KHÔNG clip (phản chứng)"))
    exps.append(cfg(exp_id="clip-lr1.0-c1.0", group="clipping", lr=1.0, clip_norm=1.0,
                    description="lr=1.0, clip c=1.0"))
    exps.append(cfg(exp_id="clip-lr2.0-noc", group="clipping", lr=2.0, clip_norm=None,
                    description="lr=2.0, KHÔNG clip (phản chứng, dự kiến sụp)"))
    exps.append(cfg(exp_id="clip-lr2.0-c1.0", group="clipping", lr=2.0, clip_norm=1.0,
                    description="lr=2.0, clip c=1.0"))
    exps.append(cfg(exp_id="clip-lr2.0-c0.5", group="clipping", lr=2.0, clip_norm=0.5,
                    description="lr=2.0, clip c=0.5"))

    # --- chủ đề mixed precision (5 epoch cho cả 3 để so công bằng; chạy CPU) ---
    for prec in ("fp32", "fp16", "bf16"):
        exps.append(cfg(exp_id=f"amp-{prec}", group="amp", precision=prec, epochs=5,
                        description=f"Mixed precision {prec}, 5 epoch (CPU)"))

    # --- chủ đề init ---
    for init in ("zeros", "normal", "xavier", "default"):
        exps.append(cfg(exp_id=f"init-{init}", group="init", init=init,
                        description=f"Khởi tạo {init} (thay He)"))

    # --- cấu hình cuối cùng (chọn theo VAL; xem REPORT/notebook để biết lý do) ---
    # Chọn sau khi xem bảng val: Adam 3e-3 là lr tốt nhất trong nhóm optimizer; M-wide và
    # 40 epoch đều cải thiện thêm. Ghép lại thành cấu hình cuối -> val macro-F1 cao nhất.
    exps.append(cfg(exp_id="fin-adam3e3", group="final", optimizer="adam", lr=3e-3,
                    description="M-base + Adam 3e-3, 20ep"))
    exps.append(cfg(exp_id="fin-adam3e3-wide", group="final", hidden=(512, 256),
                    optimizer="adam", lr=3e-3, description="M-wide + Adam 3e-3, 20ep"))
    exps.append(cfg(exp_id="fin-adam3e3-wide-40", group="final", hidden=(512, 256),
                    optimizer="adam", lr=3e-3, epochs=40,
                    description="M-wide + Adam 3e-3, 40ep (CẤU HÌNH CUỐI, chọn theo val)"))
    exps.append(cfg(exp_id="fin-adam3e3-deep-40", group="final", hidden=(256, 128, 64),
                    optimizer="adam", lr=3e-3, epochs=40,
                    description="M-deep + Adam 3e-3, 40ep"))

    return exps


EXPERIMENTS = build_experiments()
