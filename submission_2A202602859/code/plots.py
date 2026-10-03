"""plots.py — HOÀN THIỆN. Ảnh biểu đồ là sản phẩm nộp (README mục 6).

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # an toàn khi chạy headless (Colab/Kaggle/script)
import matplotlib.pyplot as plt
import numpy as np


def _cfg_label(cfg: dict) -> str:
    """Chuỗi mô tả ngắn gọn cấu hình để đưa vào tiêu đề ảnh."""
    return (f"opt={cfg.get('optimizer')} lr={cfg.get('lr')} bs={cfg.get('batch')} "
            f"ep={cfg.get('epochs')} hidden={'-'.join(map(str, cfg.get('hidden', ())))} "
            f"drop={cfg.get('dropout')} clip={cfg.get('clip_norm')} "
            f"prec={cfg.get('precision')} init={cfg.get('init')} loss={cfg.get('loss')} seed={cfg.get('seed')}")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (1) loss
    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", label="train loss (eval mode)", color="tab:blue")
    ax.plot(ep, h["val_loss"], "s-", label="val loss", color="tab:orange")
    if s["best_epoch"] and s["best_epoch"] > 0:
        ax.axvline(s["best_epoch"], ls="--", color="gray", alpha=0.7,
                   label=f"best epoch={s['best_epoch']}")
    ax.set_title("Loss theo epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2) accuracy + macro-F1
    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", label="val accuracy", color="tab:green")
    ax.plot(ep, h["val_macro_f1"], "^-", label="val macro-F1", color="tab:red")
    ax.axhline(0.4876, ls=":", color="black", alpha=0.6, label="đoán đa số (0.4876)")
    ax.set_title("Val accuracy / macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("giá trị"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (3) grad_norm (trước clip)
    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "d-", color="tab:purple", label="||g||₂ TB epoch")
    if "grad_norm_max" in h:
        ax.plot(ep, h["grad_norm_max"], "x--", color="tab:brown", alpha=0.8,
                markersize=4, label="||g||₂ lớn nhất epoch")
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], ls="--", color="red", alpha=0.8, label=f"c = {cfg['clip_norm']}")
    if "clip_frac" in h and any(v > 0 for v in h["clip_frac"]):
        frac = h["clip_frac"]
        ax2 = ax.twinx()
        ax2.bar(ep, frac, alpha=0.18, color="gray", width=0.6)
        ax2.set_ylabel("tỉ lệ bước bị clip", fontsize=8, color="gray")
        ax2.set_ylim(0, 1.05)
    ax.set_title("Grad norm (trước khi clip)")
    ax.set_xlabel("epoch"); ax.set_ylabel("||g||₂ toàn cục"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle(f"{cfg.get('exp_id')} — {_cfg_label(cfg)}\n"
                 f"step0_loss={s['step0_loss']:.4f} | best_val_loss={s['best_val_loss']:.4f} @ep{s['best_epoch']} "
                 f"| val_acc={s['val_acc']:.4f} | val_macroF1={s['val_macro_f1']:.4f}"
                 + ("  [DIVERGED]" if s.get("diverged") else ""),
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png.
    """
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in results:
        cfg, h = r["cfg"], r["history"]
        if metric not in h:
            continue
        ax.plot(h["epoch"], h[metric], "o-", markersize=4, label=cfg.get("exp_id"))
    ax.set_title(title or f"So sánh {metric}")
    ax.set_xlabel("epoch"); ax.set_ylabel(metric)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, class_names, path: str, title: str = "Confusion matrix (eval)") -> None:
    """Vẽ ma trận nhầm lẫn (hàng = nhãn thật, cột = dự đoán), giá trị = số mẫu."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    k = cm.shape[0]
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(k):
        for j in range(k):
            if cm[i, j] > 0:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(losses, path: str, title: str = "", xlabel: str = "step", ylabel: str = "loss") -> None:
    """Đường loss đơn giản — dùng cho phép thử 'quá khớp 20 mẫu'."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(losses)), losses, "-")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
