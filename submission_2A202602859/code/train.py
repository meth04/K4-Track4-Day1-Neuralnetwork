"""train.py — HOÀN THIỆN. Đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán, ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # chọn bằng val (xem notebook), không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" | "step"
    train_eval_subsample=50_000,   # train_loss đo ở eval() trên tập con CỐ ĐỊNH này
    eval_batch=8192,
    seed=1,
)

# Chỉ số cần thiết để tính macro-F1 (giống scripts/evaluate.py)
N_CLASSES = 7


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits (eval mode, không dropout)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        out.append(logits.argmax(dim=1))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở eval() (dropout tắt), no_grad.

    loss = tổng loss / N (nhất quán giữa các batch lớn nhỏ).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    n = len(X)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name, reduction="sum")
        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)
    return dict(loss=total_loss / n, acc=correct / n, macro_f1=macro_f1_from_confusion(cm))


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y, lấy trung bình trên MỌI phần tử (B*7),
               đúng như nn.MSELoss mặc định (không có hệ số 1/2).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).to(logits.dtype)
        return F.mse_loss(logits, y_onehot, reduction=reduction)
    raise ValueError(f"loss không hợp lệ: {loss_name}")


def run_experiment(cfg: dict, data: dict, logger=None) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Trả về dict: {"cfg", "history", "summary", "best_state"}.
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = dict(cfg)
    set_seed(cfg["seed"])
    device = data["X_tr"].device

    # ---- 0. model + optimizer
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    expected = EXPECTED_PARAMS.get(tuple(cfg["hidden"]))
    if expected is not None:
        assert n_params == expected, f"số tham số {n_params} != {expected} cho hidden={tuple(cfg['hidden'])}"

    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], momentum=cfg["momentum"],
    )

    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]

    # generator riêng cho việc xáo lô -> tái lập được
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"] + 12345)

    steps_per_epoch = int(np.ceil(len(X_tr) / cfg["batch"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps)

    precision = cfg.get("precision", "fp32")
    use_fp16 = precision == "fp16"
    use_amp = precision in ("fp16", "bf16")
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=True) if (use_fp16 and device.type == "cuda") else None
    # Trên CPU không có CUDA autocast; dùng torch.autocast(device.type, ...) khi có hỗ trợ.

    def amp_ctx():
        if not use_amp:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)

    # ---- 1. loss bước 0 (TRƯỚC bước cập nhật đầu tiên), kỳ vọng ≈ ln 7
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"], cfg["eval_batch"])["loss"]

    # tập con CỐ ĐỊNH của train để đo train_loss ở eval mode (so sánh được với val)
    n_sub = min(cfg.get("train_eval_subsample") or len(X_tr), len(X_tr))
    X_tr_eval = X_tr[:n_sub]
    y_tr_eval = y_tr[:n_sub]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = dict(epoch=[], train_loss=[], val_loss=[], val_acc=[], val_macro_f1=[],
                   grad_norm=[], grad_norm_max=[], clip_frac=[], epoch_time_s=[])
    best = dict(val_loss=float("inf"), epoch=-1, state=None, val_acc=0.0, val_macro_f1=0.0)
    diverged = False

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()

        model.train()
        running_loss, seen, gn_sum, gn_n = 0.0, 0, 0.0, 0
        gn_max, clipped = 0.0, 0
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"], reduction="mean")
            if not torch.isfinite(loss):
                diverged = True
                break
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                optimizer.step()
            gn_sum += gn
            gn_max = max(gn_max, gn)
            if cfg["clip_norm"] is not None and gn > cfg["clip_norm"]:
                clipped += 1
            gn_n += 1
            running_loss += float(loss.item()) * len(xb)
            seen += len(xb)
        if scheduler is not None:
            scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.time() - t0

        # đánh giá cuối epoch (eval mode)
        tr_metrics = evaluate(model, X_tr_eval, y_tr_eval, cfg["loss"], cfg["eval_batch"])
        val_metrics = evaluate(model, X_val, y_val, cfg["loss"], cfg["eval_batch"])

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(gn_sum / max(gn_n, 1))
        history["grad_norm_max"].append(gn_max)
        history["clip_frac"].append(clipped / max(gn_n, 1))
        history["epoch_time_s"].append(epoch_time)

        if val_metrics["loss"] < best["val_loss"]:
            best.update(val_loss=val_metrics["loss"], epoch=epoch,
                        state=copy.deepcopy(model.state_dict()),
                        val_acc=val_metrics["acc"], val_macro_f1=val_metrics["macro_f1"])

        if logger is not None:
            logger.log({
                "epoch": epoch,
                "train_loss": tr_metrics["loss"], "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"], "val_macro_f1": val_metrics["macro_f1"],
                "grad_norm": history["grad_norm"][-1], "epoch_time_s": epoch_time,
            }, step=epoch)

        if diverged:
            print(f"[{cfg['exp_id']}] DIVERGED ở epoch {epoch} (loss NaN/inf) — dừng sớm.")
            break

    peak_mem_MB = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else float("nan")
    times = history["epoch_time_s"]

    summary = dict(
        step0_loss=step0_loss,
        best_val_loss=best["val_loss"],
        best_epoch=best["epoch"],
        final_train_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        final_val_loss=history["val_loss"][-1] if history["val_loss"] else float("nan"),
        val_acc=best["val_acc"],
        val_macro_f1=best["val_macro_f1"],
        time_per_epoch_s=float(np.mean(times)) if times else float("nan"),
        peak_mem_MB=peak_mem_MB,
        diverged=bool(diverged),
    )

    return dict(cfg=cfg, history=history, summary=summary,
                best_state=best["state"] if best["state"] is not None else copy.deepcopy(model.state_dict()))


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    import csv
    row_id = np.asarray(row_id).astype(np.int64)
    preds = np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds), (len(row_id), len(preds))
    assert preds.min() >= 0 and preds.max() <= 6
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "pred"])
        for r, p in zip(row_id.tolist(), preds.tolist()):
            w.writerow([r, p])


def build_model_from_result(result: dict, device: str) -> MLP:
    """Dựng lại model từ best_state trong result."""
    cfg = result["cfg"]
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()
    return model


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str):
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Trả về mảng dự đoán int64 (N,).
    """
    device = data["X_tr"].device
    model = build_model_from_result(result, device)
    preds = predict(model, data["X_eval"], cfg.get("eval_batch", 8192))
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds)} dòng)")
    return preds
