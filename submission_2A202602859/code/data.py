"""data.py — HOÀN THIỆN. Nạp train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên device.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    # Kiểm tra quy ước shape/dtype ở đầu file.
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, X_train_full.shape
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, X_eval.shape
    assert X_train_full.dtype == np.float32 and X_eval.dtype == np.float32
    assert y_train_full.dtype == np.int64 and y_eval.dtype == np.int64
    assert y_train_full.min() >= 0 and y_train_full.max() <= 6
    assert y_eval.min() >= 0 and y_eval.max() <= 6
    assert len(X_train_full) == len(y_train_full) == 464_809, "train phải có 464 809 mẫu"
    assert len(X_eval) == len(y_eval) == len(eval_row_id) == 116_203, "eval phải có 116 203 mẫu"
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed, shuffle=True
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
      -> Rò rỉ thông tin (data leakage): nếu lấy thống kê từ val/eval thì mô hình đã
         "thấy" phân bố của tập đánh giá, khiến val/eval không còn là ước lượng khách quan.
    """
    Xn = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = Xn.mean(axis=0)
    std = Xn.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # tránh chia cho 0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X: 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_out = X.copy().astype(np.float32)
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)

    # 1. tách validation TỪ train
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)

    # 2. chuẩn hoá bằng thống kê CHỈ của phần train còn lại (không dùng val/eval)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # 3. tensor trên device
    def to_t(x_np, dtype):
        return torch.tensor(x_np, dtype=dtype, device=device)

    data = dict(
        X_tr=to_t(X_tr, torch.float32), y_tr=to_t(y_tr, torch.int64),
        X_val=to_t(X_val, torch.float32), y_val=to_t(y_val, torch.int64),
        X_eval=to_t(X_eval, torch.float32), y_eval=to_t(y_eval, torch.int64),
        eval_row_id=np.asarray(eval_row_id),
        standardizer=(mean, std),
    )

    # 4. in kích thước + mốc "đoán lớp đa số" trên val
    majority = np.bincount(y_val, minlength=7).argmax()
    maj_acc = float((y_val == majority).mean())
    print(f"train full : {tuple(X_train_full.shape)}")
    print(f"  -> train : {tuple(X_tr.shape)}")
    print(f"  -> val   : {tuple(X_val.shape)}")
    print(f"eval       : {tuple(X_eval.shape)}")
    print(f"lớp đa số  : {majority}; accuracy 'luôn đoán lớp đa số' trên val = {maj_acc:.4f} (mốc thấp nhất phải vượt)")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể nhỏ hơn batch_size — ta vẫn dùng nó (không bỏ), để không phí dữ liệu.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
