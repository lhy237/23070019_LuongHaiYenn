"""Shared helpers for the INS3155 midterm project.
Author: Luong Hai Yen (23070019)
"""
import time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path(".")
DATA = ROOT / "data"
RESULTS = ROOT / "results"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed=42):
    np.random.seed(seed)


def load_gray(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    if size is not None:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def load_image(path, gray=False):
    """Đọc ảnh màu BGR hoặc ảnh xám Grayscale chuẩn theo form mẫu."""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image at {path}")
    return img


def list_images(folder):
    ensure_dir(folder)
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def save_csv(rows, path):
    """Lưu danh sách dict ra file CSV."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(str(path), index=False)


def time_ms(fn, *args, repeats=10, **kwargs):
    """Đo thời gian CPU thực thi trung bình (ms) qua nhiều lần lặp."""
    fn(*args, **kwargs)  # warm-up cache
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) / repeats * 1000.0


def run_timed(fn, img, repeats=20):
    out = fn(img)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(img)
    return out, (time.perf_counter() - t0) / repeats * 1000.0


# Các hàm phục vụ Part A
def add_gaussian_noise(img, sigma=25, seed=42):
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.05, seed=42):
    rng = np.random.default_rng(seed)
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(ref, img):
    return float(peak_signal_noise_ratio(ref, img, data_range=255))


def ssim(ref, img):
    return float(structural_similarity(ref, img, data_range=255))


def evaluate(ref, img):
    return {"MSE": mse(ref, img), "PSNR": psnr(ref, img), "SSIM": ssim(ref, img)}