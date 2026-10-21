import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path(".")
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (DATA, RESULTS, add_gaussian_noise, add_salt_pepper,
                   ensure_dir, evaluate, list_images, load_gray, run_timed)

# ------------------------------------------------------------ Cấu hình tham số
IMG_SIZE = 512
KSIZES = [3, 5, 7]
GAUSS_SIGMAS = [0.8, 1.5, 2.5]
LAP_C = [0.25, 0.5, 1.0]          # Hệ số cường độ làm nét c
LAP_PRESMOOTH = [0, 5]            # 0 = không làm mờ, 5 = lọc Gauss 5x5 trước (chuẩn LoG)

NOISES = {
    "gaussian": lambda im: add_gaussian_noise(im, sigma=25, seed=42),
    "salt_pepper": lambda im: add_salt_pepper(im, amount=0.05, seed=42),
}

# Pipeline tối ưu: Median k=3 cho salt_pepper (tránh bị vón cục); Gaussian k=5 cho nhiễu Gauss
PIPELINE = {
    "gaussian": ("Gaussian", "k=5, s=1.5"),
    "salt_pepper": ("Median", "k=3")
}

LAP_KERNELS = {
    "lap4": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),
    "lap8": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),
}


# ------------------------------------------------------------- Các hàm bộ lọc
def laplacian_sharpen(img, c=0.5, kernel="lap4", pre_k=0):
    f = img.astype(np.float32)
    base = cv2.GaussianBlur(f, (pre_k, pre_k), 0) if pre_k else f
    lap = cv2.filter2D(base, cv2.CV_32F, LAP_KERNELS[kernel], borderType=cv2.BORDER_REFLECT)
    return np.clip(f + c * lap, 0, 255).astype(np.uint8)


def denoise_configs():
    cfg = []
    for k in KSIZES:
        cfg.append(("Mean", f"k={k}", lambda im, k=k: cv2.blur(im, (k, k))))
    for k in KSIZES:
        for s in GAUSS_SIGMAS:
            cfg.append(("Gaussian", f"k={k}, s={s}",
                        lambda im, k=k, s=s: cv2.GaussianBlur(im, (k, k), s)))
    for k in KSIZES:
        cfg.append(("Median", f"k={k}", lambda im, k=k: cv2.medianBlur(im, k)))
    return cfg


def sharpen_configs():
    cfg = []
    for kern in LAP_KERNELS:
        for pre in LAP_PRESMOOTH:
            for c in LAP_C:
                lab = f"{kern}, c={c}, pre={'none' if pre == 0 else f'G{pre}'}"
                cfg.append(("Laplacian", lab,
                            lambda im, c=c, kern=kern, pre=pre:
                            laplacian_sharpen(im, c, kern, pre)))
    return cfg


def lookup(cfgs, family, label):
    return next(fn for fam, lab, fn in cfgs if fam == family and lab == label)


# ---------------------------------------------------------------- Xuất ảnh và biểu đồ
def show(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def grid_denoise(out_path, clean, noisy, outs, title):
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    show(axes[0, 0], clean, "Original")
    m = evaluate(clean, noisy)
    show(axes[0, 1], noisy, f"Noisy\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.3f}")
    axes[0, 2].axis("off")
    rows = [("Mean", lambda k: f"k={k}"),
            ("Gaussian", lambda k: f"k={k}, s=1.5"),
            ("Median", lambda k: f"k={k}")]
    for r, (fam, labf) in enumerate(rows, start=1):
        for c, k in enumerate(KSIZES):
            im = outs[(fam, labf(k))]
            m = evaluate(clean, im)
            show(axes[r, c], im, f"{fam} {labf(k)}\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.3f}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def grid_sharpen(out_path, clean, noisy, denoised, tag, title, cfgs):
    items = [
        ("Original", clean),
        ("Noisy", noisy),
        ("Clean + Laplacian\n(lap4, c=1.0)",
         lookup(cfgs, "Laplacian", "lap4, c=1.0, pre=none")(clean)),
        ("Noisy + Laplacian\n(lap4, c=0.5)",
         lookup(cfgs, "Laplacian", "lap4, c=0.5, pre=none")(noisy)),
        ("Noisy + G5 + Laplacian\n(LoG style, c=0.5)",
         lookup(cfgs, "Laplacian", "lap4, c=0.5, pre=G5")(noisy)),
        (f"Denoise ({tag})\n+ Laplacian (lap4, c=0.5)",
         lookup(cfgs, "Laplacian", "lap4, c=0.5, pre=none")(denoised)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, (name, im) in zip(axes.ravel(), items):
        if im is clean:
            show(ax, im, name)
            continue
        m = evaluate(clean, im)
        show(ax, im, f"{name}\nPSNR={m['PSNR']:.2f}  SSIM={m['SSIM']:.3f}")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.25)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def edge_crop_center(img, half=32):
    g = cv2.magnitude(cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0),
                      cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1))
    score = cv2.boxFilter(g, -1, (2 * half, 2 * half))
    score[:half], score[-half:] = 0, 0
    score[:, :half], score[:, -half:] = 0, 0
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return int(y), int(x)


def zoom_figure(out_path, clean, noisy, best, title, half=32):
    y, x = edge_crop_center(clean, half)
    sl = (slice(y - half, y + half), slice(x - half, x + half))
    items = [("Original", clean), ("Noisy", noisy)] + list(best.items())
    fig, axes = plt.subplots(1, len(items), figsize=(3 * len(items), 3.6))
    for ax, (name, im) in zip(axes, items):
        ax.imshow(im[sl], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(name, fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summary_charts(df, out_dir):
    den = df[df.Family.isin(["Mean", "Gaussian", "Median"])].copy()
    den["k"] = den.Params.str.extract(r"k=(\d)").astype(int)
    den = den[(den.Family != "Gaussian") | den.Params.str.contains("s=1.5")]
    for metric, fname, ylabel in [("PSNR", "chart_psnr_vs_kernel.png", "PSNR (dB)"),
                                  ("SSIM", "chart_ssim_vs_kernel.png", "SSIM"),
                                  ("Time_ms", "chart_time_vs_kernel.png", "Time (ms)")]:
        fig, axes = plt.subplots(1, len(NOISES), figsize=(10, 3.8), sharey=True)
        for ax, noise in zip(np.atleast_1d(axes), NOISES):
            for fam in ["Mean", "Gaussian", "Median"]:
                d = den[(den.Noise == noise) & (den.Family == fam)]
                g = d.groupby("k")[metric].mean()
                ax.plot(g.index, g.values, marker="o", label=fam)
            ax.set_title(f"{noise.replace('_', ' ').capitalize()} noise")
            ax.set_xlabel("Kernel size k")
            ax.set_xticks(KSIZES)
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
        np.atleast_1d(axes)[0].legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------- Chạy chính
def main():
    in_dir = DATA / "partA"
    out_dir = ensure_dir(RESULTS / "partA")

    img_files = list_images(in_dir)
    if not img_files:
        print(f"[LOI] Khong tim thay anh trong thu muc {in_dir}!")
        return

    images = {p.stem: load_gray(p, IMG_SIZE) for p in img_files}
    print(f"[INFO] Da tim thay cac anh: {list(images.keys())}")

    d_cfgs, s_cfgs = denoise_configs(), sharpen_configs()
    rows = []

    for name, clean in images.items():
        print(f"[INFO] Dang chay thuc nghiem tren anh: {name}...")
        for noise, noise_fn in NOISES.items():
            noisy = noise_fn(clean)
            cv2.imwrite(str(out_dir / f"{name}_{noise}_noisy.png"), noisy)
            rows.append(dict(Image=name, Noise=noise, Family="Noisy (no filter)",
                             Params="-", Time_ms=0.0, **evaluate(clean, noisy)))

            outs = {}
            for fam, lab, fn in d_cfgs:
                out, ms = run_timed(fn, noisy)
                outs[(fam, lab)] = out
                rows.append(dict(Image=name, Noise=noise, Family=fam, Params=lab,
                                 Time_ms=ms, **evaluate(clean, out)))
            for fam, lab, fn in s_cfgs:
                out, ms = run_timed(fn, noisy)
                rows.append(dict(Image=name, Noise=noise, Family="Laplacian",
                                 Params=lab, Time_ms=ms, **evaluate(clean, out)))

            # Pipeline khử nhiễu rồi mới làm nét
            pf, pl = PIPELINE[noise]
            denoised = outs[(pf, pl)]
            lap = lookup(s_cfgs, "Laplacian", "lap4, c=0.5, pre=none")
            out, ms = run_timed(lambda im: lap(denoised), noisy, repeats=10)
            rows.append(dict(Image=name, Noise=noise, Family="Pipeline",
                             Params=f"{pf} ({pl}) + lap4 c=0.5", Time_ms=ms,
                             **evaluate(clean, out)))

            # Lưu ảnh lưới so sánh
            title = f"{name} - {noise.replace('_', ' ').capitalize()} noise"
            grid_denoise(out_dir / f"{name}_{noise}_grid_denoise.png",
                         clean, noisy, outs, title)
            grid_sharpen(out_dir / f"{name}_{noise}_grid_sharpen.png", clean, noisy,
                         denoised, f"{pf} {pl}", title, s_cfgs)

            # Zoom biên sắc nét
            best = {}
            for fam in ["Mean", "Gaussian", "Median"]:
                key = max((k for k in outs if k[0] == fam),
                          key=lambda k: evaluate(clean, outs[k])["PSNR"])
                best[f"{fam} best\n{key[1]}"] = outs[key]
            zoom_figure(out_dir / f"{name}_{noise}_zoom.png", clean, noisy, best,
                        title + " (zoom on strongest edges)")

        # Thí nghiệm làm nét trên ảnh sạch (kiểm tra Halo và nghịch lý PSNR)
        for fam, lab, fn in s_cfgs:
            out, ms = run_timed(fn, clean)
            rows.append(dict(Image=name, Noise="clean", Family="Laplacian",
                             Params=lab, Time_ms=ms, **evaluate(clean, out)))
        zoom_figure(out_dir / f"{name}_clean_sharpen_zoom.png", clean, clean,
                    {f"lap4 c={c}": lookup(s_cfgs, "Laplacian",
                                           f"lap4, c={c}, pre=none")(clean)
                     for c in LAP_C}, f"{name} - sharpening the clean image")

    # Xuất ra 2 bảng số liệu CSV
    df = pd.DataFrame(rows).round({"MSE": 2, "PSNR": 2, "SSIM": 4, "Time_ms": 3})
    df.to_csv(out_dir / "metrics_all.csv", index=False)

    avg = (df.groupby(["Noise", "Family", "Params"], sort=False)
             [["MSE", "PSNR", "SSIM", "Time_ms"]].mean().round(3).reset_index())
    avg.to_csv(out_dir / "metrics_mean_over_images.csv", index=False)
    summary_charts(df, out_dir)

    print("\n" + "="*50)
    print("HOAN THANH PART A THANH CONG!")
    print(f"Tat ca anh va bang CSV da duoc luu tai: {out_dir}")
    print("="*50)


if __name__ == "__main__":
    main()