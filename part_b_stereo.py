"""Part B - 3D Reconstruction from Stereo Images (25 Points).

Execution:
    python src/part_b_stereo.py
Outputs:
    results/partB/ (epipolar lines, disparity maps, point cloud .ply, metrics .csv)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path(".")
sys.path.insert(0, str(ROOT / "src"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (DATA, RESULTS, ensure_dir, load_image, save_csv, set_seed)

IN_DIR = DATA / "partB"
OUT_DIR = ensure_dir(RESULTS / "partB")

# Parameter grid settings
BM_BLOCK_SIZES = [9, 15, 21]
SGBM_BLOCK_SIZES = [3, 5, 9]
NUM_DISP_OFFSETS = [0, 16, 32]


def safe_imwrite(path, img):
    """Write image via OpenCV and verify the file exists on disk."""
    path_str = str(path)
    ok = cv2.imwrite(path_str, img)
    if not ok or not Path(path_str).exists():
        print(f"[WARNING] Failed to write image: {path_str}", flush=True)
    return ok


def save_figure_to_png(fig, out_path):
    """Export Matplotlib figure via OpenCV RGBA buffer to avoid Windows PIL Errno 22."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    safe_imwrite(out_path, bgr)
    plt.close(fig)


def find_frame_file(folder, stem):
    """Locate frame10/frame11 supporting .png, .jpg, .jpeg."""
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_flo(filename, pts1=None, pts2=None):
    """Read Middlebury .flo ground-truth optical flow file.
    Returns gt_disp (float32, invalid pixels set to NaN) and gt_valid (bool mask).
    Sets d = -u and automatically verifies sign against SIFT inliers.
    """
    with open(str(filename), "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo magic number: {magic}")
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=2 * w * h)
        flow = data.reshape((h, w, 2))

    u = flow[:, :, 0].astype(np.float32)
    v = flow[:, :, 1].astype(np.float32)
    valid = (np.abs(u) < 999.0) & (np.abs(v) < 999.0) & (~np.isnan(u)) & (~np.isinf(u))

    # frame10 is left, frame11 is right: x_R = x_L + u => d = x_L - x_R = -u
    gt_disp = -u.copy()

    # Cross-check sign with SIFT inlier disparities
    if pts1 is not None and pts2 is not None and len(pts1) >= 5:
        d_sift = pts1[:, 0] - pts2[:, 0]
        xs = np.clip(np.round(pts1[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts1[:, 1]).astype(int), 0, h - 1)
        m = valid[ys, xs]
        if np.sum(m) >= 5:
            u_sample = u[ys[m], xs[m]]
            err_neg = np.median(np.abs(-u_sample - d_sift[m]))
            err_pos = np.median(np.abs(+u_sample - d_sift[m]))
            if err_pos < err_neg:
                gt_disp = u.copy()

    gt_disp[~valid] = np.nan
    return gt_disp, valid


def estimate_fundamental(imgL, imgR, ratio=0.75):
    """SIFT + Lowe ratio test + RANSAC Fundamental Matrix + scanline outlier filtering.
    Returns F, pts1_inliers, pts2_inliers, num_good_matches.
    """
    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise ValueError("Not enough keypoints to estimate Fundamental Matrix.")

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    inlier_mask = mask.ravel() == 1
    inliers1 = pts1[inlier_mask]
    inliers2 = pts2[inlier_mask]

    # Filter false horizontal jumps along the same epipolar scanline
    d_sift = inliers1[:, 0] - inliers2[:, 0]
    q25, q75 = np.percentile(d_sift, [25, 75])
    iqr = max(float(q75 - q25), 4.0)
    keep = (d_sift >= q25 - 4.0 * iqr) & (d_sift <= q75 + 4.0 * iqr)
    if np.sum(keep) >= 8:
        inliers1 = inliers1[keep]
        inliers2 = inliers2[keep]
        F_refined, _ = cv2.findFundamentalMat(inliers1, inliers2, cv2.FM_8POINT)
        if F_refined is not None and F_refined.shape == (3, 3):
            F = F_refined

    return F, inliers1, inliers2, len(good)


def estimate_disparity_range(inliers1, inliers2):
    """Estimate minDisparity (can be negative) and base numDisparities (multiple of 16)."""
    d_sift = inliers1[:, 0] - inliers2[:, 0]
    d_low = float(np.percentile(d_sift, 1))
    d_high = float(np.percentile(d_sift, 99))

    min_disp = int(np.floor((d_low - 6.0) / 8.0) * 8)
    span = (d_high + 6.0) - min_disp
    base_num_disp = int(max(32, np.ceil(span / 16.0) * 16))
    return min_disp, base_num_disp


def draw_epilines(img1, img2, pts1, pts2, F, num_lines=25):
    """Draw corresponding epipolar lines on BOTH left and right images."""
    h, w = img1.shape[:2]
    out1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if len(img1.shape) == 2 else img1.copy()
    out2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if len(img2.shape) == 2 else img2.copy()

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    idx_list = np.linspace(0, len(pts1) - 1, min(num_lines, len(pts1))).astype(int)
    set_seed(42)
    colors = np.random.randint(40, 255, (len(idx_list), 3)).tolist()

    for idx, col in zip(idx_list, colors):
        color = tuple(int(c) for c in col)

        l1 = lines1[idx]
        p1 = tuple(np.round(pts1[idx]).astype(int))
        if abs(l1[1]) > 1e-6:
            y0 = int(np.round(-l1[2] / l1[1]))
            y1 = int(np.round(-(l1[2] + l1[0] * w) / l1[1]))
            cv2.line(out1, (0, y0), (w, y1), color, 1, cv2.LINE_AA)
        cv2.circle(out1, p1, 4, color, -1, cv2.LINE_AA)

        l2 = lines2[idx]
        p2 = tuple(np.round(pts2[idx]).astype(int))
        if abs(l2[1]) > 1e-6:
            y0 = int(np.round(-l2[2] / l2[1]))
            y1 = int(np.round(-(l2[2] + l2[0] * w) / l2[1]))
            cv2.line(out2, (0, y0), (w, y1), color, 1, cv2.LINE_AA)
        cv2.circle(out2, p2, 4, color, -1, cv2.LINE_AA)

    return np.hstack([out1, out2])


def sampson_error(F, pts1, pts2):
    """Mean Sampson distance (px) across inlier correspondences."""
    N = len(pts1)
    p1 = np.hstack([pts1, np.ones((N, 1))])
    p2 = np.hstack([pts2, np.ones((N, 1))])
    l2 = (F @ p1.T).T
    l1 = (F.T @ p2.T).T
    num = np.abs(np.sum(p2 * l2, axis=1))
    denom = l2[:, 0]**2 + l2[:, 1]**2 + l1[:, 0]**2 + l1[:, 1]**2 + 1e-12
    return float(np.mean(np.sqrt((num**2) / denom)))


def compute_epipolar_distance(F, pts1, pts2):
    """Mean symmetric point-to-epipolar-line distance (px)."""
    N = len(pts1)
    p1 = np.hstack([pts1, np.ones((N, 1))])
    p2 = np.hstack([pts2, np.ones((N, 1))])
    l2 = (F @ p1.T).T
    l1 = (F.T @ p2.T).T
    num = np.abs(np.sum(p2 * l2, axis=1))
    d2 = num / np.sqrt(l2[:, 0]**2 + l2[:, 1]**2 + 1e-12)
    d1 = num / np.sqrt(l1[:, 0]**2 + l1[:, 1]**2 + 1e-12)
    return float(np.mean(0.5 * (d1 + d2)))


def compute_disparity_bm(imgL, imgR, min_disp=0, num_disp=64, block=15):
    """StereoBM supporting negative minDisparity, scaled by 16.0, invalid set to NaN."""
    bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
    bm.setMinDisparity(min_disp)
    bm.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    bm.setTextureThreshold(10)
    bm.setUniquenessRatio(15)
    disp16 = bm.compute(imgL, imgR)
    invalid_raw = (min_disp - 1) * 16
    disp = disp16.astype(np.float32) / 16.0
    disp[disp16 <= invalid_raw] = np.nan
    return disp


def compute_disparity_sgbm(imgL, imgR, min_disp=0, num_disp=64, block=5):
    """StereoSGBM supporting negative minDisparity, scaled by 16.0, invalid set to NaN."""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )
    disp16 = sgbm.compute(imgL, imgR)
    invalid_raw = (min_disp - 1) * 16
    disp = disp16.astype(np.float32) / 16.0
    disp[disp16 <= invalid_raw] = np.nan
    return disp


def colorize_disparity(disp, vmin=None, vmax=None):
    """Colorize disparity map within [vmin, vmax] using JET colormap; NaN pixels set to black."""
    valid = ~np.isnan(disp)
    norm = np.zeros(disp.shape, dtype=np.uint8)
    if np.any(valid):
        if vmin is None:
            vmin = float(np.nanpercentile(disp, 2))
        if vmax is None:
            vmax = float(np.nanpercentile(disp, 98))
        if vmax <= vmin:
            vmax = vmin + 1.0
        clipped = np.clip(disp[valid], vmin, vmax)
        norm[valid] = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    cmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cmap[~valid] = [0, 0, 0]
    return cmap


def evaluate_disparity_metrics(disp, gt_disp=None, gt_valid=None):
    """Compute Valid Disparity Ratio (%), RMSE (px), and Bad-Pixel Rates (1px & 2px)."""
    valid_mask = ~np.isnan(disp)
    valid_ratio = float(np.mean(valid_mask) * 100.0)
    if gt_disp is None or gt_valid is None:
        return valid_ratio, None, None, None
    eval_mask = gt_valid & valid_mask & (~np.isnan(gt_disp))
    if np.sum(eval_mask) == 0:
        return valid_ratio, None, None, None
    diff = np.abs(disp[eval_mask] - gt_disp[eval_mask])
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bad1 = float(np.mean(diff > 1.0) * 100.0)
    bad2 = float(np.mean(diff > 2.0) * 100.0)
    return valid_ratio, rmse, bad1, bad2


def disparity_to_pointcloud(disp, imgL, f, baseline, cx, cy, doffs=None):
    """Reconstruct 3D point cloud Z = f*B/(d + doffs), compensating crop offset doffs."""
    valid = (~np.isnan(disp)) & (~np.isinf(disp))
    y_idx, x_idx = np.where(valid)
    d = disp[valid]

    if doffs is None:
        d_min = float(np.percentile(d, 1))
        doffs = max(0.0, -d_min + 20.0)

    d_eff = d + doffs
    pos = d_eff > 1.0
    y_idx, x_idx, d_eff = y_idx[pos], x_idx[pos], d_eff[pos]

    Z = (f * baseline) / d_eff
    X = (x_idx - cx) * Z / f
    Y = (y_idx - cy) * Z / f

    z_low, z_high = np.percentile(Z, [1.0, 99.0])
    keep = (Z >= z_low) & (Z <= z_high)
    pts3d = np.stack([X[keep], Y[keep], Z[keep]], axis=-1)

    if len(imgL.shape) == 2:
        cols_gray = imgL[y_idx[keep], x_idx[keep]]
        cols_rgb = np.repeat(cols_gray[:, None], 3, axis=1)
    else:
        cols_bgr = imgL[y_idx[keep], x_idx[keep]]
        cols_rgb = cols_bgr[:, [2, 1, 0]]
    return pts3d, cols_rgb


def save_point_cloud_ply(filename, pts3d, cols_rgb):
    """Save 3D point cloud to standard ASCII .ply file."""
    with open(str(filename), "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts3d, cols_rgb):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def visualize_point_cloud_views(out_path, pts3d, cols_rgb, title="3D Point Cloud Multi-View"):
    """Render 3-angle screenshots of the reconstructed 3D point cloud."""
    if len(pts3d) > 30000:
        set_seed(42)
        idx = np.random.choice(len(pts3d), 30000, replace=False)
        pts = pts3d[idx]
        cols = cols_rgb[idx] / 255.0
    else:
        pts = pts3d
        cols = cols_rgb / 255.0

    fig = plt.figure(figsize=(15, 5))
    views = [
        (-60, 20, "Angle 1 (Isometric: Azim=-60, Elev=20)"),
        (0, 50, "Angle 2 (Top-Frontal: Azim=0, Elev=50)"),
        (60, 20, "Angle 3 (Opposite: Azim=60, Elev=20)")
    ]

    for i, (azim, elev, v_title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=1, alpha=0.65)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(v_title, fontsize=9)
        ax.set_xlabel("X (rel. units)", fontsize=8)
        ax.set_ylabel("Z (rel. depth)", fontsize=8)
        ax.set_zlabel("Y (rel. units)", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_figure_to_png(fig, out_path)


def measure_disparity_time(fn, imgL, imgR, min_disp, num_disp, block, repeats=3):
    """Measure execution time in ms (minimum over 3 runs after 1 warm-up)."""
    res = fn(imgL, imgR, min_disp=min_disp, num_disp=num_disp, block=block)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(imgL, imgR, min_disp=min_disp, num_disp=num_disp, block=block)
        times.append((time.perf_counter() - t0) * 1000.0)
    return res, float(min(times))


def main():
    set_seed(42)
    datasets = ["Venus", "Teddy"]
    rows = []

    print("[INFO] Bắt đầu Part B: Tái dựng 3D từ ảnh Stereo (Venus & Teddy)...", flush=True)

    for d_name in datasets:
        folder = IN_DIR / d_name
        if not folder.exists():
            print(f"[CẢNH BÁO] Không tìm thấy thư mục: {folder}", flush=True)
            continue

        left_path = find_frame_file(folder, "frame10")
        right_path = find_frame_file(folder, "frame11")
        flo_path = folder / "flow10.flo"

        if left_path is None or right_path is None:
            print(f"[CẢNH BÁO] Thiếu ảnh frame10 hoặc frame11 trong {folder}", flush=True)
            continue

        print(f"\n==========================================", flush=True)
        print(f"Xử lý Dataset: {d_name}", flush=True)
        print(f"Ảnh trái:  {left_path.name}", flush=True)
        print(f"Ảnh phải:  {right_path.name}", flush=True)
        print(f"Ground truth: {flo_path.name if flo_path.exists() else 'Không có (Teddy)'}", flush=True)
        print(f"==========================================", flush=True)

        imgL_color = load_image(left_path, gray=False)
        imgR_color = load_image(right_path, gray=False)
        imgL_gray = load_image(left_path, gray=True)
        imgR_gray = load_image(right_path, gray=True)

        # 1. Estimate F, draw epipolar lines, compute Sampson error and disparity search range
        print(f"[1/4] Ước lượng F, vẽ Epipolar Lines và xác định dải Disparity...", flush=True)
        F, inliers1, inliers2, num_matches = estimate_fundamental(imgL_gray, imgR_gray, ratio=0.75)
        sampson_err = sampson_error(F, inliers1, inliers2)
        epipolar_dist = compute_epipolar_distance(F, inliers1, inliers2)
        min_disp, base_num_disp = estimate_disparity_range(inliers1, inliers2)
        num_disp_list = [base_num_disp + off for off in NUM_DISP_OFFSETS]

        print(f"   Matches: {num_matches} | Inliers: {len(inliers1)} | Sampson Err: {sampson_err:.3f} px | Epipolar Dist: {epipolar_dist:.3f} px", flush=True)
        print(f"   Ước lượng minDisparity = {min_disp}, khảo sát numDisparities = {num_disp_list}", flush=True)

        epipolar_img = draw_epilines(imgL_gray, imgR_gray, inliers1, inliers2, F, num_lines=25)
        safe_imwrite(OUT_DIR / f"{d_name}_epipolar_lines.png", epipolar_img)

        # 2. Read Ground Truth .flo (if available) and verify sign against SIFT inliers
        gt_disp, gt_valid = None, None
        vmin, vmax = float(min_disp), float(min_disp + base_num_disp)
        if flo_path.exists():
            gt_disp, gt_valid = read_flo(str(flo_path), inliers1, inliers2)
            vmin = float(np.nanpercentile(gt_disp, 2))
            vmax = float(np.nanpercentile(gt_disp, 98))
            safe_imwrite(OUT_DIR / f"{d_name}_gt_disparity.png", colorize_disparity(gt_disp, vmin, vmax))

        # 3. Evaluate StereoBM and StereoSGBM across parameter grid
        print(f"[2/4] Khảo sát tham số StereoBM và StereoSGBM...", flush=True)
        best_sgbm_disp = None
        best_sgbm_score = float("inf")

        for b_size in BM_BLOCK_SIZES:
            for n_disp in num_disp_list:
                disp_bm, t_bm = measure_disparity_time(
                    compute_disparity_bm, imgL_gray, imgR_gray, min_disp, n_disp, b_size
                )
                v_ratio, rmse, bad1, bad2 = evaluate_disparity_metrics(disp_bm, gt_disp, gt_valid)

                rows.append({
                    "Dataset": d_name, "Method": "StereoBM", "BlockSize": b_size,
                    "MinDisparity": min_disp, "NumDisparities": n_disp,
                    "Matches": num_matches, "Inliers": len(inliers1),
                    "ValidRatio_%": round(v_ratio, 2),
                    "EpipolarError_px": round(epipolar_dist, 3), "Sampson_px": round(sampson_err, 3),
                    "RMSE_px": round(rmse, 2) if rmse is not None else "N/A",
                    "Bad1_Rate_%": round(bad1, 2) if bad1 is not None else "N/A",
                    "Bad2_Rate_%": round(bad2, 2) if bad2 is not None else "N/A",
                    "Time_ms": round(t_bm, 2)
                })

        for b_size in SGBM_BLOCK_SIZES:
            for n_disp in num_disp_list:
                disp_sgbm, t_sgbm = measure_disparity_time(
                    compute_disparity_sgbm, imgL_gray, imgR_gray, min_disp, n_disp, b_size
                )
                v_ratio, rmse, bad1, bad2 = evaluate_disparity_metrics(disp_sgbm, gt_disp, gt_valid)

                if rmse is not None:
                    score = rmse
                else:
                    score = -v_ratio if (b_size == 5 and n_disp == base_num_disp) else 0.0

                if score < best_sgbm_score or best_sgbm_disp is None:
                    best_sgbm_score = score
                    best_sgbm_disp = disp_sgbm

                rows.append({
                    "Dataset": d_name, "Method": "StereoSGBM", "BlockSize": b_size,
                    "MinDisparity": min_disp, "NumDisparities": n_disp,
                    "Matches": num_matches, "Inliers": len(inliers1),
                    "ValidRatio_%": round(v_ratio, 2),
                    "EpipolarError_px": round(epipolar_dist, 3), "Sampson_px": round(sampson_err, 3),
                    "RMSE_px": round(rmse, 2) if rmse is not None else "N/A",
                    "Bad1_Rate_%": round(bad1, 2) if bad1 is not None else "N/A",
                    "Bad2_Rate_%": round(bad2, 2) if bad2 is not None else "N/A",
                    "Time_ms": round(t_sgbm, 2)
                })

        # 4. Save colorized disparity maps and side-by-side comparison figure
        print(f"[3/4] Xuất ảnh màu Disparity...", flush=True)
        disp_bm_rep = compute_disparity_bm(imgL_gray, imgR_gray, min_disp=min_disp, num_disp=base_num_disp, block=15)
        disp_sgbm_rep = compute_disparity_sgbm(imgL_gray, imgR_gray, min_disp=min_disp, num_disp=base_num_disp, block=5)

        if gt_disp is None:
            vmin = float(np.nanpercentile(disp_sgbm_rep, 2))
            vmax = float(np.nanpercentile(disp_sgbm_rep, 98))

        color_bm = colorize_disparity(disp_bm_rep, vmin, vmax)
        color_sgbm = colorize_disparity(disp_sgbm_rep, vmin, vmax)

        safe_imwrite(OUT_DIR / f"{d_name}_disparity_bm.png", color_bm)
        safe_imwrite(OUT_DIR / f"{d_name}_disparity_sgbm.png", color_sgbm)

        ncols = 5 if gt_disp is not None else 3
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4))
        axes[0].imshow(cv2.cvtColor(imgL_color, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{d_name}: Left Image", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(color_bm, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"StereoBM (k=15, min={min_disp}, n={base_num_disp})", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(color_sgbm, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"StereoSGBM (k=5, min={min_disp}, n={base_num_disp})", fontsize=9)
        axes[2].axis("off")

        if gt_disp is not None:
            color_gt = colorize_disparity(gt_disp, vmin, vmax)
            axes[3].imshow(cv2.cvtColor(color_gt, cv2.COLOR_BGR2RGB))
            axes[3].set_title("Ground Truth Disparity", fontsize=10)
            axes[3].axis("off")

            err_map = np.abs(disp_sgbm_rep - gt_disp)
            im_err = axes[4].imshow(err_map, cmap="magma", vmin=0, vmax=5.0)
            axes[4].set_title("SGBM |Error| (px)", fontsize=10)
            axes[4].axis("off")
            fig.colorbar(im_err, ax=axes[4], fraction=0.046, pad=0.04)

        fig.tight_layout()
        save_figure_to_png(fig, OUT_DIR / f"{d_name}_disparity_comparison.png")

        # 5. Reconstruct 3D point cloud, save .ply, and render multi-view screenshots
        print(f"[4/4] Dựng Point Cloud 3D và chụp ảnh nhiều góc nhìn...", flush=True)
        h, w = imgL_gray.shape[:2]
        f_val = float(w)
        baseline_val = 0.1
        cx_val, cy_val = w / 2.0, h / 2.0

        pts3d, cols_rgb = disparity_to_pointcloud(
            best_sgbm_disp, imgL_color,
            f=f_val, baseline=baseline_val, cx=cx_val, cy=cy_val, doffs=None
        )
        ply_path = OUT_DIR / f"{d_name}_pointcloud_sgbm.ply"
        save_point_cloud_ply(ply_path, pts3d, cols_rgb)
        print(f"   Đã lưu {len(pts3d)} điểm 3D -> {ply_path.name}", flush=True)

        visualize_point_cloud_views(
            OUT_DIR / f"{d_name}_pointcloud_views.png",
            pts3d, cols_rgb,
            title=f"{d_name}: 3D Reconstructed Point Cloud (StereoSGBM)"
        )

    # 6. Save quantitative table to CSV
    if rows:
        save_csv(rows, OUT_DIR / "partB_metrics.csv")
        print(f"\n[THÀNH CÔNG] Đã lưu {OUT_DIR / 'partB_metrics.csv'}", flush=True)

    print(f"[HOÀN THÀNH PART B] Toàn bộ kết quả đã lưu tại {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()