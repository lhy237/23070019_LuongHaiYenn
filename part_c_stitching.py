import gc
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS (PART C - MULTI-VIEW IMAGE STITCHING)
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
cv2.setRNGSeed(RANDOM_SEED)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "partC"
RESULTS_DIR = ROOT_DIR / "results" / "partC"

DETECTORS = ["SIFT", "ORB", "AKAZE"]
LOWE_RATIO_THRESH = 0.75
RANSAC_THRESHOLDS = [1.0, 3.0, 5.0, 8.0]
DEFAULT_RANSAC_THRESH = 3.0
PYRAMID_BANDS = 4
MAX_IMAGE_DIM = 800     # Resize large camera photos to prevent OOM during warping
MAX_CANVAS_DIM = 6000   # Safety guard against degenerate perspective blow-up


# ==============================================================================
# 1. DATA LOADING (4 OVERLAPPING VIEWS)
# ==============================================================================
def generate_fallback_4_views(out_dir: Path) -> list[Path]:
    """Fallback generator if data/partC has fewer than 4 images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    W, H = 1600, 600
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(H):
        canvas[y, :, 0] = np.clip(225 - int(y * 0.25), 0, 255)
        canvas[y, :, 1] = np.clip(185 - int(y * 0.15), 0, 255)
        canvas[y, :, 2] = np.clip(125 + int(y * 0.15), 0, 255)

    mountains = np.array([
        [0, 380], [240, 175], [520, 315], [840, 145],
        [1180, 305], [1440, 185], [1600, 345], [1600, H], [0, H]
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [mountains], (95, 85, 75))
    cv2.rectangle(canvas, (0, 380), (W, H), (65, 115, 75), -1)
    cv2.rectangle(canvas, (0, 440), (W, 535), (55, 55, 55), -1)
    for rx in range(20, W, 70):
        cv2.rectangle(canvas, (rx, 482), (rx + 40, 492), (240, 240, 240), -1)

    for b_x in range(55, 1500, 108):
        b_w = int(rng.integers(65, 92))
        b_h = int(rng.integers(135, 235))
        b_y = 425 - b_h
        col = (int(rng.integers(60, 210)), int(rng.integers(70, 205)), int(rng.integers(80, 225)))
        cv2.rectangle(canvas, (b_x, b_y), (b_x + b_w, 425), col, -1)
        cv2.rectangle(canvas, (b_x, b_y), (b_x + b_w, 425), (25, 25, 25), 2)
        for wy in range(b_y + 12, 410, 22):
            for wx in range(b_x + 10, b_x + b_w - 12, 18):
                cv2.rectangle(canvas, (wx, wy), (wx + 10, wy + 14), (215, 240, 255), -1)
                cv2.rectangle(canvas, (wx, wy), (wx + 10, wy + 14), (20, 20, 20), 1)

    view_w, view_h = 560, 480
    starts_x = [60, 390, 710, 1010]
    angles = [-2.0, -0.6, 0.6, 1.8]
    gains = [0.94, 1.00, 1.05, 0.96]

    saved_paths = []
    for idx, (sx, ang, gain) in enumerate(zip(starts_x, angles, gains), 1):
        patch = canvas[60 : 60 + view_h, sx : sx + view_w].copy()
        M = cv2.getRotationMatrix2D((view_w / 2.0, view_h / 2.0), ang, 1.0)
        warped = cv2.warpAffine(patch, M, (view_w, view_h), borderMode=cv2.BORDER_REFLECT)
        warped = np.clip(warped.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        out_path = out_dir / f"view_{idx}.png"
        cv2.imwrite(str(out_path), warped)
        saved_paths.append(out_path)
    return saved_paths


def load_four_views(data_dir: Path) -> tuple[list[np.ndarray], list[str]]:
    """Load 4 ordered overlapping images from data/partC."""
    data_dir.mkdir(parents=True, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in exts])

    if len(files) < 4:
        print(f"[WARNING] Found {len(files)} images in {data_dir}. Generating 4 fallback views...")
        files = generate_fallback_4_views(data_dir)
    else:
        files = files[:4]

    images, names = [], []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image file: {p}")
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(img)
        names.append(p.name)

    print(f"[INFO] Loaded 4 overlapping views: {names} | Resolution: {images[0].shape[1]}x{images[0].shape[0]}")
    return images, names


# ==============================================================================
# 2. FEATURE EXTRACTION, KNN MATCHING & RANSAC HOMOGRAPHY
# ==============================================================================
def create_detector(detector_name: str):
    """Create traditional feature detector/descriptor and matching distance norm."""
    name = detector_name.upper()
    if name == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2
    elif name == "ORB":
        return cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING
    elif name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported detector: {detector_name}")


def detect_and_match_pair(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    detector_name: str = "SIFT",
    ratio_thresh: float = 0.75,
    ransac_thresh: float = 3.0,
) -> dict:
    """Detect keypoints, run KNN (k=2) + Lowe's ratio test, and estimate RANSAC Homography."""
    t0 = time.perf_counter()
    det, norm_type = create_detector(detector_name)

    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    kp_src, des_src = det.detectAndCompute(gray_src, None)
    kp_dst, des_dst = det.detectAndCompute(gray_dst, None)

    if des_src is None or des_dst is None or len(kp_src) < 4 or len(kp_dst) < 4:
        raise RuntimeError(f"[{detector_name}] Insufficient keypoints detected (< 4).")

    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    raw_matches = bf.knnMatch(des_src, des_dst, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    if len(good_matches) < 4:
        raise RuntimeError(
            f"[{detector_name}] Insufficient matches after Lowe's ratio test ({len(good_matches)} < 4)."
        )

    pts_src = np.float32([kp_src[m.queryIdx].pt for m in good_matches])
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in good_matches])

    H, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        raise RuntimeError(f"[{detector_name}] Homography estimation failed.")

    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(np.sum(inlier_mask))
    if num_inliers < 4:
        raise RuntimeError(f"[{detector_name}] Too few RANSAC inliers ({num_inliers} < 4).")

    # Compute forward reprojection RMSE on inliers
    inlier_src = pts_src[inlier_mask].reshape(-1, 1, 2)
    inlier_dst = pts_dst[inlier_mask].reshape(-1, 1, 2)
    proj_dst = cv2.perspectiveTransform(inlier_src, H)
    reproj_errors = np.linalg.norm(proj_dst - inlier_dst, axis=2).ravel()
    rmse = float(np.sqrt(np.mean(reproj_errors ** 2)))

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "kp_src": kp_src,
        "kp_dst": kp_dst,
        "num_kp_src": len(kp_src),
        "num_kp_dst": len(kp_dst),
        "good_matches": good_matches,
        "num_initial_matches": len(good_matches),
        "inlier_mask": inlier_mask,
        "num_inliers": num_inliers,
        "inlier_ratio": (num_inliers / float(len(good_matches))) * 100.0,
        "rmse": rmse,
        "H": H,
        "pair_time_ms": elapsed_ms,
    }


# ==============================================================================
# 3. WARPING TO CENTER REFERENCE FRAME & BLENDING ALGORITHMS
# ==============================================================================
def compute_canvas_and_warp(images: list[np.ndarray], H_to_ref: list[np.ndarray]):
    """Compute common bounding box canvas for 4 views and warp images + masks."""
    all_corners = []
    for img, H in zip(images, H_to_ref):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(np.int32)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(np.int32)

    canvas_w = int(x_max - x_min)
    canvas_h = int(y_max - y_min)

    if canvas_w > MAX_CANVAS_DIM or canvas_h > MAX_CANVAS_DIM or canvas_w <= 0 or canvas_h <= 0:
        raise RuntimeError(f"Degenerate canvas dimensions ({canvas_w}x{canvas_h}).")

    T_shift = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float64
    )

    warped_imgs, warped_masks = [], []
    kernel = np.ones((3, 3), np.uint8)
    for img, H in zip(images, H_to_ref):
        H_total = T_shift @ H
        w_img = cv2.warpPerspective(
            img, H_total, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        src_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
        w_mask = cv2.warpPerspective(
            src_mask, H_total, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        # Erode 2px border to prevent dark interpolation seam artifacts
        w_mask = cv2.erode(w_mask, kernel, iterations=2)
        warped_imgs.append(w_img)
        warped_masks.append(w_mask)

    return warped_imgs, warped_masks


def blend_overwrite(warped_imgs: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Direct overwrite without blending (shows visible exposure/alignment seams)."""
    canvas = np.zeros_like(warped_imgs[0], dtype=np.uint8)
    for idx in [0, 3, 2, 1]:
        m = masks[idx] > 0
        canvas[m] = warped_imgs[idx][m]
    return canvas


def blend_feathering(warped_imgs: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Distance-transform alpha feathering across overlapping regions."""
    acc_img = np.zeros_like(warped_imgs[0], dtype=np.float32)
    acc_weight = np.zeros(warped_imgs[0].shape[:2], dtype=np.float32)

    for img, m in zip(warped_imgs, masks):
        m_u8 = (m > 0).astype(np.uint8)
        padded = np.pad(m_u8, 1, mode="constant", constant_values=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        w = np.maximum(dist, 1e-3) * m_u8.astype(np.float32)
        acc_img += img.astype(np.float32) * w[..., None]
        acc_weight += w

    valid = acc_weight > 0
    out = np.zeros_like(warped_imgs[0], dtype=np.uint8)
    out[valid] = np.clip(acc_img[valid] / acc_weight[valid, None], 0, 255).astype(np.uint8)
    return out


def blend_multiband_laplacian(
    warped_imgs: list[np.ndarray], masks: list[np.ndarray], num_bands: int = 4
) -> np.ndarray:
    """Multi-band Laplacian Pyramid Blending (Bonus criteria)."""
    H_c, W_c = warped_imgs[0].shape[:2]
    div = 2 ** num_bands
    pad_h = (div - H_c % div) % div
    pad_w = (div - W_c % div) % div

    weights = []
    for m in masks:
        m_u8 = (m > 0).astype(np.uint8)
        padded = np.pad(m_u8, 1, mode="constant", constant_values=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        weights.append(dist * m_u8.astype(np.float32))

    w_stack = np.stack(weights, axis=0)
    w_sum = np.sum(w_stack, axis=0, keepdims=True)
    norm_weights = np.divide(w_stack, w_sum, out=np.zeros_like(w_stack), where=w_sum > 0)

    blended_pyr = None
    for img, w in zip(warped_imgs, norm_weights):
        img_p = cv2.copyMakeBorder(img.astype(np.float32), 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        w_p = cv2.copyMakeBorder(w, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        G_img = [img_p]
        G_w = [w_p[..., None]]
        for _ in range(num_bands):
            G_img.append(cv2.pyrDown(G_img[-1]))
            down_w = cv2.pyrDown(G_w[-1])
            if down_w.ndim == 2:
                down_w = down_w[..., None]
            G_w.append(down_w)

        L_img = [G_img[num_bands]]
        for i in range(num_bands, 0, -1):
            up = cv2.pyrUp(G_img[i], dstsize=(G_img[i - 1].shape[1], G_img[i - 1].shape[0]))
            L_img.append(G_img[i - 1] - up)
        L_img = L_img[::-1]

        if blended_pyr is None:
            blended_pyr = [L_img[l] * G_w[l] for l in range(num_bands + 1)]
        else:
            for l in range(num_bands + 1):
                blended_pyr[l] += L_img[l] * G_w[l]

    res = blended_pyr[num_bands]
    for l in range(num_bands - 1, -1, -1):
        up = cv2.pyrUp(res, dstsize=(blended_pyr[l].shape[1], blended_pyr[l].shape[0]))
        res = up + blended_pyr[l]

    res = res[:H_c, :W_c]
    total_mask = np.any(np.stack(masks, axis=0) > 0, axis=0)
    out = np.zeros_like(warped_imgs[0], dtype=np.uint8)
    out[total_mask] = np.clip(res[total_mask], 0, 255).astype(np.uint8)
    return out


def crop_black_borders(pano: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Crop outer black borders around the valid union mask."""
    union_mask = np.any(np.stack(masks, axis=0) > 0, axis=0).astype(np.uint8)
    ys, xs = np.where(union_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return pano
    return pano[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


# ==============================================================================
# 4. END-TO-END PIPELINE & VISUALIZATION SAVING
# ==============================================================================
def run_full_stitching(
    images: list[np.ndarray],
    detector_name: str,
    ransac_thresh: float = 3.0,
    blend_mode: str = "feathering",
    keep_warped: bool = False,
) -> dict:
    """Stitch 4 overlapping images (I1, I2, I3, I4) using I2 (index 1) as center reference."""
    t_start = time.perf_counter()

    # Pair 1: I1 -> I2, Pair 2: I3 -> I2, Pair 3: I4 -> I3
    m12 = detect_and_match_pair(images[0], images[1], detector_name, LOWE_RATIO_THRESH, ransac_thresh)
    m32 = detect_and_match_pair(images[2], images[1], detector_name, LOWE_RATIO_THRESH, ransac_thresh)
    m43 = detect_and_match_pair(images[3], images[2], detector_name, LOWE_RATIO_THRESH, ransac_thresh)

    # Homographies mapping each view to reference view I2
    H_1_to_2 = m12["H"]
    H_2_to_2 = np.eye(3, dtype=np.float64)
    H_3_to_2 = m32["H"]
    H_4_to_2 = H_3_to_2 @ m43["H"]

    H_list = [H_1_to_2, H_2_to_2, H_3_to_2, H_4_to_2]
    warped_imgs, warped_masks = compute_canvas_and_warp(images, H_list)

    if blend_mode == "overwrite":
        pano = blend_overwrite(warped_imgs, warped_masks)
    elif blend_mode == "multiband":
        pano = blend_multiband_laplacian(warped_imgs, warped_masks, PYRAMID_BANDS)
    else:
        pano = blend_feathering(warped_imgs, warped_masks)

    pano_cropped = crop_black_borders(pano, warped_masks)
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    det, _ = create_detector(detector_name)
    kp_counts = [len(det.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)) for im in images]
    total_keypoints = int(sum(kp_counts))

    pairs = [m12, m32, m43]
    total_initial_matches = int(sum(p["num_initial_matches"] for p in pairs))
    total_inliers = int(sum(p["num_inliers"] for p in pairs))
    mean_inlier_ratio = float(np.mean([p["inlier_ratio"] for p in pairs]))
    mean_rmse = float(np.mean([p["rmse"] for p in pairs]))

    out = {
        "detector": detector_name,
        "ransac_thresh": ransac_thresh,
        "blend_mode": blend_mode,
        "total_keypoints": total_keypoints,
        "avg_keypoints_per_view": round(total_keypoints / 4.0, 1),
        "total_initial_matches": total_initial_matches,
        "total_inliers": total_inliers,
        "inlier_ratio_pct": round(mean_inlier_ratio, 2),
        "mean_reproj_rmse_px": round(mean_rmse, 4),
        "total_time_ms": round(total_time_ms, 2),
        "pairs": pairs,
        "panorama": pano_cropped,
    }
    if keep_warped:
        out["warped_imgs"] = warped_imgs
        out["warped_masks"] = warped_masks
    return out


def save_keypoint_and_match_figures(images: list[np.ndarray], res_dict: dict, out_dir: Path):
    """Save detected keypoints and RANSAC inlier match lines for a detector."""
    det_name = res_dict["detector"]
    det, _ = create_detector(det_name)

    # 1. Keypoints visualization across 4 views
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = det.detect(gray, None)
        vis = cv2.drawKeypoints(
            img, kps, None, color=(0, 230, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        axes[i].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"View {i+1}: {len(kps)} Keypoints", fontsize=11, fontweight="bold")
        axes[i].axis("off")
    plt.suptitle(f"{det_name} Keypoint Detection Across 4 Overlapping Views", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"{det_name}_keypoints_4views.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2. Inlier match lines across the 3 adjacent pairs (I1->I2, I3->I2, I4->I3)
    pair_configs = [
        (0, 1, res_dict["pairs"][0], "Pair (View 1 -> View 2)"),
        (2, 1, res_dict["pairs"][1], "Pair (View 3 -> View 2)"),
        (3, 2, res_dict["pairs"][2], "Pair (View 4 -> View 3)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for idx, (src_i, dst_i, p_info, title_str) in enumerate(pair_configs):
        inlier_matches = [
            m for m, is_in in zip(p_info["good_matches"], p_info["inlier_mask"]) if is_in
        ]
        vis_match = cv2.drawMatches(
            images[src_i], p_info["kp_src"],
            images[dst_i], p_info["kp_dst"],
            inlier_matches[:100], None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        axes[idx].imshow(cv2.cvtColor(vis_match, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(
            f"{det_name} {title_str} | Initial Matches: {p_info['num_initial_matches']} | "
            f"RANSAC Inliers: {p_info['num_inliers']} ({p_info['inlier_ratio']:.1f}%) | "
            f"RMSE: {p_info['rmse']:.3f} px",
            fontsize=10.5, fontweight="bold"
        )
        axes[idx].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"{det_name}_inlier_matches.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images, names = load_four_views(DATA_DIR)

    # Save input 4-view strip
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for i, (im, nm) in enumerate(zip(images, names)):
        axes[i].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Input View {i+1} ({nm})", fontsize=11, fontweight="bold")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "partC_input_4views.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    records = []
    default_panoramas = {}

    for det_name in DETECTORS:
        for r_thresh in RANSAC_THRESHOLDS:
            is_default = (r_thresh == DEFAULT_RANSAC_THRESH)
            keep_w = (det_name == "SIFT" and is_default)
            try:
                res = run_full_stitching(
                    images, det_name, ransac_thresh=r_thresh, blend_mode="feathering", keep_warped=keep_w
                )
                records.append({
                    "Detector": det_name,
                    "RANSAC_Threshold_px": r_thresh,
                    "Blending": "Feathering",
                    "Total_Keypoints_4Views": res["total_keypoints"],
                    "Avg_Keypoints_Per_View": res["avg_keypoints_per_view"],
                    "Total_Initial_Matches": res["total_initial_matches"],
                    "Total_Inliers": res["total_inliers"],
                    "Inlier_Ratio_Pct": res["inlier_ratio_pct"],
                    "Mean_Reproj_RMSE_px": res["mean_reproj_rmse_px"],
                    "Total_Stitching_Time_ms": res["total_time_ms"],
                })

                if is_default:
                    cv2.imwrite(str(RESULTS_DIR / f"{det_name}_panorama_feathering.png"), res["panorama"])
                    save_keypoint_and_match_figures(images, res, RESULTS_DIR)
                    default_panoramas[det_name] = {
                        "panorama": res["panorama"],
                        "total_inliers": res["total_inliers"],
                        "inlier_ratio_pct": res["inlier_ratio_pct"],
                        "mean_reproj_rmse_px": res["mean_reproj_rmse_px"],
                        "total_time_ms": res["total_time_ms"],
                    }

                if keep_w:
                    # Bonus comparison: Overwrite vs. Feathering vs. Multi-Band Laplacian Blending
                    pano_overwrite = crop_black_borders(
                        blend_overwrite(res["warped_imgs"], res["warped_masks"]),
                        res["warped_masks"]
                    )
                    pano_feather = res["panorama"]
                    t_mb0 = time.perf_counter()
                    pano_multiband = crop_black_borders(
                        blend_multiband_laplacian(res["warped_imgs"], res["warped_masks"], PYRAMID_BANDS),
                        res["warped_masks"]
                    )
                    t_mb_ms = (time.perf_counter() - t_mb0) * 1000.0

                    cv2.imwrite(str(RESULTS_DIR / "SIFT_panorama_overwrite.png"), pano_overwrite)
                    cv2.imwrite(str(RESULTS_DIR / "SIFT_panorama_multiband.png"), pano_multiband)

                    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
                    blends = [
                        (pano_overwrite, "1. Direct Overwrite (No Blending) - Visible Exposure & Alignment Seams"),
                        (pano_feather, "2. Distance-Transform Alpha Feathering - Smooth Illumination Transition"),
                        (pano_multiband, f"3. Multi-Band Laplacian Pyramid Blending ({PYRAMID_BANDS} Bands, +{t_mb_ms:.1f} ms) - Best Detail & Seam Suppression"),
                    ]
                    for ax, (im_b, title_b) in zip(axes, blends):
                        ax.imshow(cv2.cvtColor(im_b, cv2.COLOR_BGR2RGB))
                        ax.set_title(title_b, fontsize=11.5, fontweight="bold")
                        ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(RESULTS_DIR / "partC_blending_comparison.png", dpi=160, bbox_inches="tight")
                    plt.close(fig)

                del res
                gc.collect()

            except Exception as exc:
                print(f"[WARNING] {det_name} @ RANSAC={r_thresh}px skipped: {exc}")

    # Save side-by-side panorama comparison across detectors
    if default_panoramas:
        fig, axes = plt.subplots(len(default_panoramas), 1, figsize=(13, 11))
        if len(default_panoramas) == 1:
            axes = [axes]
        for ax, (det_name, r) in zip(axes, default_panoramas.items()):
            ax.imshow(cv2.cvtColor(r["panorama"], cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"{det_name} Panorama (RANSAC={DEFAULT_RANSAC_THRESH}px) | "
                f"Inliers: {r['total_inliers']} ({r['inlier_ratio_pct']}%) | "
                f"RMSE: {r['mean_reproj_rmse_px']:.3f} px | Time: {r['total_time_ms']:.1f} ms",
                fontsize=11, fontweight="bold"
            )
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "partC_detector_panoramas_grid.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Save CSV metrics and quantitative charts
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "partC_stitching_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SUCCESS] Saved Part C metrics to: {csv_path}")
    print(df.to_string(index=False))

    # Plot quantitative comparison charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for det_name, marker in zip(DETECTORS, ["o", "s", "^"]):
        sub = df[df["Detector"] == det_name]
        if not sub.empty:
            axes[0].plot(sub["RANSAC_Threshold_px"], sub["Inlier_Ratio_Pct"], marker=marker, linewidth=2, label=det_name)
            axes[1].plot(sub["RANSAC_Threshold_px"], sub["Mean_Reproj_RMSE_px"], marker=marker, linewidth=2, label=det_name)

    axes[0].set_title("Inlier Ratio (%) vs. RANSAC Threshold", fontweight="bold")
    axes[0].set_xlabel("RANSAC Threshold (px)")
    axes[0].set_ylabel("Inlier Ratio (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Inlier Reprojection RMSE (px) vs. RANSAC Threshold", fontweight="bold")
    axes[1].set_xlabel("RANSAC Threshold (px)")
    axes[1].set_ylabel("Mean Reprojection RMSE (px)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    df_def = df[df["RANSAC_Threshold_px"] == DEFAULT_RANSAC_THRESH]
    bars = axes[2].bar(df_def["Detector"], df_def["Total_Stitching_Time_ms"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[2].set_title(f"Total Stitching Time (ms) @ Threshold={DEFAULT_RANSAC_THRESH}px", fontweight="bold")
    axes[2].set_ylabel("Time (ms)")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.6)
    for b in bars:
        h_val = b.get_height()
        axes[2].text(b.get_x() + b.get_width() / 2.0, h_val + 2.0, f"{h_val:.1f} ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "partC_quantitative_charts.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()