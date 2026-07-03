from __future__ import annotations
import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, gaussian_filter1d
from scipy.stats import wilcoxon
from skimage.transform import rotate

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
METRIC_DIRECTIONS = {
    "solidity": +1,
    "hole_ratio": -1,
    "border_touch_ratio": -1,
    "norm_perimeter": -1,
    "boundary_roughness": -1,
    "fourier_hf_energy": -1,
    "symmetry": +1,
    "width_smoothness": -1,
}
SUMMARY_METRICS = list(METRIC_DIRECTIONS.keys())

@dataclass
class ImageRecord:
    method: str
    abs_path: str
    rel_path: str
    sample_key: str
    class_label: str

def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a) / float(b + eps)

def normalize_token(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\.(png|jpg|jpeg|bmp|tif|tiff|webp)$", "", s)
    s = re.sub(r"\bgroundedsam\b", "", s)
    s = re.sub(r"\btonguesam\b", "", s)
    s = re.sub(r"(^|[_\-\s])gs(?=$|[_\-\s])", r"\1", s)
    s = re.sub(r"(^|[_\-\s])ts(?=$|[_\-\s])", r"\1", s)
    s = re.sub(r"(^|[_\-\s])sam(?=$|[_\-\s])", r"\1", s)
    s = re.sub(r"[_\-\s]+", "_", s).strip("_")
    return s

def infer_class_label(parts: List[str]) -> str:
    joined = "/".join(p.lower() for p in parts)
    if "lymphoma" in joined:
        return "lymphoma"
    if "normal" in joined:
        return "normal"
    return "unknown"

def build_sample_key(root: Path, img_path: Path) -> str:
    rel_parts = list(img_path.relative_to(root).parts)
    norm_parts = [normalize_token(p) for p in rel_parts]
    norm_parts = [p for p in norm_parts if p != ""]
    return "/".join(norm_parts)

def collect_images(root_dir: str, method: str) -> List[ImageRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"{method} 根目录不存在: {root_dir}")
    records: List[ImageRecord] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel_path = str(p.relative_to(root))
        parts = list(p.relative_to(root).parts)
        class_label = infer_class_label(parts)
        sample_key = build_sample_key(root, p)
        records.append(
            ImageRecord(
                method=method,
                abs_path=str(p),
                rel_path=rel_path,
                sample_key=sample_key,
                class_label=class_label,
            )
        )
    return records

def read_image(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return img

def extract_foreground_mask(img: np.ndarray, nonblack_thresh: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
    if img.ndim == 2:
        fg = img > nonblack_thresh
    else:
        rgb = img[..., :3] if img.shape[2] >= 3 else img
        fg = np.max(rgb, axis=2) > nonblack_thresh
    fg = fg.astype(bool)
    h, w = fg.shape
    if fg.sum() == 0:
        return np.zeros((h, w), dtype=bool), {
            "raw_fg_area": 0.0,
            "component_count": 0,
            "hole_ratio": 1.0,
        }
    fg_u8 = fg.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(fg), {
            "raw_fg_area": 0.0,
            "component_count": 0,
            "hole_ratio": 1.0,
        }
    component_count = num_labels - 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    largest = labels == largest_idx
    filled = binary_fill_holes(largest)
    hole_area = float(filled.sum() - largest.sum())
    largest_area = float(max(largest.sum(), 1))
    hole_ratio = hole_area / largest_area
    cleaned = filled.astype(bool)
    return cleaned, {
        "raw_fg_area": float(fg.sum()),
        "component_count": int(component_count),
        "hole_ratio": float(hole_ratio),
    }

def mask_area(mask: np.ndarray) -> float:
    return float(mask.sum())

def find_main_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]
    if len(contour) < 5:
        return None
    return contour.astype(np.float64)

def contour_perimeter(contour: np.ndarray) -> float:
    c = contour.reshape(-1, 1, 2).astype(np.float32)
    return float(cv2.arcLength(c, True))

def contour_area(contour: np.ndarray) -> float:
    c = contour.reshape(-1, 1, 2).astype(np.float32)
    return float(abs(cv2.contourArea(c)))

def contour_solidity(contour: np.ndarray, area_from_mask: float) -> float:
    c = contour.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(c)
    hull_area = float(abs(cv2.contourArea(hull)))
    return safe_div(area_from_mask, hull_area)

def border_touch_ratio(mask: np.ndarray, perimeter: float) -> float:
    h, w = mask.shape
    border_count = (
        int(mask[0, :].sum())
        + int(mask[-1, :].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
    )
    border_count -= int(mask[0, 0]) + int(mask[0, -1]) + int(mask[-1, 0]) + int(mask[-1, -1])
    return safe_div(border_count, perimeter)

def resample_closed_contour(contour: np.ndarray, n_points: int = 256) -> np.ndarray:
    pts = np.asarray(contour, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    seg_lens = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]
    if total_len < 1e-8:
        return np.repeat(pts[:1], n_points, axis=0)
    target = np.linspace(0, total_len, n_points + 1)[:-1]
    x = np.interp(target, cumlen, pts[:, 0])
    y = np.interp(target, cumlen, pts[:, 1])
    return np.stack([x, y], axis=1)

def boundary_roughness(contour: np.ndarray, area: float, n_points: int = 256, sigma: float = 3.0) -> float:
    pts = resample_closed_contour(contour, n_points=n_points)
    xs = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="wrap")
    ys = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="wrap")
    smooth = np.stack([xs, ys], axis=1)
    d = np.sqrt(np.sum((pts - smooth) ** 2, axis=1))
    return float(np.mean(d) / (math.sqrt(area) + 1e-8))

def fourier_hf_energy(contour: np.ndarray, n_points: int = 256, hf_keep_from: float = 0.15) -> float:
    pts = resample_closed_contour(contour, n_points=n_points)
    z = (pts[:, 0] - pts[:, 0].mean()) + 1j * (pts[:, 1] - pts[:, 1].mean())
    F = np.fft.fft(z)
    power = np.abs(F) ** 2
    power[0] = 0.0
    n = len(power)
    half = n // 2
    positive = power[1:half]
    if len(positive) == 0 or positive.sum() < 1e-12:
        return 0.0
    k0 = max(1, int(len(positive) * hf_keep_from))
    hf = positive[k0:].sum()
    total = positive.sum()
    return float(hf / (total + 1e-8))

def principal_axis_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        return 0.0
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    coords = np.stack([x, y], axis=1)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = math.degrees(math.atan2(major[1], major[0]))
    return float(angle)

def rotate_mask_to_vertical(mask: np.ndarray) -> np.ndarray:
    angle = principal_axis_angle(mask)
    rotate_deg = 90.0 - angle
    rot = rotate(
        mask.astype(np.float32),
        angle=rotate_deg,
        resize=True,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return rot > 0.5

def crop_to_bbox(mask: np.ndarray, pad: int = 2) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    y1 = max(0, y1 - pad)
    y2 = min(mask.shape[0] - 1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(mask.shape[1] - 1, x2 + pad)
    return mask[y1 : y2 + 1, x1 : x2 + 1]

def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)

def symmetry_score(mask: np.ndarray) -> float:
    rot = rotate_mask_to_vertical(mask)
    rot = crop_to_bbox(rot, pad=2)
    if rot.sum() == 0:
        return 0.0
    ys, xs = np.where(rot > 0)
    cx = int(round(xs.mean()))
    left = rot[:, :cx]
    right = rot[:, cx + 1 :]
    if left.size == 0 or right.size == 0:
        return 0.0
    left_flip = np.fliplr(left)
    w = min(left_flip.shape[1], right.shape[1])
    if w <= 0:
        return 0.0
    left_flip = left_flip[:, -w:]
    right = right[:, :w]
    return dice_score(left_flip, right)

def width_profile_smoothness(mask: np.ndarray, target_len: int = 128) -> float:
    rot = rotate_mask_to_vertical(mask)
    rot = crop_to_bbox(rot, pad=2)
    if rot.sum() == 0:
        return 1e6
    widths = rot.sum(axis=1).astype(np.float64)
    valid = widths > 0
    widths = widths[valid]
    if len(widths) < 5:
        return 1e6
    widths = widths / (widths.max() + 1e-8)
    x_old = np.linspace(0, 1, len(widths))
    x_new = np.linspace(0, 1, target_len)
    widths_rs = np.interp(x_new, x_old, widths)
    d2 = np.diff(widths_rs, n=2)
    return float(np.mean(d2 ** 2))

def compute_metrics_for_image(path: str) -> Dict[str, float]:
    img = read_image(path)
    mask, extra = extract_foreground_mask(img)
    if mask.sum() == 0:
        metrics = {m: np.nan for m in SUMMARY_METRICS}
        metrics.update({
            "area": 0.0,
            "perimeter": np.nan,
            "component_count": extra.get("component_count", 0),
            "raw_fg_area": extra.get("raw_fg_area", 0.0),
        })
        return metrics
    contour = find_main_contour(mask)
    if contour is None:
        metrics = {m: np.nan for m in SUMMARY_METRICS}
        metrics.update({
            "area": float(mask.sum()),
            "perimeter": np.nan,
            "component_count": extra.get("component_count", 0),
            "raw_fg_area": extra.get("raw_fg_area", 0.0),
        })
        return metrics
    area = mask_area(mask)
    perimeter = contour_perimeter(contour)
    metrics = {
        "area": float(area),
        "perimeter": float(perimeter),
        "solidity": contour_solidity(contour, area),
        "hole_ratio": float(extra["hole_ratio"]),
        "border_touch_ratio": border_touch_ratio(mask, perimeter),
        "norm_perimeter": float(perimeter / (math.sqrt(area) + 1e-8)),
        "boundary_roughness": boundary_roughness(contour, area=area),
        "fourier_hf_energy": fourier_hf_energy(contour),
        "symmetry": symmetry_score(mask),
        "width_smoothness": width_profile_smoothness(mask),
        "component_count": int(extra["component_count"]),
        "raw_fg_area": float(extra["raw_fg_area"]),
    }
    return metrics

def build_method_dataframe(records: List[ImageRecord]) -> pd.DataFrame:
    rows = []
    total = len(records)
    for idx, rec in enumerate(records, 1):
        try:
            metrics = compute_metrics_for_image(rec.abs_path)
            row = {
                "method": rec.method,
                "class_label": rec.class_label,
                "sample_key": rec.sample_key,
                "rel_path": rec.rel_path,
                "abs_path": rec.abs_path,
                **metrics,
                "status": "ok",
            }
        except Exception as e:
            row = {
                "method": rec.method,
                "class_label": rec.class_label,
                "sample_key": rec.sample_key,
                "rel_path": rec.rel_path,
                "abs_path": rec.abs_path,
                **{m: np.nan for m in ["area", "perimeter"] + SUMMARY_METRICS},
                "component_count": np.nan,
                "raw_fg_area": np.nan,
                "status": f"error: {type(e).__name__}: {e}",
            }
        rows.append(row)
        if idx % 50 == 0 or idx == total:
            print(f"[{rec.method}] 已处理 {idx}/{total}")
    df = pd.DataFrame(rows)
    return df

def minmax_quality_score(df: pd.DataFrame, metrics: List[str]) -> pd.Series:
    score_parts = []
    for m in metrics:
        vals = df[m].astype(float)
        finite = vals.replace([np.inf, -np.inf], np.nan)
        vmin = finite.min(skipna=True)
        vmax = finite.max(skipna=True)
        if pd.isna(vmin) or pd.isna(vmax) or abs(vmax - vmin) < 1e-12:
            norm = pd.Series(np.full(len(df), 0.5), index=df.index)
        else:
            if METRIC_DIRECTIONS[m] > 0:
                norm = (vals - vmin) / (vmax - vmin)
            else:
                norm = (vmax - vals) / (vmax - vmin)
        score_parts.append(norm.clip(0, 1))
    return pd.concat(score_parts, axis=1).mean(axis=1)

def pair_method_results(
    df_gs: pd.DataFrame,
    df_ts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gs_group = defaultdict(list)
    ts_group = defaultdict(list)
    for _, row in df_gs.iterrows():
        gs_group[row["sample_key"]].append(row)
    for _, row in df_ts.iterrows():
        ts_group[row["sample_key"]].append(row)
    all_keys = sorted(set(gs_group) | set(ts_group))
    paired_rows = []
    unmatched_rows = []
    for key in all_keys:
        left = sorted(gs_group.get(key, []), key=lambda x: x["rel_path"])
        right = sorted(ts_group.get(key, []), key=lambda x: x["rel_path"])
        n_pair = min(len(left), len(right))
        n_left_extra = len(left) - n_pair
        n_right_extra = len(right) - n_pair
        for i in range(n_pair):
            a = left[i]
            b = right[i]
            pair_row = {
                "sample_key": key,
                "class_label": a["class_label"] if a["class_label"] != "unknown" else b["class_label"],
                "gs_rel_path": a["rel_path"],
                "ts_rel_path": b["rel_path"],
                "gs_status": a["status"],
                "ts_status": b["status"],
            }
            for m in ["area", "perimeter"] + SUMMARY_METRICS + ["component_count", "raw_fg_area", "composite_quality"]:
                pair_row[f"gs_{m}"] = a.get(m, np.nan)
                pair_row[f"ts_{m}"] = b.get(m, np.nan)
                if m in SUMMARY_METRICS + ["composite_quality"]:
                    pair_row[f"diff_{m}_gs_minus_ts"] = a.get(m, np.nan) - b.get(m, np.nan)
            paired_rows.append(pair_row)
        for extra in left[n_pair:]:
            unmatched_rows.append({
                "sample_key": key,
                "method": "GroundedSAM",
                "rel_path": extra["rel_path"],
                "reason": "no matching TongueSAM file",
            })
        for extra in right[n_pair:]:
            unmatched_rows.append({
                "sample_key": key,
                "method": "TongueSAM",
                "rel_path": extra["rel_path"],
                "reason": "no matching GroundedSAM file",
            })
        if n_left_extra != 0 or n_right_extra != 0:
            unmatched_rows.append({
                "sample_key": key,
                "method": "pairing_note",
                "rel_path": "",
                "reason": f"duplicate or count mismatch: gs={len(left)}, ts={len(right)}",
            })
    return pd.DataFrame(paired_rows), pd.DataFrame(unmatched_rows)

def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("method"):
        for m in ["area", "perimeter"] + SUMMARY_METRICS + ["component_count", "raw_fg_area", "composite_quality"]:
            vals = pd.to_numeric(sub[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                row = {
                    "method": method,
                    "metric": m,
                    "n": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                }
            else:
                row = {
                    "method": method,
                    "metric": m,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                }
            rows.append(row)
    return pd.DataFrame(rows)

def paired_statistics(paired_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in SUMMARY_METRICS + ["composite_quality"]:
        gs = pd.to_numeric(paired_df[f"gs_{m}"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ts = pd.to_numeric(paired_df[f"ts_{m}"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = gs.notna() & ts.notna()
        x = gs[valid].to_numpy(dtype=float)
        y = ts[valid].to_numpy(dtype=float)
        if len(x) == 0:
            rows.append({
                "metric": m,
                "direction": "higher_better" if METRIC_DIRECTIONS.get(m, +1) > 0 else "lower_better",
                "n_pairs": 0,
                "gs_mean": np.nan,
                "ts_mean": np.nan,
                "gs_median": np.nan,
                "ts_median": np.nan,
                "mean_diff_gs_minus_ts": np.nan,
                "median_diff_gs_minus_ts": np.nan,
                "gs_better_ratio": np.nan,
                "ts_better_ratio": np.nan,
                "ties_ratio": np.nan,
                "wilcoxon_stat": np.nan,
                "wilcoxon_pvalue": np.nan,
                "better_method_by_mean": np.nan,
            })
            continue
        diffs = x - y
        direction = METRIC_DIRECTIONS.get(m, +1)
        if direction > 0:
            gs_better = np.sum(x > y)
            ts_better = np.sum(x < y)
            ties = np.sum(x == y)
            better_by_mean = "GroundedSAM" if np.nanmean(x) > np.nanmean(y) else ("TongueSAM" if np.nanmean(x) < np.nanmean(y) else "Tie")
        else:
            gs_better = np.sum(x < y)
            ts_better = np.sum(x > y)
            ties = np.sum(x == y)
            better_by_mean = "GroundedSAM" if np.nanmean(x) < np.nanmean(y) else ("TongueSAM" if np.nanmean(x) > np.nanmean(y) else "Tie")
        try:
            w = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
            stat = float(w.statistic)
            pval = float(w.pvalue)
        except Exception:
            stat = np.nan
            pval = np.nan
        rows.append({
            "metric": m,
            "direction": "higher_better" if direction > 0 else "lower_better",
            "n_pairs": int(len(x)),
            "gs_mean": float(np.mean(x)),
            "ts_mean": float(np.mean(y)),
            "gs_median": float(np.median(x)),
            "ts_median": float(np.median(y)),
            "mean_diff_gs_minus_ts": float(np.mean(diffs)),
            "median_diff_gs_minus_ts": float(np.median(diffs)),
            "gs_better_ratio": float(gs_better / len(x)),
            "ts_better_ratio": float(ts_better / len(x)),
            "ties_ratio": float(ties / len(x)),
            "wilcoxon_stat": stat,
            "wilcoxon_pvalue": pval,
            "better_method_by_mean": better_by_mean,
        })
    return pd.DataFrame(rows)

def maybe_make_plots(df_all: pd.DataFrame, output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib 不可用，跳过作图。")
        return
    plot_dir = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in SUMMARY_METRICS + ["composite_quality"]:
        try:
            sub = df_all[["method", metric]].copy()
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna()
            if len(sub) == 0:
                continue
            plt.figure(figsize=(6, 5))
            groups = []
            labels = []
            for method in ["GroundedSAM", "TongueSAM"]:
                vals = sub.loc[sub["method"] == method, metric].to_numpy(dtype=float)
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(method)
            if len(groups) == 0:
                plt.close()
                continue
            plt.boxplot(groups, labels=labels, showfliers=False)
            plt.title(metric)
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{metric}_boxplot.png", dpi=180)
            plt.close()
        except Exception as e:
            print(f"[WARN] 作图失败 {metric}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compare GroundedSAM vs TongueSAM tongue segmentation quality (unsupervised).")
    parser.add_argument(
        "--groundedsam_root",
        type=str,
        default="/path/to/groundedsam",
        help="GroundedSAM 根目录",
    )
    parser.add_argument(
        "--tonguesam_root",
        type=str,
        default="/path/to/tonguesam",
        help="TongueSAM 根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./seg_quality_results",
        help="输出目录",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="不生成箱线图",
    )
    args = parser.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("1) 收集图像 ...")
    gs_records = collect_images(args.groundedsam_root, method="GroundedSAM")
    ts_records = collect_images(args.tonguesam_root, method="TongueSAM")
    print(f"   GroundedSAM 图像数: {len(gs_records)}")
    print(f"   TongueSAM   图像数: {len(ts_records)}")
    print("2) 计算每张图的无监督指标 ...")
    df_gs = build_method_dataframe(gs_records)
    df_ts = build_method_dataframe(ts_records)
    df_all = pd.concat([df_gs, df_ts], axis=0, ignore_index=True)
    df_all["composite_quality"] = minmax_quality_score(df_all, SUMMARY_METRICS)
    df_gs = df_all[df_all["method"] == "GroundedSAM"].copy()
    df_ts = df_all[df_all["method"] == "TongueSAM"].copy()
    print("3) 自动配对两种方法的结果 ...")
    paired_df, unmatched_df = pair_method_results(df_gs, df_ts)
    print(f"   成功配对数: {len(paired_df)}")
    print(f"   未配对/配对备注数: {len(unmatched_df)}")
    print("4) 统计汇总 ...")
    summary_df = summarize_by_method(df_all)
    paired_stats_df = paired_statistics(paired_df) if len(paired_df) > 0 else pd.DataFrame()
    print("5) 保存结果 ...")
    df_gs.to_csv(outdir / "groundedsam_metrics.csv", index=False, encoding="utf-8-sig")
    df_ts.to_csv(outdir / "tonguesam_metrics.csv", index=False, encoding="utf-8-sig")
    df_all.to_csv(outdir / "all_metrics.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(outdir / "paired_metrics.csv", index=False, encoding="utf-8-sig")
    unmatched_df.to_csv(outdir / "unmatched_or_pairing_notes.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(outdir / "summary_by_method.csv", index=False, encoding="utf-8-sig")
    paired_stats_df.to_csv(outdir / "paired_statistics.csv", index=False, encoding="utf-8-sig")
    if not args.skip_plots:
        print("6) 生成箱线图 ...")
        maybe_make_plots(df_all, str(outdir))
    print("\n完成。关键输出文件：")
    print(f"- {outdir / 'groundedsam_metrics.csv'}")
    print(f"- {outdir / 'tonguesam_metrics.csv'}")
    print(f"- {outdir / 'paired_metrics.csv'}")
    print(f"- {outdir / 'paired_statistics.csv'}")
    print(f"- {outdir / 'summary_by_method.csv'}")
    print(f"- {outdir / 'unmatched_or_pairing_notes.csv'}")
    print("\n指标解释（方向）：")
    print("  solidity            越大越好：越饱满，凹陷越少")
    print("  hole_ratio          越小越好：内部孔洞越少")
    print("  border_touch_ratio  越小越好：越不容易触边/裁切")
    print("  norm_perimeter      越小越好：同面积下周长越短，边界越规整")
    print("  boundary_roughness  越小越好：边界越平滑")
    print("  fourier_hf_energy   越小越好：轮廓高频越少，锯齿/抖动越少")
    print("  symmetry            越大越好：左右更对称")
    print("  width_smoothness    越小越好：纵向宽度曲线更平滑自然")
    print("  composite_quality   越大越好：基于以上指标的相对综合分")

if __name__ == "__main__":
    main()
