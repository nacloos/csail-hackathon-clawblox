"""
Evaluate FastSAM segmentation against Grounded-SAM2 ground truth.
Measures mask quality (IoU, Dice) and downstream pose estimation error.

Usage:
    python evaluate_segmentation.py
        [--image-dir PATH]
        [--depth-dir PATH]
        [--brick-model PATH]
        [--output-dir PATH]

To-do's:
    - This script doesn't seem to be computing correct metrics for pose alignment.
    - Perhaps its comparing across all matches, not the best one?
    - Or IoU threshold is too low which defines what a match is?
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bricklaying.segmentation.base import BrickDetection
from bricklaying.segmentation.grounded_sam2 import GroundedSAM2Segmentor
from bricklaying.segmentation.fastsam import FastSAMSegmentor
from bricklaying.perception.realsense import D435_DEFAULT_INTRINSICS
from bricklaying.perception.pose_estimation import BrickPoseEstimator, PoseEstimatorConfig
from bricklaying.perception.icp import ICPConfig

DEVICE = "cuda"
DEFAULT_IMAGE_DIR = "src/bricklaying/assets/vision_validation/26-02-18"
DEFAULT_DEPTH_DIR = "src/bricklaying/assets/scene_measurements/26-02-18"
DEFAULT_BRICK_MODEL = "src/bricklaying/assets/flat_brick_large.npy"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ===== Mask metrics =====

@dataclass
class MaskMetrics:
    iou: float
    dice: float
    precision: float
    recall: float


def compute_mask_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> MaskMetrics:
    gt = gt_mask.astype(bool)
    pred = pred_mask.astype(bool)
    intersection = (gt & pred).sum()
    union = (gt | pred).sum()
    gt_sum, pred_sum = gt.sum(), pred.sum()
    return MaskMetrics(
        iou=float(intersection / union) if union > 0 else 0.0,
        dice=float(2 * intersection / (gt_sum + pred_sum)) if (gt_sum + pred_sum) > 0 else 0.0,
        precision=float(intersection / pred_sum) if pred_sum > 0 else 0.0,
        recall=float(intersection / gt_sum) if gt_sum > 0 else 0.0,
    )


def resize_mask_to(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def best_match_metrics(
    gt_detections: list[BrickDetection],
    pred_detections: list[BrickDetection],
    iou_threshold: float,
) -> tuple[list[MaskMetrics], int]:
    """
    For each GT detection, find the best-IoU FastSAM match.
    Returns per-GT metrics and a count of matched GT detections.
    """
    metrics, matched = [], 0
    for gt_det in gt_detections:
        best, best_iou = MaskMetrics(0.0, 0.0, 0.0, 0.0), 0.0
        for pred_det in pred_detections:
            pred_mask = pred_det.mask
            if gt_det.mask.shape != pred_mask.shape:
                pred_mask = resize_mask_to(pred_mask, *gt_det.mask.shape[:2])
            m = compute_mask_metrics(gt_det.mask, pred_mask)
            if m.iou > best_iou:
                best_iou, best = m.iou, m
        if best_iou >= iou_threshold:
            matched += 1
            metrics.append(best)    
    return metrics, matched


# ===== Pose metrics =====

@dataclass
class PoseError:
    translation_m: float
    rotation_deg: float
    gt_converged: bool
    pred_converged: bool


def _rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compare_poses(
    rgb: np.ndarray,
    depth: np.ndarray,
    gt_detections: list[BrickDetection],
    pred_detections: list[BrickDetection],
    gt_estimator: BrickPoseEstimator,
    pred_estimator: BrickPoseEstimator,
    iou_threshold: float,
) -> list[PoseError]:
    """
    For each GT/FastSAM detection pair (matched by IoU), estimate pose independently
    using BrickPoseEstimator and return the pose differences.
    """
    errors = []
    for gt_det in gt_detections:
        best_pred, best_iou = None, -1.0
        for pred_det in pred_detections:
            pred_mask = pred_det.mask
            if gt_det.mask.shape != pred_mask.shape:
                pred_mask = resize_mask_to(pred_mask, *gt_det.mask.shape[:2])
            iou = compute_mask_metrics(gt_det.mask, pred_mask).iou
            if iou > best_iou:
                best_iou, best_pred = iou, pred_det

        if best_iou < iou_threshold or best_pred is None:
            continue

        gt_poses = gt_estimator.estimate_from_frame(rgb, depth, D435_DEFAULT_INTRINSICS, detections=[gt_det])
        pred_poses = pred_estimator.estimate_from_frame(rgb, depth, D435_DEFAULT_INTRINSICS, detections=[best_pred])

        if not gt_poses or not pred_poses:
            continue

        gt_pose, pred_pose = gt_poses[0], pred_poses[0]
        errors.append(PoseError(
            translation_m=float(np.linalg.norm(gt_pose.position - pred_pose.position)),
            rotation_deg=_rotation_error_deg(gt_pose.rotation, pred_pose.rotation),
            gt_converged=gt_pose.icp_converged,
            pred_converged=pred_pose.icp_converged,
        ))

    return errors


# ===== Visualization =====

def visualize_comparison(
    rgb: np.ndarray,
    gt_detections: list[BrickDetection],
    pred_detections: list[BrickDetection],
    save_path: Path,
) -> None:
    h, w = rgb.shape[:2]

    def overlay(base, detections, color):
        out = base.copy()
        for det in detections:
            mask = det.mask if det.mask.shape == (h, w) else resize_mask_to(det.mask, h, w)
            colored = np.zeros_like(base)
            colored[mask > 0] = color
            out = cv2.addWeighted(out, 1.0, colored, 0.5, 0)
        return out

    gt_vis = overlay(rgb, gt_detections, [0, 255, 0])
    pred_vis = overlay(rgb, pred_detections, [255, 100, 0])
    cv2.putText(gt_vis, "Ground Truth (Grounded-SAM2)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(pred_vis, "Prediction (FastSAM)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(save_path), cv2.cvtColor(np.concatenate([gt_vis, pred_vis], axis=1), cv2.COLOR_RGB2BGR))


# ===== Summary printing =====

def _print_mask_summary(results: list[dict], iou_threshold: float) -> None:
    total_gt = sum(r["n_gt"] for r in results)
    total_pred = sum(r["n_pred"] for r in results)
    total_matched = sum(r["matched_gt"] for r in results)
    mean_iou = float(np.mean([r["mean_iou"] for r in results]))
    mean_dice = float(np.mean([r["mean_dice"] for r in results]))

    print("\n" + "=" * 60)
    print("MASK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Images evaluated  : {len(results)}")
    print(f"Total GT masks    : {total_gt}")
    print(f"Total FastSAM det : {total_pred}")
    print(f"Matched (IoU≥{iou_threshold:.2f}): {total_matched} / {total_gt}")
    print(f"Overall mean IoU  : {mean_iou:.4f}")
    print(f"Overall mean Dice : {mean_dice:.4f}")
    print("=" * 60)
    print(f"\n{'Image':<40} {'GT':>4} {'Pred':>5} {'IoU':>7} {'Dice':>7} {'Match':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r['image']:<40} {r['n_gt']:>4} {r['n_pred']:>5} "
              f"{r['mean_iou']:>7.3f} {r['mean_dice']:>7.3f} "
              f"{r['matched_gt']}/{r['n_gt']}")


def _print_pose_summary(results: list[dict]) -> None:
    all_trans = [e["translation_m"] for r in results for e in r["pose_errors"]]
    all_rot = [e["rotation_deg"] for r in results for e in r["pose_errors"]]
    total_pairs = len(all_trans)

    print("\n" + "=" * 60)
    print("POSE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total matched pairs : {total_pairs}")
    if total_pairs:
        print(f"Translation error   : {np.mean(all_trans)*1000:.2f}mm mean "
              f"± {np.std(all_trans)*1000:.2f}mm std "
              f"(max {np.max(all_trans)*1000:.2f}mm)")
        print(f"Rotation error      : {np.mean(all_rot):.2f}° mean "
              f"± {np.std(all_rot):.2f}° std "
              f"(max {np.max(all_rot):.2f}°)")
    print("=" * 60)
    print(f"\n{'Image':<40} {'Pairs':>6} {'Trans(mm)':>10} {'Rot(°)':>8}")
    print("-" * 68)
    for r in results:
        errors = r["pose_errors"]
        if not errors:
            continue
        mean_trans = np.mean([e["translation_m"] for e in errors]) * 1000
        mean_rot = np.mean([e["rotation_deg"] for e in errors])
        print(f"{r['image']:<40} {len(errors):>6} {mean_trans:>10.2f} {mean_rot:>8.2f}")


# ===== Main =====

def evaluate(image_dir: str, depth_dir: str, brick_model_path: str, output_dir: str, iou_threshold: float = 0.1) -> None:
    image_dir, depth_dir, output_dir = Path(image_dir), Path(depth_dir), Path(output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(exist_ok=True)

    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    # Models
    gt_segmentor = GroundedSAM2Segmentor(device=DEVICE)
    pred_segmentor = FastSAMSegmentor(device=DEVICE)

    brick_model = np.load(brick_model_path)
    pose_config = PoseEstimatorConfig(min_points=200)
    gt_estimator = BrickPoseEstimator(brick_model, gt_segmentor, camera=None, config=pose_config)
    pred_estimator = BrickPoseEstimator(brick_model, pred_segmentor, camera=None, config=pose_config)

    print(f"Brick model: {len(brick_model)} points")
    print(f"Processing {len(image_paths)} images...\n")

    all_mask_metrics: list[MaskMetrics] = []
    all_pose_errors: list[PoseError] = []
    results_per_image = []

    for img_path in image_paths:
        img_name = "_".join(img_path.stem.split('_')[1:-1])
        depth_path = depth_dir / f"depth_{img_name}.png"

        image_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH) if depth_path.exists() else None

        gt_detections = gt_segmentor.segment(image_rgb)
        pred_detections = pred_segmentor.segment(image_rgb)

        # Mask evaluation
        mask_metrics, matched = best_match_metrics(gt_detections, pred_detections, iou_threshold)
        all_mask_metrics.extend(mask_metrics)
        mean_iou = float(np.mean([m.iou for m in mask_metrics])) if mask_metrics else 0.0
        mean_dice = float(np.mean([m.dice for m in mask_metrics])) if mask_metrics else 0.0

        print(f"{img_path.name}")
        print(f"  GT: {len(gt_detections)}  FastSAM: {len(pred_detections)}  "
              f"Matched: {matched}/{len(gt_detections)}  IoU: {mean_iou:.3f}  Dice: {mean_dice:.3f}")

        # Pose evaluation
        pose_errors = []
        if depth is not None:
            pose_errors = compare_poses(
                image_rgb, depth, gt_detections, pred_detections,
                gt_estimator, pred_estimator, iou_threshold,
            )
            all_pose_errors.extend(pose_errors)
            if pose_errors:
                mean_trans = np.mean([e.translation_m for e in pose_errors]) * 1000
                mean_rot = np.mean([e.rotation_deg for e in pose_errors])
                print(f"  Pose pairs: {len(pose_errors)}  Trans: {mean_trans:.1f}mm  Rot: {mean_rot:.2f}°")
        else:
            print(f"  WARNING: No depth image at {depth_path}")

        visualize_comparison(image_rgb, gt_detections, pred_detections, vis_dir / f"{img_path.stem}.jpg")

        results_per_image.append({
            "image": img_path.name,
            "n_gt": len(gt_detections),
            "n_pred": len(pred_detections),
            "matched_gt": matched,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "pose_errors": [
                {"translation_m": e.translation_m, "rotation_deg": e.rotation_deg,
                 "gt_converged": e.gt_converged, "pred_converged": e.pred_converged}
                for e in pose_errors
            ],
        })

    _print_mask_summary(results_per_image, iou_threshold)
    if all_pose_errors:
        _print_pose_summary(results_per_image)

    # Save JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"iou_threshold": iou_threshold, "per_image": results_per_image}, f, indent=2)
    print(f"\nSaved: {results_path}")
    print(f"Saved: {vis_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FastSAM vs Grounded-SAM2.")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--depth-dir", default=DEFAULT_DEPTH_DIR)
    parser.add_argument("--brick-model", default=DEFAULT_BRICK_MODEL)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    evaluate(args.image_dir, args.depth_dir, args.brick_model, args.output_dir, args.iou_threshold)


"""
Outputs on 02/19/26

============================================================
MASK EVALUATION SUMMARY
============================================================
Images evaluated  : 50
Total GT masks    : 90
Total FastSAM det : 566
Matched (IoU≥0.10): 90 / 90
Overall mean IoU  : 0.9527
Overall mean Dice : 0.9757
============================================================

============================================================
POSE EVALUATION SUMMARY
============================================================
Total matched pairs : 90
Translation error   : 1.22mm mean ± 0.47mm std (max 2.72mm)
Rotation error      : 1.00° mean ± 0.49° std (max 2.12°)
============================================================
"""