import os
import argparse
import math
from typing import List, Dict, Optional

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from detect_from_images import test_on_images


def infer_ground_truth_label(filepath: str) -> Optional[int]:
    """
    Infer ground-truth label from parent folder name.
    Returns 0 for real, 1 for fake, or None if not inferrable.
    """
    parent = os.path.basename(os.path.dirname(filepath)).lower()
    if parent == "real":
        return 0
    if parent == "fake":
        return 1
    # Try to infer from any part of the path as a fallback
    path_lower = filepath.lower()
    if "/real/" in path_lower or "\\real\\" in path_lower:
        return 0
    if "/fake/" in path_lower or "\\fake\\" in path_lower:
        return 1
    return None


def binary_cross_entropy(prob_fake: float, target: int, eps: float = 1e-12) -> float:
    """
    Compute binary cross entropy: -[y*log(p) + (1-y)*log(1-p)]
    where y in {0,1} and p is probability for class 1 (fake).
    """
    p = min(max(prob_fake, eps), 1.0 - eps)
    if target == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def evaluate(images_path: str, model_path: str, cuda: bool, skip_no_face: bool = True,
             output_csv: Optional[str] = None, root_folder: bool = False, vis_dir: Optional[str] = None) -> Dict[str, float]:
    """
    Run inference using test_on_images and compute accuracy, avg BCE loss, and AUC (if available).
    images_path: directory that contains subfolders named with class labels (e.g., real/ and fake/)
    """
    results = test_on_images(images_path, model_path, output_csv=output_csv, vis_dir=vis_dir, cuda=cuda, root_folder=root_folder)

    y_true: List[int] = []
    y_score: List[float] = []  # probability for class 1 (fake)
    y_pred: List[int] = []

    for r in results:
        if r is None:
            continue
        if skip_no_face and (not r.get('has_face')):
            continue

        gt = infer_ground_truth_label(r['path'])
        if gt is None:
            continue

        probs = r.get('probs')
        if probs is None or len(probs) != 2:
            continue
        p_fake = float(probs[1])
        pred = 1 if p_fake >= 0.5 else 0

        y_true.append(int(gt))
        y_score.append(p_fake)
        y_pred.append(pred)

    metrics = {}
    if len(y_true) == 0:
        print('No samples with inferable ground-truth were found. Ensure directory names are real/ and fake/.')
        return {"num_samples": 0}

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_score_np = np.array(y_score)

    # Accuracy
    acc = float((y_true_np == y_pred_np).mean())
    metrics['accuracy'] = acc

    # Average BCE Loss
    losses = [binary_cross_entropy(float(s), int(t)) for s, t in zip(y_score_np, y_true_np)]
    metrics['avg_bce_loss'] = float(np.mean(losses))

    # AUC
    if HAS_SKLEARN:
        try:
            auc = roc_auc_score(y_true_np, y_score_np)
            metrics['auc'] = float(auc)
        except Exception:
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    metrics['num_samples'] = int(len(y_true))
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--images_path', '-i', type=str, required=True,
                   help='Directory containing images under subfolders named real/ and fake/')
    p.add_argument('--model_path', '-m', type=str, required=True,
                   help='Path to Xception .p model file')
    p.add_argument('--cuda', action='store_true', help='Enable CUDA if available')
    p.add_argument('--skip_no_face', action='store_true', help='Skip images where no face is detected')
    p.add_argument('--output_csv', type=str, default=None, help='Optional CSV path to also save raw predictions')
    p.add_argument('--vis_dir', type=str, default=None, help='Optional directory for annotated images')
    p.add_argument('--root_folder', action='store_true', help='Search recursively in all subdirectories for images')
    args = p.parse_args()

    metrics = evaluate(args.images_path, args.model_path, args.cuda,
                       skip_no_face=args.skip_no_face,
                       output_csv=args.output_csv,
                       vis_dir=args.vis_dir,
                       root_folder=args.root_folder)
    print({k: v for k, v in metrics.items()})
