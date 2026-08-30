"""Continuous local improvement of the detection model from this
deployment's own footage -- the self-hosted equivalent of a managed
"submit your hard examples, get a better model back" service, with the
training step kept outside this container.

Why the training step lives elsewhere
-------------------------------------
This image ships onnxruntime for inference and deliberately not torch or
ultralytics: a recording appliance should not also be a training rig
competing for the same GPU and disk. So the split is:

  in here   -- harvest hard examples, auto-label them, export a dataset,
               EVALUATE a candidate model, promote or reject it
  elsewhere -- the actual fine-tune, in a throwaway container that has
               torch/ultralytics and reads the exported dataset

Evaluation stays in here on purpose. It only needs onnxruntime, and it is
the part that must not be skipped.

The validation gate is the whole point
--------------------------------------
Auto-labels harvested from your own model's behaviour are correlated with
that model's existing mistakes: anything it reliably fails to see never
enters the dataset, and anything it hallucinates confidently can enter as
a positive. Fine-tuning on that without a check will make the model more
confident about its current blind spots while every training-loss curve
looks healthy. So a candidate model is never promoted because training
"succeeded" -- only because it measurably beat the model currently in use
on a holdout set it has never been trained on.

The holdout is split by a hash of the event id, not randomly, so the same
event lands on the same side of the split on every run. A fresh random
split each time would quietly leak holdout frames into training as the
dataset grows, which is the subtle version of the same failure.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from typing import Any

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.const import CLIPS_DIR, MODEL_CACHE_DIR
from frigate.models import Event
from frigate.util.file import get_event_snapshot_bytes

logger = logging.getLogger(__name__)

AUTOTRAIN_DIR = os.path.join(CLIPS_DIR, "autotrain")
DATASET_DIR = os.path.join(AUTOTRAIN_DIR, "dataset")
CANDIDATE_DIR = os.path.join(AUTOTRAIN_DIR, "candidates")
MODEL_ARCHIVE_DIR = os.path.join(MODEL_CACHE_DIR, "autotrain_archive")

# Fraction of harvested events held back for evaluation and never exported
# as training data.
HOLDOUT_FRACTION = 0.2


def _is_holdout(event_id: str) -> bool:
    """Deterministic train/holdout assignment.

    Hashing the event id means an event keeps its side of the split forever,
    however many times the dataset is rebuilt. A random split would reassign
    events on each run, so frames evaluated as holdout one week become
    training data the next -- and the resulting score is then measuring how
    well the model memorised its own training set."""
    digest = hashlib.sha256(event_id.encode()).hexdigest()
    return (int(digest[:8], 16) % 100) < (HOLDOUT_FRACTION * 100)


def _label_index(config: FrigateConfig, label: str) -> int | None:
    """Map a Frigate label to its index in the detector's label map, so
    exported annotations match the classes the model actually predicts."""
    try:
        labelmap = config.model.merged_labelmap
    except Exception:
        return None

    for idx, name in labelmap.items():
        if name == label:
            return int(idx)
    return None


def harvest_training_examples(
    config: FrigateConfig, limit: int = 500, min_score: float = 0.4
) -> list[dict[str, Any]]:
    """Mine recent events for frames the current model found hard.

    The useful signal is disagreement between the detector and the tracker.
    The tracker fuses evidence over time, so when it holds a confident track
    while the per-frame score dips, that dip is very likely a detector miss
    on a real object -- a labelled hard example, for free, with no human in
    the loop.

    Events whose top score was never convincing are skipped entirely rather
    than guessed at: an event that might be a false positive is exactly the
    thing you must not auto-label, because feeding the model its own
    hallucinations is how auto-training degrades a detector."""
    events = (
        Event.select()
        .where(
            Event.label.is_null(False),
            Event.end_time.is_null(False),
            Event.has_snapshot == True,  # noqa: E712
        )
        .order_by(Event.start_time.desc())
        .limit(limit)
    )

    harvested: list[dict[str, Any]] = []

    for event in events:
        data = event.data or {}
        top_score = data.get("top_score") or data.get("score") or 0.0
        score = data.get("score") or 0.0
        box = data.get("box")

        if not box or top_score < min_score:
            continue

        label_idx = _label_index(config, event.label)
        if label_idx is None:
            continue

        # The tracker was confident overall (top_score) but this frame's own
        # score lagged -- the definition of a hard example.
        is_hard = score < (top_score - 0.15)

        harvested.append(
            {
                "event_id": event.id,
                "camera": event.camera,
                "label": event.label,
                "label_idx": label_idx,
                "box": box,
                "score": score,
                "top_score": top_score,
                "hard": is_hard,
                "holdout": _is_holdout(event.id),
            }
        )

    logger.info(
        f"Autotrain: harvested {len(harvested)} labelled examples "
        f"({sum(1 for h in harvested if h['hard'])} hard, "
        f"{sum(1 for h in harvested if h['holdout'])} holdout)"
    )
    return harvested


def export_yolo_dataset(
    config: FrigateConfig, examples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write a YOLO-format dataset for an external trainer to consume.

    Boxes in Frigate's event data are already normalised (cx, cy, w, h in
    0-1 terms after conversion), which is the format YOLO wants, so the
    export is mostly file plumbing rather than geometry."""
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped": 0}

    for ex in examples:
        event = Event.get_or_none(Event.id == ex["event_id"])
        if event is None:
            counts["skipped"] += 1
            continue

        try:
            jpg, _ = get_event_snapshot_bytes(event, ext="jpg", quality=92)
        except Exception:
            jpg = None

        if not jpg:
            counts["skipped"] += 1
            continue

        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            counts["skipped"] += 1
            continue

        split = "val" if ex["holdout"] else "train"
        stem = ex["event_id"].replace(".", "_")

        cv2.imwrite(os.path.join(DATASET_DIR, split, "images", f"{stem}.jpg"), img)

        x1, y1, x2, y2 = ex["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = abs(x2 - x1), abs(y2 - y1)

        with open(
            os.path.join(DATASET_DIR, split, "labels", f"{stem}.txt"), "w"
        ) as f:
            f.write(f"{ex['label_idx']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        counts[split] += 1

    names = {}
    try:
        names = {int(k): v for k, v in config.model.merged_labelmap.items()}
    except Exception:
        pass

    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write(f"nc: {len(names) or 80}\n")
        f.write("names:\n")
        for idx in sorted(names):
            f.write(f"  {idx}: {names[idx]}\n")

    logger.info(f"Autotrain: exported dataset {counts} to {DATASET_DIR}")
    return {"dataset_dir": DATASET_DIR, **counts}


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_on_holdout(
    config: FrigateConfig,
    examples: list[dict[str, Any]],
    detect_fn,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Score a detector against the frozen holdout split.

    detect_fn(image) -> list of (label_idx, score, [x1,y1,x2,y2]) in
    normalised coordinates, so the same routine can score the running model
    and a candidate without knowing how either is loaded.

    Reports recall and mean IoU rather than a full COCO mAP: the holdout
    labels here are single-object auto-labels, so a headline mAP would imply
    a completeness the annotations do not have. Recall against known-present
    objects is the honest question -- does the candidate still find what we
    know is there?"""
    holdout = [e for e in examples if e["holdout"]]
    if not holdout:
        return {"evaluated": 0, "recall": 0.0, "mean_iou": 0.0}

    hits = 0
    ious: list[float] = []

    for ex in holdout:
        event = Event.get_or_none(Event.id == ex["event_id"])
        if event is None:
            continue
        try:
            jpg, _ = get_event_snapshot_bytes(event, ext="jpg", quality=92)
        except Exception:
            continue
        if not jpg:
            continue
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        try:
            detections = detect_fn(img)
        except Exception as e:
            logger.debug(f"Autotrain: detect failed on {ex['event_id']}: {e}")
            continue

        best = 0.0
        for label_idx, _score, box in detections:
            if label_idx != ex["label_idx"]:
                continue
            best = max(best, _iou(box, ex["box"]))

        if best >= iou_threshold:
            hits += 1
        ious.append(best)

    evaluated = len(ious)
    return {
        "evaluated": evaluated,
        "recall": (hits / evaluated) if evaluated else 0.0,
        "mean_iou": (float(np.mean(ious)) if ious else 0.0),
    }


def promote_candidate(
    candidate_path: str,
    active_path: str,
    baseline: dict[str, float],
    candidate: dict[str, float],
    min_improvement: float = 0.01,
) -> dict[str, Any]:
    """Replace the active model with a candidate ONLY if it measurably beat
    the current one on the holdout.

    Requires a margin rather than any improvement at all: on a holdout of a
    few hundred frames, a recall difference of a fraction of a percent is
    noise, and promoting on noise means the model random-walks over time
    instead of improving. The outgoing model is archived, so a promotion
    that turns out badly in the field is one file move to undo."""
    delta = candidate["recall"] - baseline["recall"]

    if candidate["evaluated"] < 20:
        return {
            "promoted": False,
            "reason": (
                f"holdout too small to judge ({candidate['evaluated']} frames); "
                "refusing to promote on a sample this size"
            ),
            "delta_recall": delta,
        }

    if delta < min_improvement:
        return {
            "promoted": False,
            "reason": (
                f"candidate recall {candidate['recall']:.3f} vs current "
                f"{baseline['recall']:.3f} (delta {delta:+.3f}) does not clear the "
                f"{min_improvement:+.3f} margin"
            ),
            "delta_recall": delta,
        }

    os.makedirs(MODEL_ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(active_path):
        archived = os.path.join(
            MODEL_ARCHIVE_DIR, f"{int(time.time())}-{os.path.basename(active_path)}"
        )
        shutil.copy2(active_path, archived)
    else:
        archived = None

    shutil.copy2(candidate_path, active_path)

    logger.info(
        f"Autotrain: promoted {candidate_path} -> {active_path} "
        f"(recall {baseline['recall']:.3f} -> {candidate['recall']:.3f})"
    )
    return {
        "promoted": True,
        "delta_recall": delta,
        "archived_previous": archived,
        "reason": "candidate beat the active model on the holdout",
    }


def run_harvest_and_export(config: FrigateConfig, limit: int = 500) -> dict[str, Any]:
    """One full harvest+export cycle, ready for an external trainer."""
    os.makedirs(AUTOTRAIN_DIR, exist_ok=True)
    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    examples = harvest_training_examples(config, limit=limit)
    if not examples:
        return {"harvested": 0, "message": "no suitable events found"}

    summary = export_yolo_dataset(config, examples)

    manifest = {
        "generated_at": time.time(),
        "harvested": len(examples),
        "hard_examples": sum(1 for e in examples if e["hard"]),
        "holdout": sum(1 for e in examples if e["holdout"]),
        **summary,
    }
    with open(os.path.join(AUTOTRAIN_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
