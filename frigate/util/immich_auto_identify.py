"""Automatically identify Frigate's "unknown" face queue against a
self-hosted Immich instance's named people.

Immich exposes no reverse face-search API (confirmed: /api/search/faces,
/api/faces/search, /api/people/search all 404, and Person objects carry no
embedding data over the API) -- so the matching has to happen on this side,
using Frigate's own face embedder against reference photos pulled from
Immich.

Runs as a separate, on-demand pass over the unknown-face queue; never
touches the real-time detection/recording path, so an Immich outage here
can never affect live recording -- same fail-open principle as the rest of
this integration.
"""

import logging
import os
import re
import threading
from multiprocessing.synchronize import Event as MpEvent

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.config.classification import ImmichConfig
from frigate.const import FACE_DIR
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    best_match_confidence,
    similarity_to_confidence,
)
from frigate.embeddings import EmbeddingsContext
from frigate.util.immich import _get, fetch_candidate_faces

logger = logging.getLogger(__name__)

# Matches Frigate's own unknown-face queue filename convention:
# {event_id_part1}-{event_id_part2}-{timestamp}-unknown-{score}.webp
UNKNOWN_FILENAME_RE = re.compile(
    r"^[^-]+-[^-]+-[^-]+-unknown-[\d.]+\.webp$"
)


def _build_person_embedding_cache(
    recognizer: ArcFaceRecognizer, immich_config: ImmichConfig
) -> dict[str, list[np.ndarray]]:
    """Fetch every named Immich person's quality-filtered reference photos
    and embed them with Frigate's own embedder. Rebuilt fresh on every call
    -- this is an on-demand batch pass, not a long-lived cache, so it always
    reflects the current state of the Immich library."""
    resp = _get(immich_config, "/api/people", params={"withHidden": "false"})
    people = resp.json()["people"]

    cache: dict[str, list[np.ndarray]] = {}
    for person in people:
        candidates = fetch_candidate_faces(immich_config, person["id"])
        embeddings = []
        for candidate in candidates:
            img = cv2.imdecode(
                np.frombuffer(candidate["image_bytes"], dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is None:
                continue
            aligned = recognizer.align_face(img, img.shape[1], img.shape[0])
            emb = recognizer.face_embedder([aligned])[0].squeeze()
            embeddings.append(emb)

        if embeddings:
            cache[person["name"]] = embeddings

    logger.info(
        f"Immich auto-identify: built reference embeddings for {len(cache)} people"
    )
    return cache


def run_auto_identify_pass(
    config: FrigateConfig, context: EmbeddingsContext
) -> dict:
    """Scan Frigate's unknown-face queue, match each crop against a fresh
    Immich people embedding cache, and auto-register any match at or above
    the configured confidence threshold. Returns a summary dict. Raises if
    Immich is not configured -- caller is responsible for the HTTP response.
    """
    immich_config = config.face_recognition.immich

    if not immich_config.enabled or not immich_config.host or not immich_config.api_key:
        raise ValueError("Immich integration is not configured")

    if not immich_config.auto_identify_enabled:
        raise ValueError("Immich auto-identify is not enabled")

    # Only need the embedder + landmark aligner here, not the full
    # class_refs-based classifier -- both are ready synchronously from the
    # constructor, no need to wait on the async build() queue.
    recognizer = ArcFaceRecognizer(config)

    person_cache = _build_person_embedding_cache(recognizer, immich_config)

    if not person_cache:
        return {"checked": 0, "identified": 0, "people_available": 0}

    max_profile_ratio = config.face_recognition.max_profile_ratio

    train_dir = os.path.join(FACE_DIR, "train")
    checked = 0
    identified = 0
    skipped_non_frontal = 0
    matches: list[dict] = []

    for filename in os.listdir(train_dir):
        if not UNKNOWN_FILENAME_RE.match(filename):
            continue

        checked += 1
        path = os.path.join(train_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        # Reject side-profile crops before matching, not just at a
        # confidence-score level: a profile face can still score a high
        # raw similarity against the wrong person's frontal reference
        # photos, because the embedding itself is unreliable from that
        # angle in the first place, not because the match was close.
        yaw_ratio = recognizer.estimate_face_yaw_ratio(img)
        if yaw_ratio is not None and yaw_ratio > max_profile_ratio:
            skipped_non_frontal += 1
            continue

        aligned = recognizer.align_face(img, img.shape[1], img.shape[0])
        embedding = recognizer.face_embedder([aligned])[0].squeeze()

        best_name, best_score = None, 0.0
        for name, refs in person_cache.items():
            cosine_similarity = best_match_confidence(embedding, refs)
            confidence = similarity_to_confidence(cosine_similarity)
            if confidence > best_score:
                best_score = confidence
                best_name = name

        if best_name and best_score >= immich_config.auto_identify_threshold:
            with open(path, "rb") as f:
                image_bytes = f.read()

            result = context.register_face(best_name, image_bytes)
            if isinstance(result, dict) and result.get("success"):
                identified += 1
                os.remove(path)
                matches.append(
                    {"file": filename, "name": best_name, "confidence": round(best_score, 3)}
                )
                logger.info(
                    f"Immich auto-identify: matched {filename} to '{best_name}' "
                    f"at {best_score:.3f} confidence"
                )

    return {
        "checked": checked,
        "identified": identified,
        "skipped_non_frontal": skipped_non_frontal,
        "people_available": len(person_cache),
        "matches": matches,
    }


class ImmichAutoIdentifyMaintainer(threading.Thread):
    """Runs run_auto_identify_pass on a timer in the background, so new
    unknown faces get matched against Immich without waiting for someone to
    click the manual button on the Face Library page. Mirrors the
    threading.Thread + stop_event.wait(interval) pattern used by
    frigate.storage.StorageMaintainer elsewhere in this codebase.

    Fails open: any error during a scheduled pass (Immich unreachable,
    misconfigured, a transient decode failure) is logged and the next
    interval is tried again -- never crashes the app, and never touches the
    live detection/recording path either way."""

    def __init__(
        self, config: FrigateConfig, context: EmbeddingsContext, stop_event: MpEvent
    ) -> None:
        super().__init__(name="immich_auto_identify", daemon=True)
        self.config = config
        self.context = context
        self.stop_event = stop_event

    def run(self) -> None:
        immich_config = self.config.face_recognition.immich
        interval = immich_config.auto_identify_interval_seconds

        if interval <= 0:
            logger.info(
                "Immich auto-identify: background pass disabled "
                "(auto_identify_interval_seconds <= 0)"
            )
            return

        # give the rest of startup (embeddings client, etc.) time to settle
        # before the first pass
        if self.stop_event.wait(60):
            return

        while True:
            immich_config = self.config.face_recognition.immich
            if not immich_config.enabled or not immich_config.auto_identify_enabled:
                logger.debug(
                    "Immich auto-identify: skipping scheduled pass, not enabled"
                )
            else:
                try:
                    summary = run_auto_identify_pass(self.config, self.context)
                    if summary["identified"] > 0:
                        logger.info(f"Immich auto-identify (scheduled): {summary}")
                    else:
                        logger.debug(f"Immich auto-identify (scheduled): {summary}")
                except Exception as e:
                    logger.warning(f"Immich auto-identify (scheduled) pass failed: {e}")

            if self.stop_event.wait(interval):
                break

        logger.info("Exiting Immich auto-identify maintainer...")
