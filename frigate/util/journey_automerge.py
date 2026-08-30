"""Background maintainer that pre-builds one merged video per multi-camera
walk, so a cross-camera alert is a single clip you can just press play on
instead of several clips you have to stitch together in your head.

Only builds for chains that actually span more than one camera -- a
same-camera session is already one continuous recording, so merging it
would produce a copy of something you can already watch.

Deliberately conservative about disk: merged clips are a derived
convenience, not source material, so they live under their own directory
with a hard cap on how many are kept and are safe to delete at any time.
"""

import logging
import os
import threading
import time
from multiprocessing.synchronize import Event as MpEvent

from frigate.config import FrigateConfig
from frigate.const import CLIPS_DIR
from frigate.embeddings import EmbeddingsContext
from frigate.models import Event

logger = logging.getLogger(__name__)

JOURNEY_CLIP_DIR = os.path.join(CLIPS_DIR, "journeys")

# A merged clip is regenerable from the source recordings at any time, so
# this cap exists purely to stop a derived cache from competing with
# recordings for disk.
MAX_CACHED_CLIPS = 50


def merged_clip_path(event_id: str) -> str:
    return os.path.join(JOURNEY_CLIP_DIR, f"{event_id}.mp4")


def _prune_cache() -> None:
    """Keep only the newest MAX_CACHED_CLIPS merged clips."""
    try:
        entries = [
            os.path.join(JOURNEY_CLIP_DIR, f)
            for f in os.listdir(JOURNEY_CLIP_DIR)
            if f.endswith(".mp4")
        ]
    except FileNotFoundError:
        return

    if len(entries) <= MAX_CACHED_CLIPS:
        return

    entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for stale in entries[MAX_CACHED_CLIPS:]:
        try:
            os.remove(stale)
            logger.debug(f"Journey auto-merge: pruned cached clip {stale}")
        except OSError:
            pass


def build_pending_journey_clips(
    config: FrigateConfig, context: EmbeddingsContext, limit: int = 40
) -> dict:
    """Find recent multi-camera walks and make sure each has a merged clip
    on disk. Returns a summary dict. Never raises for a single failed
    chain -- one unbuildable clip must not stop the rest."""
    from frigate.util.journey_clip import build_merged_journey_clip
    from frigate.util.person_journey import find_recent_journey_sessions

    os.makedirs(JOURNEY_CLIP_DIR, exist_ok=True)

    sessions = find_recent_journey_sessions(config, context, limit=limit)

    considered = 0
    built = 0
    skipped_existing = 0
    failed = 0

    for session in sessions:
        cameras = {session["camera"]} | {m["camera"] for m in session["matches"]}
        if len(cameras) < 2:
            continue

        considered += 1
        out_path = merged_clip_path(session["event_id"])

        if os.path.exists(out_path):
            skipped_existing += 1
            continue

        event_ids = [session["event_id"]] + [m["event_id"] for m in session["matches"]]
        events = []
        for eid in event_ids:
            event = Event.get_or_none(Event.id == eid)
            if event is not None:
                events.append(event)

        if len(events) < 2:
            continue

        # write to a temp name first so a crash mid-encode can't leave a
        # truncated file that later looks like a valid cached clip
        tmp_path = f"{out_path}.partial"
        try:
            with open(tmp_path, "wb") as f:
                for chunk in build_merged_journey_clip(config, events):
                    f.write(chunk)
            os.replace(tmp_path, out_path)
            built += 1
            logger.info(
                f"Journey auto-merge: built merged clip for {session['event_id']} "
                f"across {len(cameras)} cameras ({len(events)} segments)"
            )
        except Exception as e:
            failed += 1
            logger.warning(
                f"Journey auto-merge: failed for {session['event_id']}: {e}"
            )
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    _prune_cache()

    return {
        "multi_camera_sessions": considered,
        "built": built,
        "already_cached": skipped_existing,
        "failed": failed,
    }


class JourneyClipMaintainer(threading.Thread):
    """Runs build_pending_journey_clips on a timer. Mirrors the
    threading.Thread + stop_event.wait(interval) pattern used by
    frigate.storage.StorageMaintainer.

    Fails open: any error in a pass is logged and retried next interval.
    Never touches the live detection or recording path."""

    def __init__(
        self, config: FrigateConfig, context: EmbeddingsContext, stop_event: MpEvent
    ) -> None:
        super().__init__(name="journey_clip_maintainer", daemon=True)
        self.config = config
        self.context = context
        self.stop_event = stop_event

    def run(self) -> None:
        interval = self.config.journey.auto_merge_interval_seconds

        if interval <= 0:
            logger.info(
                "Journey auto-merge: disabled (auto_merge_interval_seconds <= 0)"
            )
            return

        # let startup settle before the first (ffmpeg-heavy) pass
        if self.stop_event.wait(120):
            return

        while True:
            try:
                summary = build_pending_journey_clips(self.config, self.context)
                if summary["built"]:
                    logger.info(f"Journey auto-merge: {summary}")
                else:
                    logger.debug(f"Journey auto-merge: {summary}")
            except Exception as e:
                logger.warning(f"Journey auto-merge pass failed: {e}")

            if self.stop_event.wait(interval):
                break

        logger.info("Exiting journey clip maintainer...")
