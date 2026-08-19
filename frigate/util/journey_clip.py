"""Merge a confirmed Journey chain's per-camera clips into a single
downloadable video, in chronological order.

The individual per-event clips already come from different cameras that
can (and here, do) have different resolutions -- 2304x1296 vs 2560x1440 in
this deployment -- so a naive ffmpeg concat-demuxer stream-copy (fine for
stitching same-camera recording segments, see frigate.api.media.recording_clip)
would produce a broken or glitchy file. Instead this re-encodes every clip
to a single common resolution (the smaller of the two, to avoid pretending
an upscale adds real detail) before concatenating, via ffmpeg's concat
*filter* rather than the concat *demuxer*.

This is real transcoding work, not a stream copy, so it takes real
wall-clock time proportional to the total footage length -- expected and
accepted for an explicit, user-triggered "merge this journey" action, not
something run automatically.

Video-only: cameras in this deployment may or may not have audio enabled,
and reconciling mismatched audio tracks across a multi-camera merge adds
real complexity for a feature whose whole point is showing movement, not
sound -- so audio is dropped entirely rather than half-supported.
"""

import logging
import os
import subprocess as sp
import time
import uuid
from collections.abc import Iterator

from pathvalidate import sanitize_filename

from frigate.config import FrigateConfig
from frigate.const import CACHE_DIR
from frigate.models import Event, Recordings

logger = logging.getLogger(__name__)

MERGE_TARGET_WIDTH = 2304
MERGE_TARGET_HEIGHT = 1296


def _build_playlist_file(camera: str, start_ts: float, end_ts: float) -> str | None:
    """Same approach as frigate.api.media.recording_clip's playlist step:
    an ffmpeg concat-demuxer playlist over this one camera's recording
    segments for the given time range. Returns the playlist file path, or
    None if no recordings cover this range."""
    recordings = (
        Recordings.select(Recordings.path, Recordings.start_time, Recordings.end_time)
        .where(
            (Recordings.start_time.between(start_ts, end_ts))
            | (Recordings.end_time.between(start_ts, end_ts))
            | ((start_ts > Recordings.start_time) & (end_ts < Recordings.end_time))
        )
        .where(Recordings.camera == camera)
        .order_by(Recordings.start_time.asc())
    )

    if recordings.count() == 0:
        return None

    file_name = sanitize_filename(
        f"journey_playlist_{camera}_{start_ts}-{end_ts}-{uuid.uuid4().hex[:8]}.txt"
    )
    file_path = os.path.join(CACHE_DIR, file_name)
    with open(file_path, "w") as f:
        for clip in recordings:
            f.write(f"file '{clip.path}'\n")
            if clip.start_time < start_ts:
                f.write(f"inpoint {int(start_ts - clip.start_time)}\n")
            if clip.end_time > end_ts:
                f.write(f"outpoint {int(end_ts - clip.start_time)}\n")

    return file_path


def _extract_event_clip(config: FrigateConfig, event: Event) -> str | None:
    """Extract one event's own-camera clip to a standalone temp MP4 file
    (stream-copy, safe within a single camera's own recordings). Returns
    the output file path, or None if no recordings were available."""
    end_ts = time.time() if event.end_time is None else event.end_time
    playlist_path = _build_playlist_file(event.camera, event.start_time, end_ts)
    if playlist_path is None:
        return None

    output_name = sanitize_filename(f"journey_clip_{event.id}-{uuid.uuid4().hex[:8]}.mp4")
    output_path = os.path.join(CACHE_DIR, output_name)

    ffmpeg_cmd = [
        config.ffmpeg.ffmpeg_path,
        "-hide_banner",
        "-y",
        "-protocol_whitelist",
        "pipe,file",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        playlist_path,
        "-c",
        "copy",
        "-an",
        output_path,
    ]

    try:
        result = sp.run(ffmpeg_cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            logger.warning(
                f"Journey merge: failed to extract clip for {event.id}: "
                f"{result.stderr.decode(errors='replace')}"
            )
            return None
    finally:
        try:
            os.remove(playlist_path)
        except OSError:
            pass

    return output_path if os.path.isfile(output_path) else None


def build_merged_journey_clip(
    config: FrigateConfig, events: list[Event]
) -> Iterator[bytes]:
    """Extract each event's own clip, scale them all to a common
    resolution, concatenate in the given (chronological) order, and stream
    the merged result as it's encoded. `events` must already be in the
    order the merged clip should play -- typically [source, *chain]."""
    temp_clip_paths: list[str] = []

    try:
        for event in events:
            path = _extract_event_clip(config, event)
            if path is not None:
                temp_clip_paths.append(path)

        if len(temp_clip_paths) < 2:
            logger.warning(
                f"Journey merge: only {len(temp_clip_paths)} of {len(events)} "
                "event clips were available, nothing meaningful to merge"
            )
            return

        filter_parts = []
        inputs: list[str] = []
        for i, path in enumerate(temp_clip_paths):
            inputs += ["-i", path]
            filter_parts.append(
                f"[{i}:v]scale={MERGE_TARGET_WIDTH}:{MERGE_TARGET_HEIGHT},"
                f"setsar=1,fps=15[v{i}]"
            )

        concat_inputs = "".join(f"[v{i}]" for i in range(len(temp_clip_paths)))
        filter_complex = (
            ";".join(filter_parts)
            + f";{concat_inputs}concat=n={len(temp_clip_paths)}:v=1:a=0[outv]"
        )

        ffmpeg_cmd = [
            config.ffmpeg.ffmpeg_path,
            "-hide_banner",
            "-y",
            *inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:",
        ]

        with sp.Popen(ffmpeg_cmd, stdout=sp.PIPE, stderr=sp.PIPE) as ffmpeg:
            while True:
                data = ffmpeg.stdout.read(8192)
                if data:
                    yield data
                else:
                    if ffmpeg.returncode and ffmpeg.returncode != 0:
                        logger.error(
                            f"Journey merge: ffmpeg failed: "
                            f"{ffmpeg.stderr.read().decode(errors='replace')}"
                        )
                    break
    finally:
        for path in temp_clip_paths:
            try:
                os.remove(path)
            except OSError:
                pass
