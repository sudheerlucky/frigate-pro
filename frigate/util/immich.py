"""Immich integration: import reference face photos for face recognition,
and export Frigate clips to a self-hosted Immich instance.

Uses Immich's own face-detection bounding boxes to crop to the correct
individual in a photo -- handing Frigate the whole photo and letting it
guess the largest face in frame risks registering the wrong person under a
multi-person photo.
"""

import datetime
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import requests

from frigate.config.classification import ImmichConfig
from frigate.const import FACE_DIR

logger = logging.getLogger(__name__)

MARGIN = 0.4  # extra context around the tight face box, as a fraction of box size


def _imported_assets_path(name: str) -> Path:
    return Path(FACE_DIR) / name / ".immich_imported.json"


def load_imported_asset_ids(name: str) -> set[str]:
    """Immich asset ids already registered with Frigate for this person in a
    previous import, so re-running the import doesn't re-download and
    re-register the same photos every time."""
    path = _imported_assets_path(name)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception as e:
        logger.warning(f"Immich import: could not read imported-asset record for {name}: {e}")
        return set()


def save_imported_asset_ids(name: str, asset_ids: set[str]) -> None:
    path = _imported_assets_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(asset_ids)))


class ImmichImportError(Exception):
    pass


def _get(config: ImmichConfig, path: str, **kwargs) -> requests.Response:
    resp = requests.get(
        f"{config.host}{path}",
        headers={"x-api-key": config.api_key},
        timeout=30,
        **kwargs,
    )
    resp.raise_for_status()
    return resp


def _post_json(config: ImmichConfig, path: str, body: dict) -> dict:
    resp = requests.post(
        f"{config.host}{path}",
        json=body,
        headers={"x-api-key": config.api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def find_person_id(config: ImmichConfig, name: str) -> str | None:
    """Match a Frigate face-library name to an Immich person by name
    (case-insensitive). Returns None if no unambiguous match is found."""
    resp = _get(config, "/api/people", params={"withHidden": "false"})
    matches = [
        p for p in resp.json()["people"] if p["name"].lower() == name.lower()
    ]
    return matches[0]["id"] if len(matches) == 1 else None


def _get_asset_ids(config: ImmichConfig, person_id: str) -> list[str]:
    result = _post_json(
        config, "/api/search/metadata", {"personIds": [person_id], "size": 1000}
    )
    return [a["id"] for a in result["assets"]["items"]]


def _get_face_box(
    config: ImmichConfig, asset_id: str, person_id: str
) -> tuple[float, float, float, float, int, int] | None:
    """Return this person's face box, or None if Immich detected zero or
    more than one candidate for this person in this asset (ambiguous)."""
    resp = _get(config, "/api/faces", params={"id": asset_id})
    matches = [
        f for f in resp.json() if f.get("person") and f["person"]["id"] == person_id
    ]

    if len(matches) != 1:
        return None

    f = matches[0]
    return (
        f["boundingBoxX1"],
        f["boundingBoxY1"],
        f["boundingBoxX2"],
        f["boundingBoxY2"],
        f["imageWidth"],
        f["imageHeight"],
    )


def _crop_to_face(
    image_bytes: bytes, box: tuple[float, float, float, float, int, int]
) -> tuple[np.ndarray, int, int]:
    x1, y1, x2, y2, det_w, det_h = box
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    actual_h, actual_w = img.shape[:2]

    # Immich's face box is relative to whatever resolution its detector ran
    # at (det_w x det_h), which may differ from the image we downloaded --
    # scale proportionally.
    sx, sy = actual_w / det_w, actual_h / det_h
    x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy

    box_w, box_h = x2 - x1, y2 - y1
    x1 -= box_w * MARGIN
    y1 -= box_h * MARGIN
    x2 += box_w * MARGIN
    y2 += box_h * MARGIN

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(actual_w, int(x2)), min(actual_h, int(y2))

    cropped = img[y1:y2, x1:x2]
    return cropped, x2 - x1, y2 - y1


def _sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fetch_candidate_faces(
    config: ImmichConfig, person_id: str, skip_asset_ids: set[str] | None = None
) -> list[dict]:
    """Fetch, crop, and quality-score every candidate face photo for a
    person. Returns a list of dicts (sorted best-quality first) each with
    the encoded crop bytes and its sharpness/size scores, filtered to those
    meeting the configured minimums. Does not touch Frigate -- caller
    decides what to do with the results (e.g. take the top N).

    skip_asset_ids lets a re-import only fetch photos that weren't already
    registered in a previous run, instead of blindly re-downloading and
    re-registering the whole library every time."""
    all_asset_ids = _get_asset_ids(config, person_id)
    skip_asset_ids = skip_asset_ids or set()
    asset_ids = [a for a in all_asset_ids if a not in skip_asset_ids]
    logger.info(
        f"Immich import: found {len(all_asset_ids)} candidate photos, "
        f"{len(asset_ids)} new (skipping {len(all_asset_ids) - len(asset_ids)} already imported)"
    )

    candidates = []
    for asset_id in asset_ids:
        box = _get_face_box(config, asset_id, person_id)
        if box is None:
            continue

        try:
            resp = _get(config, f"/api/assets/{asset_id}/thumbnail", params={"size": "preview"})
            cropped, w, h = _crop_to_face(resp.content, box)
        except Exception as e:
            logger.debug(f"Immich import: skipping {asset_id}, crop failed: {e}")
            continue

        if w < config.min_face_size or h < config.min_face_size:
            continue

        sharpness = _sharpness(cropped)
        if sharpness < config.min_sharpness:
            continue

        ok, encoded = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue

        candidates.append(
            {
                "asset_id": asset_id,
                "image_bytes": encoded.tobytes(),
                "sharpness": sharpness,
                "width": w,
                "height": h,
            }
        )

    candidates.sort(key=lambda c: c["sharpness"], reverse=True)
    logger.info(
        f"Immich import: {len(candidates)}/{len(asset_ids)} candidates passed quality filtering"
    )
    return candidates[: config.max_images_per_person]


def _export_filename(export_time: float, camera: str, label: str | None) -> str:
    """HomeKit Secure Video-style filename: date and time first, so exports
    sort and scan chronologically at a glance, camera/label appended for
    context."""
    dt = datetime.datetime.fromtimestamp(export_time)
    stamp = dt.strftime("%Y-%m-%d %H-%M-%S")
    suffix = f" {label}" if label else ""
    return f"{stamp} {camera}{suffix}.mp4"


def ensure_album(config: ImmichConfig, album_name: str) -> str:
    """Return the id of an Immich album with this name, creating it if it
    doesn't exist yet."""
    resp = _get(config, "/api/albums")
    for album in resp.json():
        if album["albumName"] == album_name:
            return album["id"]

    created = _post_json(config, "/api/albums", {"albumName": album_name, "assetIds": []})
    logger.info(f"Immich export: created album '{album_name}' ({created['id']})")
    return created["id"]


def upload_video_export(
    config: ImmichConfig,
    video_path: str,
    export_time: float,
    camera: str,
    label: str | None,
    album_name: str,
    job_id: str,
) -> str:
    """Upload an exported clip to Immich, named like HomeKit Secure Video
    exports (date + time first), and add it to the given album (created if
    missing). Returns the Immich asset id.

    device_asset_id is derived from the export job's own id rather than a
    fresh random uuid4 per call. Immich's /api/assets upload is designed to
    be idempotent on (deviceId, deviceAssetId): re-uploading the same pair
    returns the existing asset instead of creating a duplicate. A random id
    per call defeated that entirely -- if this job ever got reprocessed
    (e.g. the export job was mid-upload during a container restart), it
    would silently create a second copy of the same clip in Immich every
    time. A stable id keyed to the job makes a re-run a no-op instead."""
    path = Path(video_path)
    filename = _export_filename(export_time, camera, label)
    iso_time = datetime.datetime.fromtimestamp(export_time).isoformat() + "Z"
    device_asset_id = f"frigate-pro-export-{job_id}"

    with open(path, "rb") as f:
        resp = requests.post(
            f"{config.host}/api/assets",
            headers={"x-api-key": config.api_key},
            data={
                "deviceAssetId": device_asset_id,
                "deviceId": "frigate-pro",
                "fileCreatedAt": iso_time,
                "fileModifiedAt": iso_time,
            },
            files={"assetData": (filename, f, "video/mp4")},
            timeout=300,
        )
    resp.raise_for_status()
    asset_id = resp.json()["id"]

    album_id = ensure_album(config, album_name)
    add_resp = requests.put(
        f"{config.host}/api/albums/{album_id}/assets",
        headers={"x-api-key": config.api_key},
        json={"ids": [asset_id]},
        timeout=30,
    )
    add_resp.raise_for_status()

    logger.info(f"Immich export: uploaded '{filename}' as {asset_id}")
    return asset_id
