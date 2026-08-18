"""Import reference face photos for a person from a self-hosted Immich
instance, scored and filtered for quality before handing them to Frigate's
face recognizer.

Uses Immich's own face-detection bounding boxes to crop to the correct
individual in a photo -- handing Frigate the whole photo and letting it
guess the largest face in frame risks registering the wrong person under a
multi-person photo.
"""

import logging

import cv2
import numpy as np
import requests

from frigate.config.classification import ImmichConfig

logger = logging.getLogger(__name__)

MARGIN = 0.4  # extra context around the tight face box, as a fraction of box size


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
    config: ImmichConfig, person_id: str
) -> list[dict]:
    """Fetch, crop, and quality-score every candidate face photo for a
    person. Returns a list of dicts (sorted best-quality first) each with
    the encoded crop bytes and its sharpness/size scores, filtered to those
    meeting the configured minimums. Does not touch Frigate -- caller
    decides what to do with the results (e.g. take the top N)."""
    asset_ids = _get_asset_ids(config, person_id)
    logger.info(f"Immich import: found {len(asset_ids)} candidate photos")

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
