"""Link a person's appearances across cameras into a likely journey.

Primary signal is face recognition: if the source event already carries a
confident sub_label (a recognized name, not "unknown"), any other-camera
event with the same sub_label in the time window is treated as a confirmed
match -- this is just the identity Frigate's face pipeline already assigned,
not a new inference.

Fallback, when face isn't available or didn't produce a name, is a
dedicated person re-identification model (OSNet, MSMT17 fine-tuned --
frigate/embeddings/onnx/person_reid.py) run against each event's cropped
thumbnail. This is a real re-ID model, more precise than a general-purpose
visual-similarity embedding, but still fundamentally probabilistic -- no
re-ID system reaches 100% accuracy, so matches are always reported as
"likely same person" with a confidence score, never as confirmed identity.
"""

import logging

from frigate.config import FrigateConfig
from frigate.embeddings import EmbeddingsContext
from frigate.embeddings.onnx.person_reid import PersonReidEmbedding, cosine_similarity
from frigate.models import Event
from frigate.util.file import get_event_thumbnail_bytes

logger = logging.getLogger(__name__)

DEFAULT_TIME_WINDOW_SECONDS = 30
DEFAULT_MIN_APPEARANCE_CONFIDENCE = 0.5  # cosine similarity; higher = more similar

_reid_embedder: PersonReidEmbedding | None = None


def _get_reid_embedder() -> PersonReidEmbedding:
    global _reid_embedder
    if _reid_embedder is None:
        _reid_embedder = PersonReidEmbedding()
    return _reid_embedder


def find_likely_journey_matches(
    config: FrigateConfig,
    context: EmbeddingsContext,
    event_id: str,
    time_window_seconds: int = DEFAULT_TIME_WINDOW_SECONDS,
    min_appearance_confidence: float = DEFAULT_MIN_APPEARANCE_CONFIDENCE,
) -> list[dict]:
    """Find other-camera person events likely to be the same individual as
    `event_id`, within a time window around when it ended. Returns a list of
    {event_id, camera, confidence, method} sorted best-match first, where
    method is 'face' (confirmed identity) or 'appearance' (re-ID model
    similarity, not a guarantee)."""
    source = Event.get_or_none(Event.id == event_id)

    if source is None or source.label != "person":
        return []

    window_start = source.end_time - time_window_seconds
    window_end = source.end_time + time_window_seconds

    candidates = list(
        Event.select().where(
            (Event.camera != source.camera)
            & (Event.label == "person")
            & (Event.start_time >= window_start)
            & (Event.start_time <= window_end)
        )
    )

    if not candidates:
        return []

    # Primary: exact identity match via face recognition, already computed
    if source.sub_label and source.sub_label != "unknown":
        face_matches = [
            {
                "event_id": c.id,
                "camera": c.camera,
                "confidence": 0.99,
                "method": "face",
            }
            for c in candidates
            if c.sub_label == source.sub_label
        ]
        if face_matches:
            return face_matches

    # Fallback: dedicated re-ID model similarity, computed on-demand for
    # just this event + its candidates -- not wired into the live
    # detection/recording pipeline, so a slow or failed re-ID pass here can
    # never affect recording.
    source_bytes = get_event_thumbnail_bytes(source)
    if source_bytes is None:
        return []

    embedder = _get_reid_embedder()
    source_embedding = embedder.embed(source_bytes)
    if source_embedding is None:
        return []

    matches = []
    for candidate in candidates:
        candidate_bytes = get_event_thumbnail_bytes(candidate)
        if candidate_bytes is None:
            continue

        candidate_embedding = embedder.embed(candidate_bytes)
        if candidate_embedding is None:
            continue

        confidence = cosine_similarity(source_embedding, candidate_embedding)
        if confidence < min_appearance_confidence:
            continue

        matches.append(
            {
                "event_id": candidate.id,
                "camera": candidate.camera,
                "confidence": round(confidence, 3),
                "method": "appearance",
            }
        )

    matches.sort(key=lambda m: m["confidence"], reverse=True)
    return matches
