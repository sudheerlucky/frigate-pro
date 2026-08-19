"""Link a person's appearances across cameras into a likely journey.

Walks hop-by-hop rather than doing one lookup with a single fixed window:
from the source event, find the single best other-camera match within a
short per-hop window, then repeat from that match, and so on. A short
per-hop window (tens of seconds -- one realistic camera-to-camera transit)
composes into a journey of any total length, which a single window sized
for the whole trip cannot do without also risking false positives on a
short trip.

Primary signal per hop is face recognition: if the current event already
carries a confident sub_label (a recognized name, not "unknown"), the
closest-in-time other-camera event with the same sub_label is a confirmed
match -- this is just the identity Frigate's face pipeline already
assigned, not a new inference.

Fallback per hop, when face isn't available or didn't produce a name, is a
dedicated person re-identification model (OSNet, MSMT17 fine-tuned --
frigate/embeddings/onnx/person_reid.py) run against each candidate's
cropped thumbnail. This is a real re-ID model, more precise than a
general-purpose visual-similarity embedding, but still fundamentally
probabilistic -- no re-ID system reaches 100% accuracy, so matches are
always reported as "likely same person" with a confidence score, never as
confirmed identity. Each hop's confidence is relative to its own immediate
predecessor, not the original source event -- chained hops compound
uncertainty, so a blended single score across the whole chain would
overstate how sure hop 4 really is.

Walks forward in time only. Clicking the first event of a trip sees the
whole thing; clicking a middle event only sees what comes after it, not
before. Backward-chaining would double the search per hop for a case
that's rare in practice (most journeys get inspected from their start).
"""

import logging

from frigate.config import FrigateConfig
from frigate.embeddings import EmbeddingsContext
from frigate.embeddings.onnx.person_reid import PersonReidEmbedding, cosine_similarity
from frigate.models import Event
from frigate.util.file import get_event_thumbnail_bytes

logger = logging.getLogger(__name__)

DEFAULT_HOP_WINDOW_SECONDS = 45
DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_TOTAL_SPAN_SECONDS = 600  # 10 minutes -- stop chaining past this regardless of hop count

_reid_embedder: PersonReidEmbedding | None = None


def _get_reid_embedder() -> PersonReidEmbedding:
    global _reid_embedder
    if _reid_embedder is None:
        _reid_embedder = PersonReidEmbedding()
    return _reid_embedder


def _best_face_match(current: Event, candidates: list[Event]) -> Event | None:
    """Closest-in-time same-name candidate, or None if `current` has no
    confident name or no candidate shares it."""
    if not current.sub_label or current.sub_label == "unknown":
        return None

    named = [c for c in candidates if c.sub_label == current.sub_label]
    if not named:
        return None

    named.sort(key=lambda c: abs(c.start_time - current.end_time))
    return named[0]


def _best_appearance_match(
    current: Event,
    candidates: list[Event],
    min_confidence: float,
    ambiguity_margin: float,
) -> tuple[Event, float] | None:
    """Highest-similarity candidate above the confidence floor -- but only if
    it's clearly the best one. If a second candidate scores within
    ambiguity_margin of the top score, that's not a confident match, that's
    two plausible people (e.g. a second person who happened to also be on
    the target camera around the same time): refuse to guess and return
    None rather than silently picking the higher of two close scores."""
    current_bytes = get_event_thumbnail_bytes(current)
    if current_bytes is None:
        return None

    embedder = _get_reid_embedder()
    current_embedding = embedder.embed(current_bytes)
    if current_embedding is None:
        return None

    scored: list[tuple[Event, float]] = []
    for candidate in candidates:
        candidate_bytes = get_event_thumbnail_bytes(candidate)
        if candidate_bytes is None:
            continue

        candidate_embedding = embedder.embed(candidate_bytes)
        if candidate_embedding is None:
            continue

        confidence = cosine_similarity(current_embedding, candidate_embedding)
        if confidence < min_confidence:
            continue
        scored.append((candidate, confidence))

    if not scored:
        return None

    scored.sort(key=lambda pair: pair[1], reverse=True)
    best = scored[0]

    if len(scored) > 1:
        runner_up = scored[1]
        if best[1] - runner_up[1] < ambiguity_margin:
            logger.info(
                f"Journey: ambiguous hop from {current.id} -- top two candidates "
                f"{best[0].id} ({best[1]:.3f}) and {runner_up[0].id} ({runner_up[1]:.3f}) "
                f"are within {ambiguity_margin} of each other, refusing to guess"
            )
            return None

    return best


def find_likely_journey_matches(
    config: FrigateConfig,
    context: EmbeddingsContext,
    event_id: str,
    hop_window_seconds: int = DEFAULT_HOP_WINDOW_SECONDS,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_total_span_seconds: int = DEFAULT_MAX_TOTAL_SPAN_SECONDS,
    min_appearance_confidence: float | None = None,
    ambiguity_margin: float | None = None,
) -> list[dict]:
    """Chain-walk forward from `event_id` across cameras, one hop at a time.
    Returns the chain in chronological order as a list of
    {event_id, camera, confidence, method, start_time, hop}, where hop
    counts from 1 (the first link away from the source) and confidence is
    that hop's own match strength, not compounded across the chain.

    A hop stops the chain (rather than guessing) when the appearance model
    can't clearly tell two candidates apart -- e.g. a second person who
    happens to also be on the target camera around the same time. See
    _best_appearance_match.

    min_appearance_confidence and ambiguity_margin default to
    config.journey.* (user-tunable in Settings) when not explicitly
    overridden."""
    if min_appearance_confidence is None:
        min_appearance_confidence = config.journey.min_appearance_confidence
    if ambiguity_margin is None:
        ambiguity_margin = config.journey.ambiguity_margin

    source = Event.get_or_none(Event.id == event_id)

    if source is None or source.label != "person":
        return []

    chain: list[dict] = []
    visited_ids = {source.id}
    current = source

    for hop_number in range(1, max_hops + 1):
        if current.end_time - source.start_time > max_total_span_seconds:
            break

        window_start = current.end_time - hop_window_seconds
        window_end = current.end_time + hop_window_seconds

        candidates = list(
            Event.select().where(
                (Event.camera != current.camera)
                & (Event.label == "person")
                & (Event.start_time >= window_start)
                & (Event.start_time <= window_end)
                & (Event.id.not_in(visited_ids))
            )
        )

        if not candidates:
            break

        next_event: Event | None = None
        confidence: float
        method: str

        face_match = _best_face_match(current, candidates)
        if face_match is not None:
            next_event, confidence, method = face_match, 0.99, "face"
        else:
            appearance_match = _best_appearance_match(
                current, candidates, min_appearance_confidence, ambiguity_margin
            )
            if appearance_match is None:
                break
            next_event, confidence, method = (
                appearance_match[0],
                appearance_match[1],
                "appearance",
            )

        chain.append(
            {
                "event_id": next_event.id,
                "camera": next_event.camera,
                "confidence": round(confidence, 3),
                "method": method,
                "start_time": next_event.start_time,
                "end_time": next_event.end_time,
                "hop": hop_number,
            }
        )
        visited_ids.add(next_event.id)
        current = next_event

    return chain
