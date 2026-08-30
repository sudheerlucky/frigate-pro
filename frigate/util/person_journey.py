"""Link a person's appearances across cameras into a likely journey.

Walks hop-by-hop rather than doing one lookup with a single fixed window:
from the source event, find the single best other-camera match within a
short per-hop window, then repeat from that match, and so on. A short
per-hop window (tens of seconds -- one realistic camera-to-camera transit)
composes into a journey of any total length, which a single window sized
for the whole trip cannot do without also risking false positives on a
short trip.

Matching is whole-person appearance only -- face recognition is
deliberately NOT used here, not even as a hint. On this deployment faces
are usually too small, too turned, or too motion-blurred at the distances
these cameras cover to be trustworthy, and a wrong face match is worse
than no match: it silently welds two different people's timelines
together with false confidence. Person re-identification looks at build,
clothing, hair and gait cues across the whole body, which survives the
angles and distances that defeat face recognition.

The model is a dedicated person re-identification network (OSNet,
MSMT17 fine-tuned -- frigate/embeddings/onnx/person_reid.py). It is a
real re-ID model rather than a general-purpose visual-similarity
embedding, but still fundamentally probabilistic -- no re-ID system
reaches 100% accuracy, so matches are always reported as "likely same
person" with a confidence score, never as confirmed identity. Each hop's
confidence is relative to its own immediate predecessor, not the original
source event -- chained hops compound uncertainty, so a blended single
score across the whole chain would overstate how sure hop 4 really is.

Crop quality matters more than anything else for re-ID accuracy, so the
embedder is fed the event's clean snapshot cropped to the person's box at
native resolution (typically 300-700px on a side here) rather than the
175x175 thumbnail Frigate stores for UI use. Feeding a 175px thumbnail to
a re-ID model throws away most of the detail the model is looking at
before it ever sees the image. The thumbnail remains a fallback for
events with no snapshot.

Embeddings are cached per event id for the life of a call tree, because
the feed-level walk (find_recent_journey_sessions) re-examines the same
candidate events many times over -- without the cache that endpoint spent
most of its time re-embedding images it had already embedded, to the
point of timing out on larger windows.

Walks forward in time only. Clicking the first event of a trip sees the
whole thing; clicking a middle event only sees what comes after it, not
before. Backward-chaining would double the search per hop for a case
that's rare in practice (most journeys get inspected from their start).

Two different window lengths, not one fixed size for every hop. A
cross-camera hop (the person leaves one camera's view and should appear
on another) uses a short window scaled by how long the source event
itself lasted -- a person who lingered is plausibly slower to reach the
next camera too, so a brief glimpse and a long loiter don't get the same
travel-time budget. A same-camera re-appearance (the person leaves frame
and comes back to the SAME camera -- e.g. someone working in view who
steps in and out repeatedly) uses a much longer, separately configured
window, since "gone from view" on one camera says nothing about how long
until they're back in that same view, unlike an actual cross-camera
transit. Without this, a person who repeatedly re-enters one camera's
frame over hours shows up as a new standalone session every single time.
"""

import logging
import time

from frigate.config import FrigateConfig
from frigate.embeddings import EmbeddingsContext
from frigate.embeddings.onnx.person_reid import PersonReidEmbedding, cosine_similarity
from frigate.models import Event
from frigate.util.file import get_event_snapshot_bytes, get_event_thumbnail_bytes

logger = logging.getLogger(__name__)

DEFAULT_HOP_WINDOW_SECONDS = 45
DEFAULT_MAX_HOPS = 5
DEFAULT_MAX_TOTAL_SPAN_SECONDS = 600  # 10 minutes -- stop chaining past this regardless of hop count
DEFAULT_SAME_CAMERA_GAP_SECONDS = 1200  # 20 minutes -- a same-camera re-appearance is a continuation, not a new visit, even after a much longer gap than a cross-camera hop would tolerate
DEFAULT_MAX_SAME_CAMERA_SESSION_SECONDS = 21600  # 6 hours -- a chain that has stayed on one camera the whole time (e.g. someone working in view for hours) is one long presence, not a "trip" -- the tight cross-camera trip cap below doesn't apply to it
DEFAULT_MAX_SAME_CAMERA_HOPS = 50  # a same-camera presence can legitimately re-appear far more than a 5-hop cross-camera trip would
MAX_HOP_WINDOW_SECONDS = 180  # ceiling on the duration-scaled cross-camera window, even for a long-lingering source event
HOP_WINDOW_DURATION_FACTOR = 0.5  # cross-camera window grows by this fraction of the source event's own duration

_reid_embedder: PersonReidEmbedding | None = None


def _get_reid_embedder() -> PersonReidEmbedding:
    global _reid_embedder
    if _reid_embedder is None:
        _reid_embedder = PersonReidEmbedding()
    return _reid_embedder


def _get_person_crop_bytes(event: Event) -> bytes | None:
    """Best available image of just this person, for the re-ID model.

    Prefers the clean snapshot cropped to the person's own box at its
    native resolution -- several times the linear resolution of the stored
    thumbnail, which is a 175x175 UI asset. Re-ID accuracy tracks crop
    detail closely, so this is the single highest-leverage input to the
    whole match. Falls back to the thumbnail when no snapshot exists (an
    in-progress event, or snapshots disabled for that camera), so a
    missing snapshot degrades match quality instead of dropping the hop."""
    try:
        snapshot_bytes, _ = get_event_snapshot_bytes(
            event, ext="jpg", crop=True, quality=95
        )
        if snapshot_bytes:
            return snapshot_bytes
    except Exception as e:
        logger.debug(f"Journey: snapshot crop unavailable for {event.id}: {e}")

    return get_event_thumbnail_bytes(event)


def _embed_event(event: Event, cache: dict[str, "np.ndarray | None"]):
    """Embed an event's person crop, memoized per event id.

    find_recent_journey_sessions walks many overlapping chains, so the same
    candidate event is examined repeatedly within one request; embedding is
    by far the most expensive step, and re-running it per visit is what made
    that endpoint time out on larger windows."""
    if event.id in cache:
        return cache[event.id]

    embedding = None
    crop_bytes = _get_person_crop_bytes(event)
    if crop_bytes is not None:
        embedding = _get_reid_embedder().embed(crop_bytes)

    cache[event.id] = embedding
    return embedding


def _best_appearance_match(
    current: Event,
    candidates: list[Event],
    min_confidence: float,
    ambiguity_margin: float,
    cache: dict,
) -> tuple[Event, float] | None:
    """Highest-similarity candidate above the confidence floor -- but only if
    it's clearly the best one. If a second candidate scores within
    ambiguity_margin of the top score, that's not a confident match, that's
    two plausible people (e.g. a second person who happened to also be on
    the target camera around the same time): refuse to guess and return
    None rather than silently picking the higher of two close scores."""
    current_embedding = _embed_event(current, cache)
    if current_embedding is None:
        return None

    scored: list[tuple[Event, float]] = []
    for candidate in candidates:
        candidate_embedding = _embed_event(candidate, cache)
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
    same_camera_gap_seconds: int = DEFAULT_SAME_CAMERA_GAP_SECONDS,
    embedding_cache: dict | None = None,
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
    if embedding_cache is None:
        embedding_cache = {}

    source = Event.get_or_none(Event.id == event_id)

    if source is None or source.label != "person":
        return []

    chain: list[dict] = []
    visited_ids = {source.id}
    current = source
    all_same_camera_so_far = True

    # Loop bound is the larger (same-camera) cap; the tighter cross-camera
    # max_hops is enforced inside the loop once/if a hop actually crosses
    # cameras, same reasoning as the span cap below.
    for hop_number in range(1, DEFAULT_MAX_SAME_CAMERA_HOPS + 1):
        hop_cap = DEFAULT_MAX_SAME_CAMERA_HOPS if all_same_camera_so_far else max_hops
        if hop_number > hop_cap:
            break

        # A chain that hasn't left its starting camera yet is one ongoing
        # presence (see module docstring), so it gets the long same-camera
        # cap instead of the tight cross-camera trip cap. The moment a hop
        # actually crosses cameras, this chain is a trip and the tight cap
        # applies from then on.
        # current.end_time is None while the event is still in progress
        # (the person is still on screen) -- treat "now" as its effective
        # end for all the window/span math below rather than crashing.
        effective_end_time = (
            current.end_time if current.end_time is not None else time.time()
        )

        span_cap = (
            DEFAULT_MAX_SAME_CAMERA_SESSION_SECONDS
            if all_same_camera_so_far
            else max_total_span_seconds
        )
        if effective_end_time - source.start_time > span_cap:
            break

        # Cross-camera window scales with how long the source event itself
        # lasted, capped so a very long loiter doesn't blow the window out
        # indefinitely.
        current_duration = effective_end_time - current.start_time
        adaptive_hop_window = min(
            max(hop_window_seconds, current_duration * HOP_WINDOW_DURATION_FACTOR),
            MAX_HOP_WINDOW_SECONDS,
        )

        # Bounded below by current.start_time, not just
        # effective_end_time - adaptive_hop_window: for a short event, that
        # subtraction can land before the event's own start, which let a
        # candidate that started BEFORE current even began get matched as
        # if it were a forward hop. This is meant to walk forward only
        # (see module docstring) -- a match earlier than current's own
        # start is never legitimate here.
        cross_camera_start = max(current.start_time, effective_end_time - adaptive_hop_window)
        cross_camera_end = effective_end_time + adaptive_hop_window

        # Same-camera re-appearances only look forward from current's own
        # end -- an event starting inside current's own span isn't a
        # "return", it would overlap current itself.
        same_camera_end = effective_end_time + same_camera_gap_seconds

        candidates = list(
            Event.select().where(
                (Event.label == "person")
                & (Event.id.not_in(visited_ids))
                & (
                    (
                        (Event.camera != current.camera)
                        & (Event.start_time >= cross_camera_start)
                        & (Event.start_time <= cross_camera_end)
                    )
                    | (
                        (Event.camera == current.camera)
                        & (Event.start_time >= effective_end_time)
                        & (Event.start_time <= same_camera_end)
                    )
                )
            )
        )

        if not candidates:
            break

        appearance_match = _best_appearance_match(
            current,
            candidates,
            min_appearance_confidence,
            ambiguity_margin,
            embedding_cache,
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
                "same_camera": next_event.camera == current.camera,
                "current_activity": next_event.data.get("current_activity"),
            }
        )
        visited_ids.add(next_event.id)
        if next_event.camera != current.camera:
            all_same_camera_so_far = False
        current = next_event

    return chain


DEFAULT_SESSION_LIMIT = 50


def find_recent_journey_sessions(
    config: FrigateConfig,
    context: EmbeddingsContext,
    limit: int = DEFAULT_SESSION_LIMIT,
    hop_window_seconds: int = DEFAULT_HOP_WINDOW_SECONDS,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_total_span_seconds: int = DEFAULT_MAX_TOTAL_SPAN_SECONDS,
    same_camera_gap_seconds: int | None = None,
) -> list[dict]:
    """One entry per continuous cross-camera walk among the most recent
    `limit` person events, instead of one entry per raw event. Without
    this, a single walk that hops cameras 3 times shows up as 3 separate
    rows in a feed -- indistinguishable from 3 separate real visits.

    Walks events oldest-first: whichever event starts a chain "consumes"
    every event that chain reaches, so those consumed events are skipped
    rather than also becoming session heads of their own. An event that
    genuinely doesn't chain onward from anything earlier (including one
    where the chain-walk stopped due to ambiguity -- see
    _best_appearance_match) correctly becomes its own session instead of
    being silently folded into someone else's.

    Returns sessions newest-first, each shaped like a single
    find_likely_journey_matches call's result plus the head event's own
    identity: {event_id, camera, start_time, end_time, matches}."""
    if same_camera_gap_seconds is None:
        same_camera_gap_seconds = config.journey.same_camera_gap_seconds

    # Take the most recent `limit` events (descending), then walk them
    # oldest-first so a chain always processes before the events it
    # consumes -- selecting ascending+limit here would instead grab the
    # OLDEST events in the entire history, not the most recent ones.
    recent_events = list(
        Event.select()
        .where(Event.label == "person")
        .order_by(Event.start_time.desc())
        .limit(limit)
    )
    events = list(reversed(recent_events))

    # One cache for the whole feed walk: consecutive sessions examine
    # heavily overlapping candidate sets, so this is where the memoization
    # pays for itself most.
    embedding_cache: dict = {}

    consumed_ids: set[str] = set()
    sessions: list[dict] = []

    for event in events:
        if event.id in consumed_ids:
            continue

        chain = find_likely_journey_matches(
            config,
            context,
            event.id,
            hop_window_seconds=hop_window_seconds,
            max_hops=max_hops,
            max_total_span_seconds=max_total_span_seconds,
            same_camera_gap_seconds=same_camera_gap_seconds,
            embedding_cache=embedding_cache,
        )
        consumed_ids.update(match["event_id"] for match in chain)

        sessions.append(
            {
                "event_id": event.id,
                "camera": event.camera,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "sub_label": event.sub_label,
                "current_activity": event.data.get("current_activity"),
                "matches": chain,
            }
        )

    sessions.sort(key=lambda s: s["start_time"], reverse=True)
    return sessions
