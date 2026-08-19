"""Lightweight, camera-agnostic activity classification (standing / sitting /
walking / running) from a tracked object's own bounding-box motion.

Deliberately independent of face recognition or identity -- it only looks
at how the box moves and its shape frame to frame, so it classifies a
known person and an unknown one exactly the same way, and needs no
per-camera calibration (unlike the real-world speed-zone feature, which
requires configured zone.distances).
"""

# Body-heights of on-screen travel per second. Normalizing by the
# object's own bounding-box height keeps this roughly scale-invariant
# to how far the person is from the camera, instead of a raw pixel
# speed that would mean something different at 3m vs 15m.
WALK_SPEED_THRESHOLD = 0.15
RUN_SPEED_THRESHOLD = 0.8

# width/height above this reads as a "squashed" box -- consistent with
# a seated posture rather than a standing or moving one.
SITTING_ASPECT_RATIO = 0.85


def classify_activity(normalized_speed: float, aspect_ratio: float) -> str:
    """Classify a single frame's motion sample into an activity label.
    Callers should smooth over several frames (e.g. majority vote) rather
    than trusting a single sample, since detection jitter alone can spike
    normalized_speed for one frame."""
    if normalized_speed >= RUN_SPEED_THRESHOLD:
        return "running"
    if normalized_speed >= WALK_SPEED_THRESHOLD:
        return "walking"
    return "sitting" if aspect_ratio >= SITTING_ASPECT_RATIO else "standing"
