# frigate-pro Roadmap

Personal fork of [Frigate](https://github.com/blakeblackshear/frigate) for a home security
deployment. This file tracks what's shipped and what's planned.

## Shipped

- Immich integration: native config, Face Library rebuild button, Export-to-Immich, auto-identify
  unknown faces against Immich's named people (local matching -- Immich has no reverse face-search
  API), on a background scheduler
- Cross-camera person "Journey" tracking: face-primary, OSNet re-ID appearance-model fallback,
  ambiguity-margin safety check, hop-by-hop chain walking
- Journeys feed: session deduplication (one row per walk/presence, not per raw event),
  same-camera repeat-appearance merging, adaptive hop windows (cross-camera window scales with
  source event duration; same-camera re-appearance gets a much longer window)
- Journey view: mobile-compatible HLS video playback, per-hop timeline, merged single-clip
  download
- Side-profile face rejection in Immich auto-identify (landmark-based frontality check) --
  root-caused a false-registration incident where side-profile crops were auto-matched to the
  wrong person at high confidence
- Motion-based activity classification (standing/sitting/walking/running), independent of face
  recognition, from each tracked person's own box motion/shape
- IR/night-vision toggle for the Tapo C110 entrance camera (Frigate UI button + Home Assistant
  MQTT discovery `select` entity)
- Fixed a credential leak: `/api/config` was returning Tapo password and Immich API key in
  plaintext

## In progress / actively diagnosing

- **Entrance Tapo night-vision entity not visible in Home Assistant** -- confirmed via direct MQTT
  broker read that Frigate publishes the discovery config and state correctly; the gap is on HA's
  side (never discovered, or discovered-but-disabled). Waiting on a check of HA's entity registry.

## Planned

### Recognition & tracking
- ONVIF-based IR/night-vision toggle for the GF camera (it's a Hikvision-style camera reached via
  ONVIF, not Tapo -- the existing toggle doesn't apply to it)
- Surface the activity classifier (standing/sitting/walking/running) in the UI -- currently
  backend-only
- Adopt "best frame from the whole clip + multi-frame agreement" for auto-identify matching,
  instead of matching every saved crop independently (inspired by the suslik project)
- Evaluate a VLM-based "vision detect" as a third, independent recognition path (local
  llama.cpp/OpenAI-compatible endpoint judging a candidate grid against approved galleries)
- License plate recognition (ALPR)
- Vehicle make/model/color classification
- Package detection with dwell-time alerting
- Extended audio event detection
- Zone-based loitering detection
- Multi-camera 3D positional tracking
- Anomaly detection for unusual activity

### Home Assistant / notifications
- Deepen Home Assistant integration beyond Tapo (ONVIF-based camera controls, etc.)
- HA Voice "who's at the door" response
- Native push notifications
- Slack/Discord webhook alerts
- Matter/Google/Apple Home bridge

### Dashboards & UI
- Known visitors dashboard
- Immich sync status dashboard
- Per-camera health scoring dashboard
- Timeline activity heatmap
- Face Library bulk management UI
- Journey view v2 with floorplan overlay
- Multi-instance fleet dashboard

### Platform & ops
- Fix `events_search` crash on `None` `search_distance` during sort (real bug, spotted in logs)
- Investigate GF camera QSV decode crash-loop
- Startup smoke-test / self-check
- Structured JSON logging + log shipping
- Automated backup/restore
- Config diff/rollback
- Embedding store retention policy + pruning UI
- Duplicate-export guard for Immich
- Encrypt local face embedding backups
- Refine RBAC to per-camera permissions
- Audit log for config/data changes
- API rate limiting and abuse protection
