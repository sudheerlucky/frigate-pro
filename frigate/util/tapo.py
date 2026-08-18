"""Local (no-cloud) control of TP-Link Tapo camera device settings, such as
night vision / IR mode, via the pytapo library's reverse-engineered local
camera API.

This is separate from the camera's RTSP stream credentials -- Tapo cameras
authenticate the local control API against either a dedicated "Camera
Account" (Advanced Settings > Camera Account in the Tapo app) or the Tapo
app's own cloud account email/password, neither of which is necessarily the
same as the RTSP account.
"""

from pytapo import Tapo

from frigate.config.camera.tapo import TapoConfig

VALID_DAY_NIGHT_MODES = ("off", "on", "auto")


def _client(config: TapoConfig) -> Tapo:
    return Tapo(config.host, config.username, config.password)


def get_night_vision_mode(config: TapoConfig) -> str:
    """Returns the current day/night mode: 'off' (never IR), 'on' (always
    IR), or 'auto' (sensor-based)."""
    return _client(config).getDayNightMode()


def set_night_vision_mode(config: TapoConfig, mode: str) -> None:
    if mode not in VALID_DAY_NIGHT_MODES:
        raise ValueError(f"mode must be one of {VALID_DAY_NIGHT_MODES}")

    _client(config).setDayNightMode(mode)
