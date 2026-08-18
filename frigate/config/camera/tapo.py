from pydantic import Field

from ..base import FrigateBaseModel
from ..env import EnvString

__all__ = ["TapoConfig"]


class TapoConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable Tapo local control",
        description="Enable local (no-cloud) control of a TP-Link Tapo camera's device settings, such as night vision mode.",
    )
    host: EnvString = Field(
        default="",
        title="Camera IP address",
        description="LAN IP address of the Tapo camera.",
    )
    username: EnvString | None = Field(
        default=None,
        title="Account username or email",
        description="Tapo local Camera Account username, or the Tapo app cloud account email if the camera account doesn't authenticate.",
    )
    password: EnvString | None = Field(
        default=None,
        title="Account password",
        description="Password matching the configured username.",
    )
