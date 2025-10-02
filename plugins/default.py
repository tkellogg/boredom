from __future__ import annotations

from typing import Any, Dict, List

from .plugin_base import BasePlugin


class Plugin(BasePlugin):
    """Default no-op plugin.

    Use this as the baseline when no custom plugins are configured.
    """

    name = "default"

    # All BasePlugin methods already no-op; override nothing.

