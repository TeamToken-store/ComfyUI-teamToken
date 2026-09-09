"""Dynamic model list for the node dropdowns.

The list MUST come from the live catalog: a hardcoded one rots the first time a
model is added or repriced. But INPUT_TYPES is called on every /object_info
build, and a network round-trip there stalls ComfyUI's node list. So this layer
NEVER blocks INPUT_TYPES on the network: it returns whatever it has right now (a
cached list, or the bundled snapshot) and refreshes the live catalog in a
background thread whose result is picked up by the next /object_info build. The
dropdowns are therefore never empty and never wait for the wire.
"""

from __future__ import annotations

import json
import os
import threading
import time

from .client import TeamTokenClient
from .settings import catalog_server_url

_TTL_SECONDS = 300.0
# Background refresh only; kept short so a wedged host frees the worker thread
# quickly rather than pinning it for the whole request timeout.
_FETCH_TIMEOUT = 4.0
_FALLBACK_PATH = os.path.join(os.path.dirname(__file__), "catalog_fallback.json")

# module-level cache: (fetched_at, server_url, list-of-models). The URL is part
# of the key so changing the gateway in settings re-fetches instead of serving
# the previous gateway's models. Guarded by _lock because the refresh runs off
# the request thread.
_cache: tuple[float, str, list[dict]] | None = None
_lock = threading.Lock()
_refreshing = False


def _load_fallback() -> list[dict]:
    try:
        with open(_FALLBACK_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _refresh_now(url: str) -> list[dict]:
    """Fetch the live catalog for ``url`` and store it; return what was stored.

    Synchronous: the background thread calls this, and tests call it directly.
    A network failure or an empty live list (misconfig, wrong host) both degrade
    to the bundled snapshot, so the stored value is never empty when a snapshot
    exists — the dropdowns stay populated.
    """
    global _cache
    try:
        models = TeamTokenClient.fetch_catalog(url, timeout=_FETCH_TIMEOUT)
    except Exception as exc:  # network/DNS/timeout — degrade, never crash
        print(f"[teamToken] catalog fetch failed ({exc}); using bundled snapshot.")
        models = []
    if not models:
        models = _load_fallback()
    with _lock:
        _cache = (time.monotonic(), url, models)
    return models


def _spawn_refresh(url: str) -> None:
    """Kick off one background refresh for ``url`` (never two at once)."""
    global _refreshing
    with _lock:
        if _refreshing:
            return
        _refreshing = True

    def _run() -> None:
        global _refreshing
        try:
            _refresh_now(url)
        finally:
            with _lock:
                _refreshing = False

    threading.Thread(target=_run, name="teamtoken-catalog", daemon=True).start()


def _all_models() -> list[dict]:
    """Catalog entries: cached-or-snapshot NOW, live refresh in the background.

    Fresh cache for this gateway is returned as-is. Otherwise we return the best
    thing we already have on hand — a stale cache for the same gateway, or the
    bundled snapshot — WITHOUT touching the network, and spawn a background fetch
    so the next /object_info build serves live data. A snapshot read is a local
    file, so INPUT_TYPES never waits on a socket.
    """
    url = catalog_server_url()
    now = time.monotonic()
    with _lock:
        cached = _cache
    if cached and cached[1] == url and (now - cached[0]) < _TTL_SECONDS:
        return cached[2]
    # Same gateway but stale → keep showing its last-known list; different gateway
    # or cold start → the bundled snapshot (never the wrong gateway's models).
    snapshot = cached[2] if (cached and cached[1] == url) else _load_fallback()
    _spawn_refresh(url)
    return snapshot or _load_fallback()


def models_for(modality: str) -> list[str]:
    """Sorted model names for a modality ('image' | 'video'); never empty."""
    names = sorted(m["model"] for m in _all_models() if m.get("modality") == modality and m.get("model"))
    return names or ["(no models — check API/network)"]


def image_models() -> list[str]:
    return models_for("image")


def video_models() -> list[str]:
    return models_for("video")


def refresh() -> None:
    """Drop the cache so the next INPUT_TYPES build re-snapshots and re-fetches."""
    global _cache
    with _lock:
        _cache = None
