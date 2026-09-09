"""Where the API key and server URL come from, and in what order.

The key should be settable from ComfyUI's own Settings pane. ComfyUI persists
settings server-side in ``user/<user>/comfy.settings.json`` (written by the JS
extension in ``web/teamtoken.js``), so we read that file. But settings-file
layout is not a stable public API and varies across ComfyUI versions, so the key
is resolved from three sources with a precedence that never leaves a user stuck:

    per-node input  >  environment variable  >  settings file

The node input is the only source guaranteed to exist on every ComfyUI version;
env var suits headless/CI installs; the settings file is the nice in-UI path.
Server URL follows the same order and defaults to the gateway.
"""

from __future__ import annotations

import json
import os

DEFAULT_SERVER_URL = "https://api.teamtoken.store"

_APIKEY_SETTING = "teamToken.apiKey"
_SERVER_SETTING = "teamToken.serverUrl"


def _user_dir() -> str | None:
    """ComfyUI's user directory, where ``comfy.settings.json`` lives.

    Prefer the public ``folder_paths`` helper when present; fall back to the
    conventional ``./user`` next to the running process. Returns None if neither
    is usable — the caller then just skips the settings-file source.
    """
    try:
        import folder_paths  # provided by ComfyUI at runtime

        get = getattr(folder_paths, "get_user_directory", None)
        if callable(get):
            return get()
        base = getattr(folder_paths, "base_path", None)
        if base:
            return os.path.join(base, "user")
    except Exception:
        pass
    guess = os.path.join(os.getcwd(), "user")
    return guess if os.path.isdir(guess) else None


def _read_settings() -> dict:
    """Load ``comfy.settings.json`` for a single, explicitly-named profile.

    ComfyUI namespaces settings per user (``user/<name>/comfy.settings.json``);
    a single-user install — the overwhelming majority — uses ``default``. We read
    ONLY the ``default`` profile (and the legacy root file), never scanning
    sibling profiles: on a multi-user instance, picking an arbitrary profile's
    key by directory order would let one user spend another user's balance.

    Multi-user installs are supported without that leak by naming the profile via
    ``$TEAMTOKEN_COMFY_USER`` (a server-side env a shared workflow can't set), or
    by supplying the key through the node input / ``$TEAMTOKEN_API_KEY``.
    """
    root = _user_dir()
    if not root or not os.path.isdir(root):
        return {}
    named = os.environ.get("TEAMTOKEN_COMFY_USER", "").strip()
    candidates = []
    if named:
        candidates.append(os.path.join(root, named, "comfy.settings.json"))
    candidates += [os.path.join(root, "default", "comfy.settings.json"),
                   os.path.join(root, "comfy.settings.json")]
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and (_APIKEY_SETTING in data or _SERVER_SETTING in data):
            return data
    return {}


def resolve_api_key(node_input: str | None = None) -> str:
    """The API key to authenticate with, or "" if none is configured anywhere."""
    if node_input and node_input.strip():
        return node_input.strip()
    env = os.environ.get("TEAMTOKEN_API_KEY") or os.environ.get("TEAMTOKEN_KEY")
    if env and env.strip():
        return env.strip()
    val = _read_settings().get(_APIKEY_SETTING)
    return val.strip() if isinstance(val, str) else ""


def resolve_server_url(node_input: str | None = None) -> str:
    """Base gateway URL, trailing slash stripped, defaulting to the prod gateway."""
    if node_input and node_input.strip():
        return node_input.strip().rstrip("/")
    env = os.environ.get("TEAMTOKEN_SERVER_URL")
    if env and env.strip():
        return env.strip().rstrip("/")
    val = _read_settings().get(_SERVER_SETTING)
    if isinstance(val, str) and val.strip():
        return val.strip().rstrip("/")
    return DEFAULT_SERVER_URL


def catalog_server_url() -> str:
    """Server URL for the model catalog fetched at INPUT_TYPES time.

    INPUT_TYPES has no node context, so it can't read a per-node URL — it uses
    env/settings/default only. NOT cached: settings can change mid-session, and
    the catalog layer already keys its own cache by this URL, so re-reading here
    is what lets a server-URL change actually take effect.
    """
    return resolve_server_url(None)
