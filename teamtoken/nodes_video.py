"""Video nodes: teamToken Video and teamToken Video Extend.

Video is billed by the minute and generation runs 30s–several minutes, so these
nodes never block the graph on a synchronous wait. They submit (HTTP 202), then
poll ``GET /v1/videos/{id}`` while advancing the node's progress bar, and only
then stream the finished MP4 through the gateway's own content proxy — the
provider CDN host is never exposed. Extend continues a previous teamToken job by
its ``ref_video_job_id``; that's why the Video node also outputs ``job_id``.
"""

from __future__ import annotations

import base64
import os
import re

from . import catalog
from .client import TeamTokenClient, TeamTokenError
from .common import build_media_params, poll_job, tensors_to_data_urls
from .settings import resolve_api_key, resolve_server_url

CATEGORY = "teamToken/video"
VIDEO_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]

# Native VIDEO output when the ComfyUI build supports it (nicer in-graph
# preview); otherwise the nodes emit the saved file path as a STRING so they
# still work on older builds. RETURN_TYPES is fixed at class-definition time, so
# this capability flag decides the socket shape once, here.
try:
    from comfy_api.input_impl import VideoFromFile  # type: ignore
    HAS_VIDEO_TYPE = True
except Exception:  # noqa: BLE001 — any import failure means "no native type"
    try:
        from comfy_api.input_impl.video_types import VideoFromFile  # type: ignore
        HAS_VIDEO_TYPE = True
    except Exception:  # noqa: BLE001
        VideoFromFile = None  # type: ignore
        HAS_VIDEO_TYPE = False

if HAS_VIDEO_TYPE:
    _VIDEO_RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    _VIDEO_RETURN_NAMES = ("video", "video_path", "job_id", "cost_usd")
else:
    _VIDEO_RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    _VIDEO_RETURN_NAMES = ("video_path", "video_url", "job_id", "cost_usd")


_MAX_VIDEO_BYTES = 80 * 1024 * 1024  # matches the gateway's cap on one decoded inline input
_VIDEO_EXTS = (".mp4", ".mov", ".webm")


def _comfy_dirs(*getters: str) -> list[str]:
    dirs = []
    try:
        import folder_paths

        for name in getters:
            fn = getattr(folder_paths, name, None)
            if callable(fn):
                try:
                    dirs.append(os.path.realpath(fn()))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass
    return dirs


def _output_dir() -> str:
    dirs = _comfy_dirs("get_output_directory")
    path = dirs[0] if dirs else os.path.join(os.getcwd(), "output")
    os.makedirs(path, exist_ok=True)
    return path


def _allowed_local_dirs() -> list[str]:
    """Folders a local video may be read from — ComfyUI's own input/output/temp.

    Reading an arbitrary path would let a shared workflow exfiltrate any file on
    disk (e.g. ``video=/etc/passwd``) by uploading it to the gateway.
    """
    dirs = _comfy_dirs("get_input_directory", "get_output_directory", "get_temp_directory")
    if not dirs:
        dirs = [os.path.realpath(os.path.join(os.getcwd(), d)) for d in ("input", "output")]
    return dirs


def _safe_name(job_id: str) -> str:
    """A filesystem-safe basename from a (server-provided, untrusted) job id."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", (job_id or "").strip()).lstrip(".") or "video"


def _video_input_value(s: str) -> str:
    """Normalize a user-supplied video reference into what the gateway accepts.

    A URL or data-URL passes through; a local file is base64-encoded, but only if
    it lives under a ComfyUI media folder, is a video type, and is within the size
    cap — otherwise it is refused. Anything else (e.g. a provider UUID) passes
    through untouched.
    """
    s = s.strip()
    if s.startswith(("http://", "https://", "data:")):
        return s
    if os.path.isfile(s):
        real = os.path.realpath(s)
        if not any(real == d or real.startswith(d + os.sep) for d in _allowed_local_dirs()):
            raise TeamTokenError("local video must live in ComfyUI's input/output folder — "
                                 "or pass a URL / data-URL instead", status=400)
        if os.path.splitext(real)[1].lower() not in _VIDEO_EXTS:
            raise TeamTokenError("input video must be .mp4/.mov/.webm", status=400)
        size = os.path.getsize(real)
        if size > _MAX_VIDEO_BYTES:
            raise TeamTokenError(f"input video is {size // 1024 // 1024} MB (> 80 MB) — "
                                 "pass a URL instead of a local file", status=400)
        with open(real, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:video/mp4;base64,{b64}"
    return s


def _finalize(client: TeamTokenClient, body: dict):
    """Download the finished video and shape the node's return tuple."""
    job_id = body.get("id", "")
    data = body.get("data") or []
    if not data:
        raise TeamTokenError("generation completed but returned no video")
    url = data[0].get("url") if isinstance(data[0], dict) else str(data[0])
    dest = os.path.join(_output_dir(), f"teamtoken_{_safe_name(job_id)}.mp4")
    client.download(url, dest)
    cost = str(body.get("cost_usd", ""))
    if HAS_VIDEO_TYPE:
        return (VideoFromFile(dest), dest, job_id, cost)
    return (dest, url, job_id, cost)


def _submit_and_poll(client: TeamTokenClient, path: str, payload: dict):
    status, body = client.post_json(path, payload)
    job_id = body.get("id")
    if status == 200 and str(body.get("status", "")).lower() == "completed":
        return _finalize(client, body)  # gateway finished inside the request
    if not job_id:
        raise TeamTokenError("gateway did not return a job id to poll")
    final = poll_job(client, f"/v1/videos/{job_id}",
                     estimate_seconds=120.0, max_seconds=900.0, interval=3.0,
                     on_status=lambda s: print(f"[teamToken] video {job_id}: {s}"))
    return _finalize(client, final)


class TeamTokenVideo:
    """Text-, image-, or video-to-video (and motion control) via teamToken."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.video_models(), {"tooltip": "Video model, fetched live from the teamToken catalog"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What the video should show"}),
            },
            "optional": {
                "duration": ("INT", {"default": 0, "min": 0, "max": 15,
                                     "tooltip": "0 = model's own default (recommended — each model allows "
                                                "different values, e.g. veo 4/6/8, grok only 6, "
                                                "kling-2.1-10s only 10); ignored by edit/motion models"}),
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"tooltip": "Output aspect ratio (default 16:9)"}),
                "image": ("IMAGE", {"tooltip": "Reference frame for image-to-video / motion character"}),
                "video": ("STRING", {"default": "",
                                     "tooltip": "URL or local path for video-to-video / motion control"}),
                # Not sent to the API — steers ComfyUI's input-equality cache so a
                # re-run only pays for a NEW clip; control_after_generate adds the
                # fixed/increment/randomize companion across frontend versions.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Change to force a fresh (paid) generation; identical inputs reuse the cached result"}),
                "api_key": ("STRING", {"default": "", "tooltip": "Overrides the teamToken settings key. "
                       "⚠️ This value is saved INTO the workflow file — prefer the teamToken "
                       "Settings pane or $TEAMTOKEN_API_KEY so your key isn't shared with the workflow"}),
                "server_url": ("STRING", {"default": "",
                                          "tooltip": "Leave empty to use the teamToken settings / env / default gateway"}),
            },
        }

    RETURN_TYPES = _VIDEO_RETURN_TYPES
    RETURN_NAMES = _VIDEO_RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model, prompt, duration=0, aspect_ratio="16:9", image=None,
                 video="", seed=0, api_key="", server_url=""):
        if not prompt or not prompt.strip():
            raise TeamTokenError("prompt is required", status=400, code="EMPTY_PROMPT")
        client = TeamTokenClient(resolve_server_url(server_url), resolve_api_key(api_key))
        extra = {}
        if image is not None:
            urls = tensors_to_data_urls(image)
            extra["image"] = urls if len(urls) > 1 else urls[0]
        if video and video.strip():
            extra["video"] = _video_input_value(video)
        payload = {"model": model, **build_media_params(
            prompt, aspect_ratio=aspect_ratio, duration=duration, extra=extra)}
        return _submit_and_poll(client, "/v1/videos", payload)


class TeamTokenVideoExtend:
    """Extend a previous teamToken video, referenced by its job id."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.video_models(), {"tooltip": "Video model, fetched live from the teamToken catalog"}),
                "ref_video_job_id": ("STRING", {"default": "",
                                                "tooltip": "job_id output of a prior teamToken Video node"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "How the extended segment should continue"}),
            },
            "optional": {
                "aspect_ratio": (VIDEO_ASPECT_RATIOS, {"tooltip": "Output aspect ratio (default 16:9)"}),
                # Not sent to the API — steers ComfyUI's input-equality cache; see
                # the seed comment in TeamTokenVideo.INPUT_TYPES.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": True,
                                 "tooltip": "Change to force a fresh (paid) generation; identical inputs reuse the cached result"}),
                "api_key": ("STRING", {"default": "", "tooltip": "Overrides the teamToken settings key. "
                       "⚠️ This value is saved INTO the workflow file — prefer the teamToken "
                       "Settings pane or $TEAMTOKEN_API_KEY so your key isn't shared with the workflow"}),
                "server_url": ("STRING", {"default": "",
                                          "tooltip": "Leave empty to use the teamToken settings / env / default gateway"}),
            },
        }

    RETURN_TYPES = _VIDEO_RETURN_TYPES
    RETURN_NAMES = _VIDEO_RETURN_NAMES
    FUNCTION = "extend"
    CATEGORY = CATEGORY

    def extend(self, model, ref_video_job_id, prompt, aspect_ratio="16:9",
               seed=0, api_key="", server_url=""):
        if not ref_video_job_id or not ref_video_job_id.strip():
            raise TeamTokenError("ref_video_job_id is required — wire it from a teamToken Video node",
                                 status=400)
        client = TeamTokenClient(resolve_server_url(server_url), resolve_api_key(api_key))
        payload = {"model": model, "ref_video_job_id": ref_video_job_id.strip(),
                   **build_media_params(prompt, aspect_ratio=aspect_ratio)}
        return _submit_and_poll(client, "/v1/videos/extend", payload)
