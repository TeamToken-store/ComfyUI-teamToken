"""Bridging between ComfyUI runtime types and the gateway wire format.

Kept separate from ``client.py`` so the HTTP layer stays free of torch/PIL and
can be unit-tested without a ComfyUI install. Everything ComfyUI-shaped lives
here: IMAGE tensors, the interruptible progress-polling loop, and output paths.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any, Callable

import numpy as np
import requests
import torch
from PIL import Image

from .client import TeamTokenClient, TeamTokenError

ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4"]
RESOLUTIONS = ["1K", "2K", "4K"]

# INPUT_TYPES needs a non-empty COMBO even before any catalog fetch succeeds.
_COMBO_EMPTY = "(no models — check API/network)"

# How many consecutive transient network errors a poll loop rides out before it
# gives up. A running job is billed server-side, so a brief outage must not
# abandon (and later double-bill) it — but a truly dead gateway shouldn't hang
# forever either. The overall max_seconds deadline still applies underneath.
_MAX_POLL_NET_ERRORS = 5


# -- IMAGE tensor <-> bytes ------------------------------------------------
def tensor_to_data_url(image: "torch.Tensor", index: int = 0) -> str:
    """Encode one frame of a ComfyUI IMAGE batch as a PNG data URL.

    ComfyUI IMAGE is [B,H,W,C] float32 in 0..1; the gateway accepts a data URL
    (or bare base64) for reference-image inputs.
    """
    arr = image[index].detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def tensors_to_data_urls(image: "torch.Tensor") -> list[str]:
    """Every frame of an IMAGE batch as data URLs (multi-reference inputs).

    A zero-frame batch ([0,H,W,C]) can reach here from upstream filter/batch
    nodes; callers index ``urls[0]``, so refuse it here with a bounded error
    instead of letting an IndexError surface far from the cause.
    """
    if image is None or image.shape[0] == 0:
        raise TeamTokenError("the IMAGE input is empty (0 frames) — connect a non-empty image")
    return [tensor_to_data_url(image, i) for i in range(image.shape[0])]


def _b64_to_tensor(b64_json: str) -> "torch.Tensor":
    raw = base64.b64decode(b64_json)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1,H,W,3]


def _pad_to(t: "torch.Tensor", height: int, width: int) -> "torch.Tensor":
    """Zero-pad a [1,H,W,C] frame (bottom/right) up to height×width."""
    _, h, w, c = t.shape
    if h == height and w == width:
        return t
    out = torch.zeros((1, height, width, c), dtype=t.dtype)
    out[:, :h, :w, :] = t
    return out


def images_payload_to_tensor(data: list[dict]) -> "torch.Tensor":
    """Turn a media ``data`` array of {b64_json} into one IMAGE batch.

    A ComfyUI IMAGE is a single stacked tensor, so frames of differing sizes
    can't be concatenated directly. Rather than silently drop the odd ones
    (losing generations the user paid for), we keep every frame's pixels and
    zero-pad each onto a common (batch-max) canvas — so smaller frames gain black
    borders; downstream nodes can crop back. Equal-sized frames (the common case,
    since one request uses one model+resolution) take the fast path unchanged.
    """
    # A malformed entry (a bare string/null instead of {b64_json}, or bytes that
    # aren't a valid image) must surface as a bounded error, not a raw
    # AttributeError/binascii.Error/UnidentifiedImageError — the generation was
    # already paid for, so the failure has to be legible enough to act on.
    tensors = []
    for item in data:
        if not isinstance(item, dict) or not item.get("b64_json"):
            continue
        try:
            tensors.append(_b64_to_tensor(item["b64_json"]))
        except (ValueError, OSError) as exc:  # binascii.Error ⊂ ValueError; PIL ⊂ OSError
            raise TeamTokenError("gateway returned an image that could not be decoded") from exc
    if not tensors:
        raise TeamTokenError("generation returned no image bytes")
    shapes = {t.shape for t in tensors}
    if len(shapes) == 1:
        return torch.cat(tensors, dim=0)
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    print(f"[teamToken] {len(tensors)} images had differing sizes; zero-padded to "
          f"{max_h}×{max_w} so none are lost — crop downstream if needed.")
    return torch.cat([_pad_to(t, max_h, max_w) for t in tensors], dim=0)


# -- polling with progress -------------------------------------------------
def _interrupt_check() -> None:
    """Let the ComfyUI Cancel button break a long poll loop."""
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _progress_bar(total: int):
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(total)
    except ImportError:
        return None


def poll_job(
    client: TeamTokenClient,
    poll_path: str,
    *,
    interval: float = 3.0,
    max_seconds: float = 900.0,
    estimate_seconds: float = 120.0,
    on_status: Callable[[str], None] | None = None,
) -> dict:
    """Poll a media job to a terminal state, driving the node progress bar.

    Video is billed by the minute, so a blocking wait would freeze the graph;
    instead we poll ``GET /v1/videos/{id}`` (or the images jobs endpoint) and
    advance an indeterminate bar against ``estimate_seconds`` so the user sees
    motion even though the true finish time is unknown. Returns the final body;
    raises TeamTokenError on ``failed`` or timeout.
    """
    pbar = _progress_bar(1000)
    started = time.monotonic()
    last_status = ""
    net_errors = 0
    while True:
        _interrupt_check()
        # A single connection reset / DNS blip / read timeout mid-poll must NOT
        # abandon a job that is still running and being billed server-side — the
        # user would get no result yet pay, and a re-queue would pay again. So
        # tolerate a bounded run of transient network errors (still honouring the
        # overall deadline); a real HTTP error or 'failed' status is a
        # TeamTokenError and propagates unchanged.
        try:
            status_code, body = client.get_json(poll_path)
            net_errors = 0
        except requests.RequestException as exc:
            net_errors += 1
            if net_errors > _MAX_POLL_NET_ERRORS or time.monotonic() - started > max_seconds:
                raise TeamTokenError(
                    f"lost contact with the gateway while polling after {net_errors} network "
                    f"error(s) — the job may still be running; poll it later with its id") from exc
            time.sleep(interval)
            continue
        status = str(body.get("status", "")).lower()
        if status and status != last_status:
            last_status = status
            if on_status:
                on_status(status)
        if status == "completed":
            if pbar:
                pbar.update_absolute(1000, 1000)
            return body
        if status == "failed":
            err = body.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else None
            code = err.get("code") if isinstance(err, dict) else None
            raise TeamTokenError(msg or "generation failed", status=status_code, code=code)
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            raise TeamTokenError(
                f"job still '{status or 'processing'}' after {int(elapsed)}s — "
                f"poll it later with the job id: {body.get('id')}")
        if pbar:
            # Approach but never reach 100% until 'completed' actually arrives.
            frac = min(0.99, elapsed / max(estimate_seconds, 1.0))
            pbar.update_absolute(int(frac * 1000), 1000)
        time.sleep(interval)


def build_media_params(
    prompt: str,
    *,
    aspect_ratio: str | None = None,
    resolution: str | None = None,
    n: int | None = None,
    duration: int | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Assemble a generation body, omitting fields left at their neutral value.

    The gateway forwards unknown keys straight to the provider, so we only send
    what the user actually set — an empty aspect_ratio or a zero seed must not
    override a provider default.
    """
    body: dict[str, Any] = {"prompt": prompt}
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if resolution:
        body["resolution"] = resolution
    if n and n > 1:
        body["n"] = n
    if duration and duration > 0:
        body["duration"] = duration
    if seed:
        body["seed"] = seed
    if extra:
        body.update({k: v for k, v in extra.items() if v not in (None, "", [])})
    return body
