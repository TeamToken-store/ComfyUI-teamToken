"""Image nodes: teamToken Image (text/img-to-img) and teamToken Image Edit.

Both hit the same synchronous facade — the gateway polls the provider for up to
~150s and returns base64 inline, or hands back a 202 job we finish polling here.
Edit is the same call with a required input image, routed to the
``/v1/images/edits`` alias, which the gateway treats identically to
``/v1/images/generations`` (JSON body, not multipart).
"""

from __future__ import annotations

from . import catalog
from .client import TeamTokenClient, TeamTokenError
from .common import (
    ASPECT_RATIOS,
    RESOLUTIONS,
    build_media_params,
    images_payload_to_tensor,
    poll_job,
    tensors_to_data_urls,
)
from .settings import resolve_api_key, resolve_server_url

CATEGORY = "teamToken/image"


def _run_image(path: str, model, prompt, aspect_ratio, resolution, n,
               image=None, api_key="", server_url="", seed=0):
    if not prompt or not prompt.strip():
        raise TeamTokenError("prompt is required", status=400, code="EMPTY_PROMPT")
    client = TeamTokenClient(resolve_server_url(server_url), resolve_api_key(api_key))
    extra = {}
    if image is not None:
        # One reference for a single frame, a list for a batch — the gateway
        # accepts both under `images`; sending base64 keeps it self-contained.
        urls = tensors_to_data_urls(image)
        extra["images"] = urls if len(urls) > 1 else urls[0]
    # `seed` deliberately stays OUT of the payload — the media API doesn't take
    # it, and forwarding an unknown param risks a provider 400. It exists only to
    # steer ComfyUI's own input-equality cache (see the seed widget comment in
    # INPUT_TYPES): identical inputs reuse the prior result and are NOT re-billed;
    # changing the seed forces a fresh, paid generation.
    payload = {"model": model, **build_media_params(prompt, aspect_ratio=aspect_ratio,
                                                     resolution=resolution, n=n, extra=extra)}
    status, body = client.post_json(path, payload)
    if status == 202:  # generation outran the sync window — finish the job
        job_id = body.get("id")
        if not job_id:  # same guard the video path has — don't poll .../jobs/None
            raise TeamTokenError("gateway accepted the job but returned no id to poll")
        body = poll_job(client, f"/v1/images/jobs/{job_id}",
                        estimate_seconds=30.0, max_seconds=600.0,
                        on_status=lambda s: print(f"[teamToken] image {job_id}: {s}"))
    data = body.get("data") or []
    if not data and body.get("archived"):
        raise TeamTokenError("this result expired (images live 7 days) — regenerate it")
    cost = str(body.get("cost_usd", body.get("estimated_cost_usd", "")))
    return images_payload_to_tensor(data), cost


class TeamTokenImage:
    """Text-to-image and image-to-image via the teamToken gateway."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.image_models(), {"tooltip": "Image model, fetched live from the teamToken catalog"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "What to generate; with an IMAGE input it guides the edit"}),
            },
            "optional": {
                "aspect_ratio": (ASPECT_RATIOS, {"tooltip": "Output aspect ratio (default 1:1)"}),
                "resolution": (RESOLUTIONS, {"tooltip": "Output resolution tier (default 1K)"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10,
                              "tooltip": "How many images to generate in one request"}),
                "image": ("IMAGE", {"tooltip": "Optional reference image for image-to-image"}),
                # Not sent to the API — steers ComfyUI's input-equality cache so a
                # re-run only pays for a NEW generation. control_after_generate adds
                # the fixed/increment/randomize companion regardless of frontend
                # version, so bumping the seed to force a fresh image is one click.
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

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "cost_usd")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model, prompt, aspect_ratio="1:1", resolution="1K", n=1,
                 image=None, seed=0, api_key="", server_url=""):
        return _run_image("/v1/images/generations", model, prompt, aspect_ratio,
                          resolution, n, image=image, api_key=api_key,
                          server_url=server_url, seed=seed)


class TeamTokenImageEdit:
    """Edit / restyle an input image (image-to-image with a required source)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (catalog.image_models(), {"tooltip": "Image model, fetched live from the teamToken catalog"}),
                "image": ("IMAGE", {"tooltip": "Source image to edit / restyle"}),
                "prompt": ("STRING", {"multiline": True, "default": "",
                                      "tooltip": "How to edit the source image"}),
            },
            "optional": {
                "aspect_ratio": (ASPECT_RATIOS, {"tooltip": "Output aspect ratio (default 1:1)"}),
                "resolution": (RESOLUTIONS, {"tooltip": "Output resolution tier (default 1K)"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10,
                              "tooltip": "How many edited images to generate in one request"}),
                # Not sent to the API — steers ComfyUI's input-equality cache; see
                # the seed comment in TeamTokenImage.INPUT_TYPES.
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

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "cost_usd")
    FUNCTION = "edit"
    CATEGORY = CATEGORY

    def edit(self, model, image, prompt, aspect_ratio="1:1", resolution="1K", n=1,
             seed=0, api_key="", server_url=""):
        return _run_image("/v1/images/edits", model, prompt, aspect_ratio,
                          resolution, n, image=image, api_key=api_key,
                          server_url=server_url, seed=seed)
