# ComfyUI-teamToken

**Veo, Seedance, Kling and nano-banana — video & image generation through one API key.**

These are closed models with no downloadable weights: local ComfyUI cannot run
Veo or Seedance at all. This node pack calls the [teamToken](https://app.teamtoken.store)
gateway, so you generate with them straight from your graph — no local GPU, one
key, pay per generation.

## Nodes

| Node | What it does | Endpoint |
|---|---|---|
| **teamToken Image** | Text-to-image and image-to-image | `POST /v1/images/generations` |
| **teamToken Image Edit** | Edit/restyle a required input image | `POST /v1/images/edits` |
| **teamToken Video** | Text-, image-, video-to-video and motion control | `POST /v1/videos` |
| **teamToken Video Extend** | Continue a previous teamToken video | `POST /v1/videos/extend` |

The model dropdowns are populated **live** from the catalog
(`GET /cabinet/api/public/media-models`), so new models appear without a node
update; a bundled snapshot keeps the dropdowns working offline.

## Install

**ComfyUI-Manager:** search for `ComfyUI-teamToken` and click Install.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/TeamToken-store/ComfyUI-teamToken
```
The only extra dependency is `requests` (in `requirements.txt`, which
ComfyUI-Manager installs automatically on a git-clone install; the registry
install reads the same list from `pyproject.toml`). To install it by hand:
`pip install requests`. Restart ComfyUI.

## Set your API key

Get a key at **app.teamtoken.store → Keys** (`sk-…`), then set it any of these
ways (first one found wins):

1. **Node input** — type it into a node's `api_key` field.
2. **Environment** — `export TEAMTOKEN_API_KEY=sk-…` before launching ComfyUI.
3. **Settings pane** — ComfyUI → Settings → **teamToken → API key**.

Leave the node's `api_key` and `server_url` widgets **empty** to use the settings
/ environment / default gateway (`https://api.teamtoken.store`); fill them only to
override per-node (e.g. a private gateway).

> ⚠️ **A key typed into the node's `api_key` field is saved into the workflow file**
> and travels with any workflow you share or export. Prefer the **Settings pane** or
> **`$TEAMTOKEN_API_KEY`** — both keep the key out of the workflow — and use the node
> input only for throwaway or already-shared keys.

**Multi-user ComfyUI:** the Settings pane is read from the `default` profile only
(reading every profile would let one user spend another's balance). On a
multi-user install, set the key per user via `TEAMTOKEN_API_KEY`, the node's
`api_key` input, or name the profile with `TEAMTOKEN_COMFY_USER=<name>`.

## Security — importing shared workflows

Workflow `.json` files carry widget values, so treat a downloaded workflow as
untrusted input. These nodes are hardened accordingly:

- The API key is only ever sent over **HTTPS** to a **trusted host** — the
  teamToken gateways by default. A workflow that points `server_url` at another
  host is refused, so it can't exfiltrate your key. Running your own gateway? Add
  its host to `TEAMTOKEN_ALLOWED_HOSTS` (comma-separated), or set it to `*` to
  disable the check.
- A local file passed to a video node is only read if it lives under ComfyUI's
  input/output folder, is a video type, and is ≤80 MB — so a workflow can't make
  the node upload arbitrary files from your disk.
- Downloaded results are written with a sanitized filename (no path traversal).

## Usage notes

- **Images** come back inline and appear as a normal `IMAGE` output you can pipe
  anywhere. Results live for 7 days on the server — this node returns the bytes
  immediately, so that only matters if you re-poll an old job id.
- **Video** is billed per second and takes 30s–several minutes. The node submits
  the job and **polls with a live progress bar** — the graph never freezes, and
  the ComfyUI Cancel button stops the poll. The finished MP4 is downloaded to
  your `output/` folder and returned as a `VIDEO` (or a file path on older
  ComfyUI builds), plus the `job_id` and the `cost_usd` actually charged.
- **Duration** (video): leave at `0` to use each model's own default — allowed
  values differ per model (veo 4/6/8, grok only 6, kling-2.1-10s only 10), so a
  fixed number would 400 on some models. Set a specific value only when you know
  the model accepts it.
- **Video Extend** takes the `job_id` output of a teamToken Video node — wire
  them together to continue a clip.
- **Reference inputs**: feed an `IMAGE` into Video for image-to-video, and/or a
  video URL/path into `video` for video-to-video and motion control.
- **Reproducibility / re-runs**: the `seed` widget only forces ComfyUI to re-run
  the node (media generation is server-side non-deterministic); it is not sent
  to the API.

## Cost

Every node outputs `cost_usd` — the exact amount charged for that generation.
Failed generations are never charged. Check your balance at app.teamtoken.store.

## Errors

Provider errors are surfaced with their message and code (e.g. `EMPTY_PROMPT`,
`INVALID_VIDEO_FILE`, `GEMINI_RAI_MEDIA_FILTERED`). A `402` means insufficient
balance; a `401` means the API key is missing or wrong.

---

*Not affiliated with Comfy-Org. teamToken is a commercial API gateway; using
these nodes spends credits on your teamToken account.*
