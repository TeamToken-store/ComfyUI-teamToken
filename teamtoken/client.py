"""Thin HTTP client for the teamToken media gateway.

One place owns the wire contract so the node classes stay declarative: auth
header shape, the ``{"error": {...}}`` envelope, the sync-facade vs 202 split,
and streaming the final bytes. Nothing here knows about ComfyUI tensors — that
translation lives in ``common.py``.

Contract reference: the teamToken API guide (app.teamtoken.store -> Docs).
Verified against the live gateway on 2026-08-31.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import requests  # ships with every ComfyUI install

# Hosts the API key may be attached to. The threat is a SHARED workflow: node
# widget values (including server_url) travel inside a workflow .json, and result
# URLs come from the gateway response — either could redirect the key to an
# attacker. So the key is only ever sent to a trusted host. A private gateway is
# allowed by listing its host in $TEAMTOKEN_ALLOWED_HOSTS (comma-separated), or
# "*" to disable the check — an ENV setting, which a shared workflow cannot set.
_DEFAULT_ALLOWED_HOSTS = {"api.teamtoken.store", "app.teamtoken.store"}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _allowed_hosts() -> set[str] | None:
    raw = os.environ.get("TEAMTOKEN_ALLOWED_HOSTS", "").strip()
    if raw == "*":
        return None
    extra = {h.strip().lower() for h in raw.split(",") if h.strip()}
    return _DEFAULT_ALLOWED_HOSTS | _LOCAL_HOSTS | extra


class TeamTokenError(RuntimeError):
    """A gateway error surfaced to the ComfyUI user with the provider's message.

    ``code`` carries the provider error code (e.g. GEMINI_RAI_MEDIA_FILTERED)
    when the gateway forwarded one; ``status`` is the HTTP status.
    """

    def __init__(self, message: str, status: int | None = None, code: str | None = None):
        self.status = status
        self.code = code
        prefix = f"[{code}] " if code else ""
        super().__init__(f"teamToken: {prefix}{message}")


def _envelope_message(body: Any) -> tuple[str, str | None]:
    """Pull (message, code) out of the gateway's error envelope, defensively."""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or "request failed"), err.get("code")
        if isinstance(err, str):
            return err, None
        if "message" in body:
            return str(body["message"]), body.get("code")
    return "request failed", None


def _assert_key_safe_url(url: str) -> None:
    """Refuse to attach the API key to a plaintext or untrusted destination."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https"):
        raise TeamTokenError(f"unsupported URL scheme: {parsed.scheme or '(none)'}", status=400)
    if parsed.scheme != "https" and host not in _LOCAL_HOSTS:
        raise TeamTokenError(f"refusing to send the API key over plaintext http to '{host}' — use https",
                             status=400)
    allowed = _allowed_hosts()
    if allowed is not None and host not in allowed:
        raise TeamTokenError(
            f"refusing to send the API key to untrusted host '{host}'. If this is your own "
            f"gateway, add it to $TEAMTOKEN_ALLOWED_HOSTS (comma-separated), or set it to '*'.",
            status=400)


class TeamTokenClient:
    def __init__(self, server_url: str, api_key: str, timeout: float = 300.0):
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # Validate up front: post_json/get_json all target this host with the key.
        _assert_key_safe_url(self.server_url)

    # -- headers -----------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        # Bearer is the preferred form; the gateway also accepts x-api-key.
        return {"Authorization": f"Bearer {self.api_key}"}

    def _url(self, path: str) -> str:
        return f"{self.server_url}/{path.lstrip('/')}"

    # -- catalog (public, no auth) ----------------------------------------
    @staticmethod
    def fetch_catalog(server_url: str, timeout: float = 10.0) -> list[dict]:
        """Live media-model catalog. Public endpoint — no key required.

        Reachable on both the gateway (api.) and cabinet (app.) hosts, so the
        same server URL the nodes call for generation also serves this.
        """
        url = f"{server_url.rstrip('/')}/cabinet/api/public/media-models"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    # -- requests ----------------------------------------------------------
    def post_json(self, path: str, payload: dict) -> tuple[int, dict]:
        """POST JSON, returning (status, body). 200 and 202 are both success."""
        if not self.api_key:
            raise TeamTokenError("no API key configured — set it in teamToken settings, "
                                 "the node's api_key input, or $TEAMTOKEN_API_KEY", status=401)
        headers = {**self._auth_headers(), "Content-Type": "application/json"}
        resp = requests.post(self._url(path), json=payload, headers=headers, timeout=self.timeout)
        return self._handle(resp)

    def get_json(self, path: str) -> tuple[int, dict]:
        if not self.api_key:
            raise TeamTokenError("no API key configured", status=401)
        resp = requests.get(self._url(path), headers=self._auth_headers(), timeout=self.timeout)
        return self._handle(resp)

    def _handle(self, resp: requests.Response) -> tuple[int, dict]:
        try:
            body = resp.json()
        except ValueError:
            if resp.ok:
                raise TeamTokenError("gateway returned a non-JSON success body", status=resp.status_code)
            raise TeamTokenError(resp.text[:300] or "request failed", status=resp.status_code)
        # 200 = done, 202 = still processing (a job to poll). Both are success.
        if resp.status_code in (200, 202):
            # Callers immediately do body.get(...); a JSON array/string/null on a
            # 2xx would otherwise surface as an opaque AttributeError far from the
            # wire instead of a bounded error naming the malformed response.
            if not isinstance(body, dict):
                raise TeamTokenError("gateway returned a non-object JSON body",
                                     status=resp.status_code)
            return resp.status_code, body
        message, code = _envelope_message(body)
        # An ambiguous submit (5xx "will reconcile") returns the id of a job the
        # gateway parked and MAY still finalize and charge. Surface that id so the
        # user can reference the existing job instead of re-queuing — a fresh queue
        # would create a second paid generation while the first bills in the
        # background. The id lives inside the error envelope.
        job_id = None
        if isinstance(body, dict):
            err = body.get("error")
            job_id = (err.get("id") if isinstance(err, dict) else None) or body.get("id")
        if job_id:
            message = (f"{message} (job id: {job_id} — it may still be processing and get "
                       f"billed; reference this id instead of re-running to avoid paying twice)")
        raise TeamTokenError(message, status=resp.status_code, code=code)

    # -- binary download (video bytes) ------------------------------------
    def download(self, url_or_path: str, dest_path: str) -> str:
        """Stream a media byte-stream to ``dest_path`` with the caller's key.

        Video results come back as URLs pointing at the gateway's own content
        proxy (``/v1/videos/{id}/content``), which requires the API key — the
        provider CDN host is never exposed. A relative path is joined onto the
        server URL; an absolute URL is used as-is.
        """
        url = url_or_path if url_or_path.startswith("http") else self._url(url_or_path)
        # The result URL comes from the gateway response; re-check it so a rogue
        # gateway can't redirect the key to an arbitrary host.
        _assert_key_safe_url(url)
        with requests.get(url, headers=self._auth_headers(), stream=True, timeout=self.timeout) as resp:
            if not resp.ok:
                raise TeamTokenError(f"failed to download result ({resp.status_code})",
                                     status=resp.status_code)
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        fh.write(chunk)
        return dest_path
