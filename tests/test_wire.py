"""Offline wire-layer tests for ComfyUI-teamToken.

Runs WITHOUT a ComfyUI install and without touching the network: torch and the
ComfyUI runtime (``comfy.utils``, ``comfy.model_management``, ``folder_paths``)
are stubbed in ``sys.modules`` before the package is imported, while ``requests``,
``numpy`` and ``Pillow`` are the real libraries. The HTTP transport is replaced
with an in-memory fake, so no request ever leaves the process.

What this pins down (the parts that would silently rot):
  - IMAGE tensor round-trip: BHWC float32 0..1, batch axis, size-mismatch padding
  - API-key precedence (node input > env > settings) and default-profile-only read
  - catalog live/empty/offline fallback (dropdowns never empty)
  - client status handling (200/202/error envelope/no-key/non-JSON)
  - the key-safety guard (plaintext / untrusted host refused; allow-list env)
  - progress polling (completed / failed / timeout / Cancel-interrupt)
  - the node contract of all four nodes (inputs, outputs, mappings) — a
    regression here breaks users' saved workflows

Run it two ways (both print a pass/fail summary):
    python tests/test_wire.py
    pytest -q tests/test_wire.py

`.comfyignore` excludes ``tests/`` from the published archive, so this file is
committed for CI/dev but never shipped to the registry.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types

# --------------------------------------------------------------------------
# Stub the runtime ComfyUI provides, BEFORE importing the package.
# --------------------------------------------------------------------------
import numpy as np  # real
from PIL import Image  # real


class _FakeTensor:
    """A stand-in for a torch IMAGE tensor, backed by a numpy array.

    ComfyUI IMAGE tensors are just NHWC float arrays; the wire layer only needs
    indexing, shape/dtype, slice-assignment and ``.detach().cpu().numpy()``.
    """

    def __init__(self, arr):
        self._a = np.asarray(arr)

    @property
    def shape(self):
        return tuple(self._a.shape)

    @property
    def dtype(self):
        return self._a.dtype

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def __getitem__(self, idx):
        return _FakeTensor(self._a[idx])

    def __setitem__(self, idx, val):
        self._a[idx] = val._a if isinstance(val, _FakeTensor) else val


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.Tensor = _FakeTensor
    torch.float32 = np.float32
    torch.from_numpy = lambda a: _FakeTensor(a)
    torch.zeros = lambda shape, dtype=None: _FakeTensor(np.zeros(shape, dtype=dtype))
    torch.cat = lambda tensors, dim=0: _FakeTensor(
        np.concatenate([t._a for t in tensors], axis=dim))
    sys.modules["torch"] = torch

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    class _ProgressBar:
        def __init__(self, total):
            self.total = total
            self.calls = []

        def update_absolute(self, value, total=None):
            self.calls.append((value, total))

    utils.ProgressBar = _ProgressBar

    mm = types.ModuleType("comfy.model_management")
    mm.interrupt = False

    def _throw():
        if mm.interrupt:
            raise RuntimeError("ComfyUI processing interrupted")

    mm.throw_exception_if_processing_interrupted = _throw

    comfy.utils = utils
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.model_management"] = mm

    fp = types.ModuleType("folder_paths")
    fp._user_dir = None
    fp._out = fp._in = fp._tmp = None
    fp.get_user_directory = lambda: fp._user_dir
    fp.get_output_directory = lambda: fp._out
    fp.get_input_directory = lambda: fp._in
    fp.get_temp_directory = lambda: fp._tmp
    sys.modules["folder_paths"] = fp
    return sys.modules["comfy.model_management"], fp


_MM, _FP = _install_stubs()

# --------------------------------------------------------------------------
# Load the package as one tree rooted at the real __init__.py.
# --------------------------------------------------------------------------
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "comfyui_teamtoken", os.path.join(_PKG_ROOT, "__init__.py"),
    submodule_search_locations=[_PKG_ROOT])
_root = importlib.util.module_from_spec(_spec)
sys.modules["comfyui_teamtoken"] = _root
_spec.loader.exec_module(_root)

client = importlib.import_module("comfyui_teamtoken.teamtoken.client")
common = importlib.import_module("comfyui_teamtoken.teamtoken.common")
catalog = importlib.import_module("comfyui_teamtoken.teamtoken.catalog")
settings = importlib.import_module("comfyui_teamtoken.teamtoken.settings")
TeamTokenError = client.TeamTokenError


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
class _Resp:
    def __init__(self, status=200, json_body=None, text="", bad_json=False, chunks=None):
        self.status_code = status
        self._json = json_body
        self.text = text
        self._bad_json = bad_json
        self._chunks = chunks or []

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._bad_json:
            raise ValueError("no json")
        return self._json

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeRequests:
    """Records the last call and returns a preset response."""

    def __init__(self, resp=None, err=None):
        self.resp = resp
        self.err = err
        self.last = None

    def post(self, url, json=None, headers=None, timeout=None):
        self.last = {"method": "POST", "url": url, "json": json, "headers": headers}
        if self.err:
            raise self.err
        return self.resp

    def get(self, url, headers=None, timeout=None, stream=False):
        self.last = {"method": "GET", "url": url, "headers": headers, "stream": stream}
        if self.err:
            raise self.err
        return self.resp


class _fake_requests_ctx:
    def __init__(self, resp=None, err=None):
        self.fake = _FakeRequests(resp, err)

    def __enter__(self):
        self._orig = client.requests
        client.requests = self.fake
        return self.fake

    def __exit__(self, *a):
        client.requests = self._orig
        return False


class _env_ctx:
    """Set env vars for the duration of a block, restoring afterwards."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


class _FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s  # advance virtual time instead of really sleeping


class _FakePollClient:
    """A client whose get_json walks a scripted list of (status_code, body)."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def get_json(self, path):
        step = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return step


class _FlakyPollClient:
    """get_json raises a transient network error the first ``fails`` times."""

    def __init__(self, fails, then):
        self.fails = fails
        self.then = then
        self.calls = 0

    def get_json(self, path):
        self.calls += 1
        if self.calls <= self.fails:
            raise common.requests.ConnectionError("transient")
        return self.then


def _blank_env():
    # Neutralise any ambient teamToken env so precedence tests are deterministic.
    return _env_ctx(TEAMTOKEN_API_KEY=None, TEAMTOKEN_KEY=None,
                    TEAMTOKEN_SERVER_URL=None, TEAMTOKEN_COMFY_USER=None,
                    TEAMTOKEN_ALLOWED_HOSTS=None)


def _png_tensor(h, w, value=0.5):
    return _FakeTensor(np.full((1, h, w, 3), value, dtype=np.float32))


def _b64_png(h, w, color=(10, 20, 30)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _reset_catalog():
    catalog._cache = None
    catalog._refreshing = False


# --------------------------------------------------------------------------
# IMAGE tensor bridge
# --------------------------------------------------------------------------
def test_tensor_to_data_url_is_png_data_url():
    url = common.tensor_to_data_url(_png_tensor(4, 6))
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert Image.open(io.BytesIO(raw)).size == (6, 4)  # PIL is (W, H)


def test_b64_roundtrip_shape_and_range():
    t = common._b64_to_tensor(_b64_png(4, 6))
    assert t.shape == (1, 4, 6, 3)
    arr = t.numpy()
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def test_tensors_to_data_urls_covers_batch():
    batch = _FakeTensor(np.full((3, 4, 4, 3), 0.5, dtype=np.float32))
    assert len(common.tensors_to_data_urls(batch)) == 3


def test_images_payload_equal_sizes_fast_path():
    data = [{"b64_json": _b64_png(4, 4)}, {"b64_json": _b64_png(4, 4)}]
    out = common.images_payload_to_tensor(data)
    assert out.shape == (2, 4, 4, 3)


def test_images_payload_pads_mismatched_sizes_without_loss():
    data = [{"b64_json": _b64_png(4, 4)}, {"b64_json": _b64_png(8, 6)}]
    out = common.images_payload_to_tensor(data)
    # Neither frame dropped; both padded onto the batch-max canvas.
    assert out.shape == (2, 8, 6, 3)


def test_images_payload_empty_raises():
    try:
        common.images_payload_to_tensor([{"b64_json": ""}])
        assert False, "expected TeamTokenError"
    except TeamTokenError:
        pass


def test_empty_image_batch_refused():
    # A 0-frame IMAGE ([0,H,W,C]) can arrive from an upstream filter/batch node;
    # callers index urls[0], so it must be a bounded error, not an IndexError.
    empty = _FakeTensor(np.zeros((0, 4, 4, 3), dtype=np.float32))
    try:
        common.tensors_to_data_urls(empty)
        assert False, "expected TeamTokenError for an empty batch"
    except TeamTokenError:
        pass


def test_images_payload_malformed_is_bounded():
    # Undecodable/невалидные bytes must surface as TeamTokenError, not a raw
    # PIL/binascii error after an already-paid generation.
    not_an_image = base64.b64encode(b"not-an-image").decode("ascii")
    try:
        common.images_payload_to_tensor([{"b64_json": not_an_image}])
        assert False, "expected TeamTokenError for undecodable image bytes"
    except TeamTokenError:
        pass
    # Non-dict entries must be skipped (no AttributeError on .get); with nothing
    # valid left it degrades to the bounded 'no image bytes'.
    try:
        common.images_payload_to_tensor(["a-bare-string", None])
        assert False, "expected TeamTokenError for non-dict entries"
    except TeamTokenError:
        pass


# --------------------------------------------------------------------------
# build_media_params
# --------------------------------------------------------------------------
def test_build_media_params_omits_neutral_values():
    body = common.build_media_params("hi", aspect_ratio="", resolution=None,
                                     n=1, duration=0, seed=0)
    assert body == {"prompt": "hi"}  # neutral values must not override provider defaults


def test_build_media_params_includes_set_values():
    body = common.build_media_params("hi", aspect_ratio="16:9", n=3, duration=6,
                                     seed=42, extra={"image": "x", "empty": ""})
    assert body["aspect_ratio"] == "16:9" and body["n"] == 3
    assert body["duration"] == 6 and body["seed"] == 42
    assert body["image"] == "x" and "empty" not in body  # empty extras dropped


# --------------------------------------------------------------------------
# Key / server-URL precedence
# --------------------------------------------------------------------------
def test_api_key_node_input_wins():
    with _blank_env():
        _FP._user_dir = None
        assert settings.resolve_api_key("  sk-node  ") == "sk-node"


def test_api_key_env_over_settings():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "default"))
        with open(os.path.join(d, "default", "comfy.settings.json"), "w") as fh:
            json.dump({"teamToken.apiKey": "sk-file"}, fh)
        _FP._user_dir = d
        with _blank_env(), _env_ctx(TEAMTOKEN_API_KEY="sk-env"):
            assert settings.resolve_api_key() == "sk-env"


def test_api_key_settings_file_last():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "default"))
        with open(os.path.join(d, "default", "comfy.settings.json"), "w") as fh:
            json.dump({"teamToken.apiKey": "sk-file"}, fh)
        _FP._user_dir = d
        with _blank_env():
            assert settings.resolve_api_key() == "sk-file"


def test_settings_reads_only_default_profile():
    # A sibling profile's key must never be read (multi-user balance leak).
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "alice"))
        with open(os.path.join(d, "alice", "comfy.settings.json"), "w") as fh:
            json.dump({"teamToken.apiKey": "sk-alice"}, fh)
        _FP._user_dir = d
        with _blank_env():
            assert settings.resolve_api_key() == ""  # not 'sk-alice'


def test_server_url_default_and_override():
    with _blank_env():
        _FP._user_dir = None
        assert settings.resolve_server_url() == settings.DEFAULT_SERVER_URL
        assert settings.resolve_server_url("https://gw.example/") == "https://gw.example"


# --------------------------------------------------------------------------
# Catalog: live / empty / offline
# --------------------------------------------------------------------------
def test_catalog_refresh_uses_live_when_present():
    _reset_catalog()
    live = [{"model": "m-live", "modality": "image"}]
    orig = catalog.TeamTokenClient
    try:
        catalog.TeamTokenClient = types.SimpleNamespace(
            fetch_catalog=lambda url, timeout=None: live)
        stored = catalog._refresh_now("https://api.teamtoken.store")
        assert stored == live
    finally:
        catalog.TeamTokenClient = orig
        _reset_catalog()


def test_catalog_empty_live_falls_back_to_snapshot():
    _reset_catalog()
    orig = catalog.TeamTokenClient
    try:
        catalog.TeamTokenClient = types.SimpleNamespace(
            fetch_catalog=lambda url, timeout=None: [])  # empty = miss
        stored = catalog._refresh_now("https://api.teamtoken.store")
        assert stored, "empty live catalog must degrade to the bundled snapshot"
        assert any(m.get("modality") == "image" for m in stored)
    finally:
        catalog.TeamTokenClient = orig
        _reset_catalog()


def test_catalog_fetch_error_falls_back_to_snapshot():
    _reset_catalog()
    orig = catalog.TeamTokenClient

    def boom(url, timeout=None):
        raise RuntimeError("network down")

    try:
        catalog.TeamTokenClient = types.SimpleNamespace(fetch_catalog=boom)
        stored = catalog._refresh_now("https://api.teamtoken.store")
        assert stored, "fetch failure must degrade to the bundled snapshot"
    finally:
        catalog.TeamTokenClient = orig
        _reset_catalog()


def test_catalog_dropdowns_never_empty_and_never_block():
    _reset_catalog()
    orig = catalog.TeamTokenClient
    try:
        # Even if a background refresh would fail, the immediate call returns the
        # snapshot synchronously — INPUT_TYPES never waits on the wire.
        catalog.TeamTokenClient = types.SimpleNamespace(
            fetch_catalog=lambda url, timeout=None: (_ for _ in ()).throw(RuntimeError("x")))
        imgs = catalog.image_models()
        vids = catalog.video_models()
        assert imgs and vids
        assert all(isinstance(x, str) for x in imgs + vids)
    finally:
        catalog.TeamTokenClient = orig
        _reset_catalog()


def test_catalog_models_for_filters_and_sorts():
    _reset_catalog()
    catalog._cache = (1e18, catalog.catalog_server_url(), [
        {"model": "z-img", "modality": "image"},
        {"model": "a-img", "modality": "image"},
        {"model": "v-vid", "modality": "video"},
    ])
    try:
        assert catalog.models_for("image") == ["a-img", "z-img"]
        assert catalog.models_for("video") == ["v-vid"]
    finally:
        _reset_catalog()


def test_catalog_fetch_catalog_non_list_is_empty():
    with _fake_requests_ctx(resp=_Resp(200, json_body={"not": "a list"})):
        assert client.TeamTokenClient.fetch_catalog("https://api.teamtoken.store") == []


# --------------------------------------------------------------------------
# HTTP client: statuses and errors
# --------------------------------------------------------------------------
def _client(url="https://api.teamtoken.store", key="sk-x"):
    return client.TeamTokenClient(url, key)


def test_post_200_returns_body():
    with _fake_requests_ctx(resp=_Resp(200, json_body={"data": [{"b64_json": "x"}]})):
        status, body = _client().post_json("/v1/images/generations", {"model": "m"})
        assert status == 200 and body["data"][0]["b64_json"] == "x"


def test_post_202_is_success_job():
    with _fake_requests_ctx(resp=_Resp(202, json_body={"id": "job-1"})):
        status, body = _client().post_json("/v1/videos", {"model": "m"})
        assert status == 202 and body["id"] == "job-1"


def test_error_envelope_becomes_teamtoken_error():
    body = {"error": {"message": "content filtered", "code": "GEMINI_RAI_MEDIA_FILTERED"}}
    with _fake_requests_ctx(resp=_Resp(400, json_body=body)):
        try:
            _client().post_json("/v1/images/generations", {})
            assert False, "expected TeamTokenError"
        except TeamTokenError as e:
            assert e.status == 400 and e.code == "GEMINI_RAI_MEDIA_FILTERED"
            assert "content filtered" in str(e)


def test_error_surfaces_job_id_for_recovery():
    # An ambiguous submit 5xx returns the id of a job that may still be billed
    # server-side. That id must reach the user so they reference the existing job
    # instead of re-queuing — a fresh queue would pay for a second generation.
    body = {"error": {"message": "Upstream submit failed; will reconcile", "id": "job-9"}}
    with _fake_requests_ctx(resp=_Resp(502, json_body=body)):
        try:
            _client().post_json("/v1/videos", {"model": "m"})
            assert False, "expected TeamTokenError"
        except TeamTokenError as e:
            assert e.status == 502 and "job-9" in str(e)


def test_missing_key_is_401_before_any_request():
    with _fake_requests_ctx(resp=_Resp(200, json_body={})) as fake:
        try:
            client.TeamTokenClient("https://api.teamtoken.store", "").post_json("/x", {})
            assert False, "expected TeamTokenError"
        except TeamTokenError as e:
            assert e.status == 401
        assert fake.last is None  # never hit the wire without a key


def test_non_json_success_raises():
    with _fake_requests_ctx(resp=_Resp(200, bad_json=True, text="<html>")):
        try:
            _client().post_json("/x", {})
            assert False, "expected TeamTokenError"
        except TeamTokenError:
            pass


def test_non_object_json_success_raises():
    # A 2xx whose JSON is a list/str/null must become a bounded TeamTokenError,
    # not an opaque AttributeError when a caller then does body.get(...).
    for bad in ([{"b64_json": "x"}], "done", None):
        with _fake_requests_ctx(resp=_Resp(200, json_body=bad)):
            try:
                _client().post_json("/x", {})
                assert False, "expected TeamTokenError for non-object body"
            except TeamTokenError:
                pass


def test_image_202_without_job_id_refuses_to_poll():
    # A 202 that omits the job id must fail fast, not poll /v1/images/jobs/None.
    node = _root.NODE_CLASS_MAPPINGS["TeamTokenImage"]()
    with _blank_env(), _env_ctx(TEAMTOKEN_API_KEY="sk-x"), \
            _fake_requests_ctx(resp=_Resp(202, json_body={})) as fake:
        try:
            node.generate(model="m", prompt="hello")
            assert False, "expected TeamTokenError when a 202 carries no id"
        except TeamTokenError as e:
            assert "no id" in str(e)
        # never reached the poll GET — the last call is still the submit POST
        assert fake.last["url"].endswith("/v1/images/generations")


# --------------------------------------------------------------------------
# Key-safety guard
# --------------------------------------------------------------------------
def test_plaintext_http_to_remote_host_refused():
    with _blank_env():
        try:
            _client(url="http://api.teamtoken.store")
            assert False, "expected refusal of plaintext http"
        except TeamTokenError:
            pass


def test_untrusted_host_refused():
    with _blank_env():
        try:
            _client(url="https://evil.example.com")
            assert False, "expected refusal of untrusted host"
        except TeamTokenError:
            pass


def test_trusted_https_and_localhost_allowed():
    with _blank_env():
        _client(url="https://api.teamtoken.store")  # no raise
        _client(url="http://localhost:8000")  # localhost http is allowed


def test_allowed_hosts_env_extends_and_wildcard_disables():
    with _blank_env(), _env_ctx(TEAMTOKEN_ALLOWED_HOSTS="gw.example.com"):
        _client(url="https://gw.example.com")  # no raise: allow-listed
    with _blank_env(), _env_ctx(TEAMTOKEN_ALLOWED_HOSTS="*"):
        _client(url="https://anything.example.com")  # no raise: check disabled


def test_download_rechecks_untrusted_host():
    with _blank_env():
        c = _client()
        try:
            c.download("https://evil.example.com/x.mp4", os.devnull)
            assert False, "expected download to refuse an untrusted redirect"
        except TeamTokenError:
            pass


# --------------------------------------------------------------------------
# Progress polling
# --------------------------------------------------------------------------
def test_poll_completes():
    orig = common.time
    common.time = _FakeClock()
    try:
        c = _FakePollClient([(200, {"status": "processing"}), (200, {"status": "completed", "id": "j"})])
        body = common.poll_job(c, "/v1/videos/j", interval=3.0, max_seconds=100.0)
        assert body["status"] == "completed"
    finally:
        common.time = orig


def test_poll_failed_surfaces_provider_error():
    orig = common.time
    common.time = _FakeClock()
    try:
        c = _FakePollClient([(200, {"status": "failed", "error": {"message": "boom", "code": "E"}})])
        try:
            common.poll_job(c, "/v1/videos/j")
            assert False, "expected TeamTokenError"
        except TeamTokenError as e:
            assert e.code == "E" and "boom" in str(e)
    finally:
        common.time = orig


def test_poll_times_out_with_job_id():
    orig = common.time
    common.time = _FakeClock()
    try:
        c = _FakePollClient([(200, {"status": "processing", "id": "j"})])
        try:
            common.poll_job(c, "/v1/videos/j", interval=3.0, max_seconds=9.0)
            assert False, "expected timeout"
        except TeamTokenError as e:
            assert "j" in str(e)
    finally:
        common.time = orig


def test_poll_honours_cancel_interrupt():
    orig = common.time
    common.time = _FakeClock()
    _MM.interrupt = True
    try:
        c = _FakePollClient([(200, {"status": "processing"})])
        try:
            common.poll_job(c, "/v1/videos/j")
            assert False, "Cancel should break the poll loop"
        except RuntimeError:
            pass
    finally:
        _MM.interrupt = False
        common.time = orig


def test_poll_tolerates_transient_network_errors():
    # A couple of connection resets mid-poll must NOT abandon a running (billed)
    # job — the loop rides them out and still returns the completed body.
    orig = common.time
    common.time = _FakeClock()
    try:
        c = _FlakyPollClient(2, (200, {"status": "completed", "id": "j"}))
        body = common.poll_job(c, "/v1/videos/j", interval=3.0, max_seconds=100.0)
        assert body["status"] == "completed" and c.calls == 3
    finally:
        common.time = orig


def test_poll_gives_up_after_sustained_outage():
    # A truly dead gateway must not hang forever: after the bounded run of
    # network errors the loop surfaces a TeamTokenError naming the failure mode.
    orig = common.time
    common.time = _FakeClock()
    try:
        c = _FlakyPollClient(999, (200, {"status": "completed"}))
        try:
            common.poll_job(c, "/v1/videos/j", interval=3.0, max_seconds=100.0)
            assert False, "expected TeamTokenError after a sustained outage"
        except TeamTokenError as e:
            assert "network" in str(e)
    finally:
        common.time = orig


# --------------------------------------------------------------------------
# Node contract — a regression here breaks users' saved workflows.
# --------------------------------------------------------------------------
def test_node_mappings_and_web_dir():
    assert set(_root.NODE_CLASS_MAPPINGS) == {
        "TeamTokenImage", "TeamTokenImageEdit", "TeamTokenVideo", "TeamTokenVideoExtend"}
    assert set(_root.NODE_DISPLAY_NAME_MAPPINGS) == set(_root.NODE_CLASS_MAPPINGS)
    assert _root.WEB_DIRECTORY == "./web"
    assert os.path.isdir(os.path.join(_PKG_ROOT, "web"))  # WEB_DIRECTORY exists


def test_image_node_contract():
    _reset_catalog()
    cls = _root.NODE_CLASS_MAPPINGS["TeamTokenImage"]
    spec = cls.INPUT_TYPES()
    assert set(spec["required"]) == {"model", "prompt"}
    assert set(spec["optional"]) == {"aspect_ratio", "resolution", "n", "image",
                                     "seed", "api_key", "server_url"}
    assert cls.RETURN_TYPES == ("IMAGE", "STRING")
    assert cls.RETURN_NAMES == ("images", "cost_usd")
    assert cls.FUNCTION == "generate"
    assert cls.CATEGORY == "teamToken/image"
    _reset_catalog()


def test_image_edit_node_contract():
    _reset_catalog()
    cls = _root.NODE_CLASS_MAPPINGS["TeamTokenImageEdit"]
    spec = cls.INPUT_TYPES()
    assert set(spec["required"]) == {"model", "image", "prompt"}
    assert cls.FUNCTION == "edit" and cls.RETURN_NAMES == ("images", "cost_usd")
    _reset_catalog()


def test_video_node_contract():
    _reset_catalog()
    cls = _root.NODE_CLASS_MAPPINGS["TeamTokenVideo"]
    spec = cls.INPUT_TYPES()
    assert set(spec["required"]) == {"model", "prompt"}
    assert set(spec["optional"]) == {"duration", "aspect_ratio", "image", "video",
                                     "seed", "api_key", "server_url"}
    assert len(cls.RETURN_TYPES) == 4 and cls.RETURN_NAMES[-1] == "cost_usd"
    assert "job_id" in cls.RETURN_NAMES
    assert cls.FUNCTION == "generate" and cls.CATEGORY == "teamToken/video"
    _reset_catalog()


def test_video_extend_node_contract():
    _reset_catalog()
    cls = _root.NODE_CLASS_MAPPINGS["TeamTokenVideoExtend"]
    spec = cls.INPUT_TYPES()
    assert set(spec["required"]) == {"model", "ref_video_job_id", "prompt"}
    assert cls.FUNCTION == "extend" and "job_id" in cls.RETURN_NAMES
    _reset_catalog()


def test_seed_widget_has_control_after_generate():
    # control_after_generate makes "bump the seed for a fresh paid run" one click,
    # independent of frontend seed-name heuristics.
    for name in ("TeamTokenImage", "TeamTokenImageEdit", "TeamTokenVideo", "TeamTokenVideoExtend"):
        _reset_catalog()
        opt = _root.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()["optional"]["seed"][1]
        assert opt.get("control_after_generate") is True, name
    _reset_catalog()


# --------------------------------------------------------------------------
# Minimal runner (works without pytest installed).
# --------------------------------------------------------------------------
def _main():
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001 — report, don't abort the run
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {failed} failed ({len(tests)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
