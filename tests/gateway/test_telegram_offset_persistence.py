"""Telegram polling offset persistence (bot-capture-and-conversation, Slice 1).

The gateway must survive cold boots without losing messages: it persists the
last processed Telegram ``update_id`` to ``<hermes-home>/state/
telegram_offset.json`` (atomic write), resumes polling from that offset on
cold boot (``drop_pending_updates=False`` + PTB ``_last_update_id``
injection), skips pending history older than the TTL, and writes the offset
with a 500ms debounce / every-10-updates flush.

These tests confirm (spec R-1.1 .. R-1.4, scenarios S-1.1.x .. S-1.4.x):

  1. Offset file I/O: atomic write (tmp + rename + fsync), schema
     validation, corrupt/missing fallback to ``None``.
  2. Cold boot resume: ``connect()`` injects ``updater._last_update_id =
     persisted + 1`` (PTB 22.6 semantics — the attribute IS the getUpdates
     offset and Telegram's offset is inclusive) and starts polling with
     ``drop_pending_updates=False``; no offset → ``drop_pending_updates=
     True`` (no-regression); reconnect always preserves the queue.
  3. Fail-safe: if PTB's private ``_last_update_id`` is missing, the adapter
     keeps the current behavior instead of breaking startup.
  4. Debounce: 500ms idle flush and immediate flush every 10 updates.
  5. TTL: stale first pending update advances the offset; fresh keeps it;
     probe failure fail-safes.

Hermetic: PTB is mocked (``_ensure_telegram_mock`` + a fake
``Application.builder``), and the offset file lands under the conftest
sandboxed HERMES_HOME — never the operator's real ``~/.hermes``.
"""
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()


# ---------------------------------------------------------------------------
# Harness (mirrors tests/gateway/test_telegram_polling_progress.py)
# ---------------------------------------------------------------------------

def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _clear_offset_file() -> None:
    path = tg_mod._offset_file_path()
    path.unlink(missing_ok=True)
    path.with_suffix(".json.tmp").unlink(missing_ok=True)


class _ControlledRequest:
    """Minimal PTB request double (kept for the builder plumbing)."""

    @staticmethod
    def parse_json_payload(payload):
        return json.loads(payload.decode("utf-8", "replace"))

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def do_request(self, *args, **kwargs):
        return None


class _LifecycleBuilder:
    def __init__(self, app):
        self.app = app

    def token(self, _token):
        return self

    def request(self, _request):
        return self

    def get_updates_request(self, _request):
        return self

    def build(self):
        return self.app


def _lifecycle_app(updater=None):
    app = MagicMock()
    updater = updater if updater is not None else MagicMock()
    updater.running = True
    updater.start_polling = AsyncMock()
    updater.start_webhook = AsyncMock()
    updater.stop = AsyncMock()
    app.updater = updater
    app.bot = MagicMock()
    app.bot.delete_webhook = AsyncMock()
    app.bot.get_updates = AsyncMock(return_value=[])
    app.bot.get_me = AsyncMock(return_value=MagicMock())
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.running = True
    return app


def _configure_lifecycle_connect(monkeypatch, adapter, apps):
    builders = [_LifecycleBuilder(app) for app in apps]
    remaining = iter(builders)

    class _Application:
        @staticmethod
        def builder():
            return next(remaining)

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_mod, "Application", _Application)
    monkeypatch.setattr(tg_mod, "HTTPXRequest", _ControlledRequest)
    monkeypatch.setattr(tg_mod, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_mod, "resolve_proxy_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tg_mod.TelegramAdapter, "_acquire_platform_lock",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        tg_mod.TelegramAdapter, "_release_platform_lock", MagicMock()
    )
    monkeypatch.setattr(
        tg_mod.TelegramAdapter, "_fallback_ips", lambda self: []
    )
    monkeypatch.setattr(
        tg_mod.TelegramAdapter, "_start_post_connect_housekeeping", MagicMock()
    )
    return builders


# ---------------------------------------------------------------------------
# 1. Offset file I/O (T-1.1)
# ---------------------------------------------------------------------------

def test_write_offset_is_atomic_and_roundtrips():
    _clear_offset_file()
    path = tg_mod._offset_file_path()
    assert tg_mod.write_offset(12345, timestamp=1724793600.123) is True
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "update_id": 12345,
        "timestamp": 1724793600.123,
        "schema_version": 1,
    }
    # No .tmp leftover after the rename.
    assert not path.with_suffix(".json.tmp").exists()
    assert tg_mod.read_offset() == data


def test_write_offset_uses_hermes_home_state_dir():
    _clear_offset_file()
    from hermes_constants import get_hermes_home

    tg_mod.write_offset(1)
    expected = get_hermes_home() / "state" / "telegram_offset.json"
    assert expected.exists()
    # Conftest sandboxes HERMES_HOME, so the offset never lands in the real
    # production root.
    assert str(expected) != str(Path.home() / ".hermes" / "state" / "telegram_offset.json")


def test_read_offset_missing_file_returns_none():
    _clear_offset_file()
    assert tg_mod.read_offset() is None


def test_read_offset_corrupt_json_returns_none():
    path = tg_mod._offset_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json !!", encoding="utf-8")
    assert tg_mod.read_offset() is None


@pytest.mark.parametrize(
    "payload",
    [
        {"update_id": "abc", "timestamp": 1.0, "schema_version": 1},
        {"update_id": 0, "timestamp": 1.0, "schema_version": 1},
        {"update_id": -5, "timestamp": 1.0, "schema_version": 1},
        {"update_id": 5, "timestamp": 1.0},  # missing schema_version
        {"update_id": 5, "timestamp": 1.0, "schema_version": 2},
        {"update_id": 5, "schema_version": 1},  # missing timestamp
        [1, 2, 3],  # not a dict
        "telegram_offset",
    ],
)
def test_read_offset_rejects_invalid_schema(payload):
    path = tg_mod._offset_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert tg_mod.read_offset() is None


# ---------------------------------------------------------------------------
# 2. Debounce + process_update hook (T-1.3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debounced_flush_writes_after_idle(monkeypatch):
    monkeypatch.setattr(tg_mod, "_OFFSET_DEBOUNCE_SECONDS", 0.05)
    _clear_offset_file()
    adapter = _make_adapter()
    adapter._on_update_processed(SimpleNamespace(update_id=7))
    await asyncio.sleep(0.15)
    data = tg_mod.read_offset()
    assert data is not None and data["update_id"] == 7
    await asyncio.sleep(0)  # let the done callback consume the task result
    assert adapter._offset_debounce_task.done()


@pytest.mark.asyncio
async def test_flush_is_immediate_every_ten_updates():
    _clear_offset_file()
    adapter = _make_adapter()
    for i in range(1, 11):
        adapter._on_update_processed(SimpleNamespace(update_id=i))
    await asyncio.sleep(0.05)
    data = tg_mod.read_offset()
    assert data is not None and data["update_id"] == 10


@pytest.mark.asyncio
async def test_process_update_hook_tracks_last_processed():
    adapter = _make_adapter()
    processed = []

    async def fake_process_update(update):
        processed.append(update)

    app = SimpleNamespace(process_update=fake_process_update)
    adapter._app = app
    adapter._install_process_update_offset_hook()
    assert adapter._offset_hook_app is app
    # Re-install is idempotent for the same app instance.
    adapter._install_process_update_offset_hook()

    update = SimpleNamespace(update_id=42)
    await app.process_update(update)
    assert processed == [update]
    assert adapter._offset_last_processed_update_id == 42

    # Cleanup: cancel the armed debounce write so no task leaks.
    task = adapter._offset_debounce_task
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# ---------------------------------------------------------------------------
# 3. TTL (T-1.4)
# ---------------------------------------------------------------------------

def test_update_is_stale():
    now = datetime.now(timezone.utc)
    assert not TelegramAdapter._update_is_stale(
        SimpleNamespace(date=now)
    )
    assert not TelegramAdapter._update_is_stale(
        SimpleNamespace(date=now - timedelta(hours=2))
    )
    assert TelegramAdapter._update_is_stale(
        SimpleNamespace(date=now - timedelta(days=10))
    )
    # Naive datetimes are treated as UTC.
    assert TelegramAdapter._update_is_stale(
        SimpleNamespace(date=datetime.now() - timedelta(days=10))
    )
    # Unix-epoch timestamps.
    assert TelegramAdapter._update_is_stale(
        SimpleNamespace(date=time.time() - 10 * 86400)
    )
    assert not TelegramAdapter._update_is_stale(SimpleNamespace(date=None))
    assert not TelegramAdapter._update_is_stale(SimpleNamespace())


@pytest.mark.asyncio
async def test_skip_stale_pending_advances_offset_past_backlog():
    adapter = _make_adapter()
    stale = SimpleNamespace(
        update_id=100,
        date=datetime.now(timezone.utc) - timedelta(days=10),
    )
    latest = SimpleNamespace(
        update_id=1000, date=datetime.now(timezone.utc)
    )
    bot = MagicMock()
    bot.get_updates = AsyncMock(side_effect=[[stale], [latest]])
    adapter._app = SimpleNamespace(bot=bot)

    result = await adapter._skip_stale_pending_updates(100)
    assert result == 1001  # latest.update_id + 1
    assert bot.get_updates.await_args_list[0].kwargs["offset"] == 100
    assert bot.get_updates.await_args_list[1].kwargs["offset"] == -1


@pytest.mark.asyncio
async def test_skip_stale_pending_fresh_update_keeps_offset():
    adapter = _make_adapter()
    fresh = SimpleNamespace(
        update_id=100, date=datetime.now(timezone.utc)
    )
    bot = MagicMock()
    bot.get_updates = AsyncMock(return_value=[fresh])
    adapter._app = SimpleNamespace(bot=bot)

    result = await adapter._skip_stale_pending_updates(100)
    assert result == 100
    assert bot.get_updates.await_count == 1  # no second probe


@pytest.mark.asyncio
async def test_skip_stale_pending_probe_error_fail_safe():
    adapter = _make_adapter()
    bot = MagicMock()
    bot.get_updates = AsyncMock(side_effect=OSError("probe boom"))
    adapter._app = SimpleNamespace(bot=bot)

    assert await adapter._skip_stale_pending_updates(100) == 100


# ---------------------------------------------------------------------------
# 4. connect() integration: cold boot, reconnect, fail-safe (T-1.2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_cold_boot_resumes_from_persisted_offset(monkeypatch):
    _clear_offset_file()
    assert tg_mod.write_offset(12345) is True
    adapter = _make_adapter()
    app = _lifecycle_app()
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    app.updater.start_polling = AsyncMock(side_effect=start_polling_with_progress)

    try:
        assert await adapter.connect() is True
    finally:
        await adapter.disconnect()

    app.updater.start_polling.assert_awaited_once()
    assert app.updater.start_polling.call_args.kwargs[
        "drop_pending_updates"
    ] is False
    # PTB 22.6: _last_update_id IS the getUpdates offset (inclusive), so the
    # persisted last-PROCESSED id resumes as persisted + 1.
    assert app.updater._last_update_id == 12346
    assert adapter._offset_hook_app is app


@pytest.mark.asyncio
async def test_connect_cold_boot_without_offset_preserves_queue_and_installs_hook(monkeypatch):
    # No offset file → the pending queue must STILL be preserved (messages the
    # owner sent while the gateway was down must not be silently lost), and the
    # process_update hook MUST be installed so the offset gets written after the
    # first processed update — this is what makes a later cold boot resume.
    # Previously the hook was gated on an existing offset, so the file was never
    # created and every cold boot dropped the queue (chicken-and-egg).
    _clear_offset_file()
    adapter = _make_adapter()
    app = _lifecycle_app()
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    app.updater.start_polling = AsyncMock(side_effect=start_polling_with_progress)

    try:
        assert await adapter.connect() is True
    finally:
        await adapter.disconnect()

    assert app.updater.start_polling.call_args.kwargs[
        "drop_pending_updates"
    ] is False
    # No resume injection when there is nothing to resume from: the
    # attribute was never set to an int (MagicMock auto-creates a child mock).
    assert not isinstance(app.updater._last_update_id, int)
    # The hook is installed so the offset file gets written after the first
    # processed update (chicken-and-egg fix).
    assert adapter._offset_hook_app is app


@pytest.mark.asyncio
async def test_connect_reconnect_preserves_queue_with_offset(monkeypatch):
    _clear_offset_file()
    assert tg_mod.write_offset(12345) is True
    adapter = _make_adapter()
    app = _lifecycle_app()
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    try:
        assert await adapter.connect(is_reconnect=True) is True
    finally:
        await adapter.disconnect()

    app.updater.start_polling.assert_awaited_once()
    # Reconnect always preserves the queue (existing behavior).
    assert app.updater.start_polling.call_args.kwargs[
        "drop_pending_updates"
    ] is False
    # The persisted offset is still injected so the fresh app never
    # re-fetches already-processed history from offset 0.
    assert app.updater._last_update_id == 12346


class _NoPrivateOffsetUpdater:
    """PTB Updater double WITHOUT the private _last_update_id attribute."""

    def __init__(self):
        self.running = True
        self.start_polling = AsyncMock()
        self.stop = AsyncMock()


@pytest.mark.asyncio
async def test_connect_fail_safe_when_last_update_id_missing(monkeypatch):
    # Offset exists, but the updater lacks PTB's private attribute → the
    # resume point cannot be injected, but the adapter must NEVER break
    # startup and must still preserve the pending queue (drop_pending False)
    # rather than dropping messages the owner sent while offline.
    _clear_offset_file()
    assert tg_mod.write_offset(12345) is True
    adapter = _make_adapter()
    app = _lifecycle_app(updater=_NoPrivateOffsetUpdater())
    _configure_lifecycle_connect(monkeypatch, adapter, [app])
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)

    async def start_polling_with_progress(**_kwargs):
        adapter._record_polling_progress(adapter._polling_generation)

    app.updater.start_polling = AsyncMock(side_effect=start_polling_with_progress)

    try:
        assert await adapter.connect() is True
    finally:
        await adapter.disconnect()

    assert app.updater.start_polling.call_args.kwargs[
        "drop_pending_updates"
    ] is False
    # The tracking hook is still installed so the offset keeps getting written.
    assert adapter._offset_hook_app is app