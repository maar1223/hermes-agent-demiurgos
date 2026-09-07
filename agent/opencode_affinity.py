"""``x-opencode-session`` — OpenCode relay session-affinity header.

The OpenCode Go relay (opencode.ai) REQUIRES an ``x-opencode-session`` header
on every request so its backend can route (and cache) a conversation against a
stable per-session id. Without it the gateway fails the call.

Hermes therefore sends a stable per-session value on every OpenCode request —
both the main agent turn (on every transport / api_mode) and the auxiliary
client calls (compression / titles / vision / MoA / etc.) — so OpenCode's
backend affinity keys all of a conversation's traffic the same way.

The header value is normalized from the ambient conversation id (the
session-lineage ROOT id published via ``agent.portal_tags``), falling back to
the explicit ``session_id`` when no ambient context is set. ``None`` yields no
header (non-OpenCode targets and no-session paths).
"""

from __future__ import annotations

from typing import Any, Optional

OPENCODE_SESSION_HEADER = "x-opencode-session"

# v0.20.0 fork: ``hermes_cli.models`` has the ``opencode-zen``/``opencode-go``
# providers and ``normalize_opencode_base_url``, but NOT upstream's
# ``opencode_provider_family`` helper. ``agent/anthropic_endpoints.py`` (and its
# ``_is_opencode_endpoint``) does not exist here either. Detection is therefore
# done from the provider name and the base_url directly.
_OPENCODE_PROVIDER_NAMES = {"opencode-zen", "opencode-go"}


def is_opencode_target(provider: Optional[str], base_url: Optional[str]) -> bool:
    """True when *provider* or *base_url* addresses the OpenCode relay.

    Matches when the provider is one of the built-in OpenCode families
    (``opencode-zen`` / ``opencode-go``), any custom ``opencode-<family>-*``
    provider, or the base_url host is an opencode.ai endpoint. Handles
    ``None`` safely.
    """
    if provider:
        p = str(provider).strip().lower()
        if p in _OPENCODE_PROVIDER_NAMES or p.startswith("opencode-"):
            return True
    if base_url:
        try:
            if "opencode.ai" in str(base_url).lower():
                return True
        except Exception:
            return False
    return False


def opencode_session_headers(provider, base_url, session_id=None):
    """Return ``{OPENCODE_SESSION_HEADER: <key>}`` for an OpenCode target.

    The key is the stable per-conversation cache scope: the ambient
    conversation context published by the agent loop (``agent.portal_tags``),
    falling back to the explicit ``session_id``. Empty on non-OpenCode targets
    or when no session key is available.
    """
    if not is_opencode_target(provider, base_url):
        return {}
    try:
        from agent.portal_tags import get_conversation_context
        from agent.transports.codex import _cache_scope_from_session_id

        key = _cache_scope_from_session_id(get_conversation_context() or session_id)
    except Exception:
        key = str(session_id or "")
    return {OPENCODE_SESSION_HEADER: key} if key else {}


def merge_opencode_session_headers(kwargs, provider, base_url, session_id=None):
    """Merge the OpenCode session-affinity header onto *kwargs*.

    Injects ``extra_headers['x-opencode-session']`` (never overwriting a
    caller-supplied value) when the target is an OpenCode relay. Returns the
    (possibly unchanged) kwargs dict.
    """
    headers = opencode_session_headers(provider, base_url, session_id)
    if headers:
        existing = kwargs.get("extra_headers")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for key, value in headers.items():
            merged.setdefault(key, value)
        kwargs["extra_headers"] = merged
    return kwargs