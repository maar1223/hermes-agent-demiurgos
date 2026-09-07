"""Session-affinity header (``x-opencode-session``) on OpenCode requests.

Covers the v0.20.0 port of upstream commit 139396995a:

* the main agent turn sends a STABLE per-session ``x-opencode-session`` header
  on every transport / api_mode (via ``chat_completion_helpers.build_api_kwargs``);
* auxiliary calls (``auxiliary_client._build_call_kwargs``) share the same
  per-session key as the owning main turn;
* non-OpenCode targets get NO ``x-opencode-session`` header.
"""

from __future__ import annotations

import pytest

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from agent.opencode_affinity import (
    OPENCODE_SESSION_HEADER,
    is_opencode_target,
    opencode_session_headers,
)
from run_agent import AIAgent

_MSGS = [{"role": "user", "content": "hi"}]
_SESSION = "sess-affinity-1"


def _agent(provider, model, base_url, api_mode=None):
    agent = AIAgent(
        api_key="test-key",
        base_url=base_url,
        model=model,
        provider=provider,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_id=_SESSION,
    )
    if api_mode:
        agent.api_mode = api_mode
        agent._transport = None
        agent._anthropic_base_url = base_url
    return agent


# (provider, model, base_url, api_mode) for the main-turn transports.
_TRANSPORT_CASES = [
    pytest.param("opencode-go", "qwen3-coder-30b", "https://opencode.ai", "chat_completions",
                 id="opencode-go-chat_completions"),
    pytest.param("opencode-go", "qwen3-coder-30b", "https://opencode.ai", "codex_responses",
                 id="opencode-go-codex_responses"),
    pytest.param("opencode-go", "qwen3-coder-30b", "https://opencode.ai", "anthropic_messages",
                 id="opencode-go-anthropic_messages"),
    pytest.param("opencode-free", "qwen3-coder-30b", "https://custom.opencode.example", "chat_completions",
                 id="opencode-free-custom-family"),
    pytest.param("custom", "some-model", "https://opencode.ai", "chat_completions",
                 id="custom-with-opencode-url"),
]


@pytest.mark.parametrize("provider,model,base_url,api_mode", _TRANSPORT_CASES)
def test_main_turn_sends_stable_session_header_on_every_transport(
    provider, model, base_url, api_mode
):
    agent = _agent(provider, model, base_url, api_mode=api_mode)
    kwargs = build_api_kwargs(agent, _MSGS)
    headers = kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER) == _SESSION


def test_main_turn_header_is_stable_across_calls():
    agent = _agent("opencode-go", "qwen3-coder-30b", "https://opencode.ai",
                   api_mode="chat_completions")
    first = (build_api_kwargs(agent, _MSGS).get("extra_headers") or {}).get(
        OPENCODE_SESSION_HEADER
    )
    second = (build_api_kwargs(agent, _MSGS).get("extra_headers") or {}).get(
        OPENCODE_SESSION_HEADER
    )
    assert first == _SESSION
    assert second == _SESSION


def test_auxiliary_calls_share_the_main_turn_session_key():
    provider = "opencode-go"
    model = "qwen3-coder-30b"
    base_url = "https://opencode.ai"
    # Publish the owning main-turn runtime so the aux client can read the
    # session_id. 2nd positional param is ``model`` (keyword args preferred).
    token = aux.set_runtime_main(
        provider=provider,
        model=model,
        base_url=base_url,
        session_id=_SESSION,
    )
    try:
        kwargs = aux._build_call_kwargs(
            provider=provider,
            model=model,
            messages=_MSGS,
            base_url=base_url,
        )
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)
    headers = kwargs.get("extra_headers") or {}
    assert headers.get(OPENCODE_SESSION_HEADER) == _SESSION


def test_non_opencode_target_gets_no_header():
    agent = _agent("anthropic", "claude-sonnet-4-5", "https://api.anthropic.com",
                   api_mode="chat_completions")
    kwargs = build_api_kwargs(agent, _MSGS)
    headers = kwargs.get("extra_headers") or {}
    assert OPENCODE_SESSION_HEADER not in headers


@pytest.mark.parametrize(
    "provider,base_url,expected",
    [
        ("opencode-go", "https://opencode.ai", True),
        ("opencode-zen", None, True),
        ("opencode-free", "https://anything.example", True),
        ("custom", "https://opencode.ai", True),
        ("anthropic", "https://api.anthropic.com", False),
        (None, None, False),
        ("custom", None, False),
    ],
)
def test_is_opencode_target_detection(provider, base_url, expected):
    assert is_opencode_target(provider, base_url) is expected


def test_opencode_session_headers_empty_without_key():
    # No ambient conversation context and no explicit session -> empty.
    assert opencode_session_headers("opencode-go", "https://opencode.ai", None) == {}