import asyncio

import pytest

from unittest.mock import MagicMock, patch

from src import llm_client, llm_config, model_swap


# ##################################################################
# helpers
# build a minimal OpenAI-compatible chat-completion response payload


def _resp(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


class _FakeHttpResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient that returns scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None):
        self.calls.append({"url": url, "json": json})
        nxt = self._responses.pop(0)
        if isinstance(nxt, _FakeHttpResponse):
            return nxt
        return _FakeHttpResponse(nxt)


@pytest.fixture(autouse=True)
def _reset_swap_state():
    # ensure a clean module state for model_swap between tests
    model_swap._reset_for_tests()
    # default: keep models loaded so ensure_model is a no-op unless a test
    # overrides this explicitly
    llm_config.configure(keep_models_loaded=True, model_swap_url="")
    yield
    model_swap._reset_for_tests()
    llm_config.configure(keep_models_loaded=False, model_swap_url="")


# ##################################################################
# tier resolution via convenience wrappers


def test_query_small_resolves_small_model():
    llm_config.configure(small_model="SMALL-X", large_model="LARGE-X")
    fake = _FakeAsyncClient([_resp("hello")])

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        out = asyncio.run(llm_client.query_small("hi"))

    assert out == "hello"
    assert fake.calls[0]["json"]["model"] == "SMALL-X"


def test_query_large_resolves_large_model():
    llm_config.configure(small_model="SMALL-X", large_model="LARGE-X")
    fake = _FakeAsyncClient([_resp("world")])

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        out = asyncio.run(llm_client.query_large("hi"))

    assert out == "world"
    assert fake.calls[0]["json"]["model"] == "LARGE-X"


def test_query_llm_default_tier_is_small():
    llm_config.configure(small_model="SMALL-X", large_model="LARGE-X")
    fake = _FakeAsyncClient([_resp("ok")])

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        asyncio.run(llm_client.query_llm("hi"))

    assert fake.calls[0]["json"]["model"] == "SMALL-X"


# ##################################################################
# continue-on-truncation


def test_continue_on_truncation_concatenates():
    llm_config.configure(small_model="SMALL-X")
    fake = _FakeAsyncClient(
        [
            _resp("part-one-", finish_reason="length"),
            _resp("part-two", finish_reason="stop"),
        ]
    )

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        out = asyncio.run(llm_client.query_llm("hi"))

    assert out == "part-one-part-two"
    assert len(fake.calls) == 2
    # second call should include the partial assistant turn plus a continue user turn
    second_messages = fake.calls[1]["json"]["messages"]
    assert second_messages[0] == {"role": "user", "content": "hi"}
    assert second_messages[1] == {"role": "assistant", "content": "part-one-"}
    assert second_messages[2] == {"role": "user", "content": "continue"}


def test_custom_continuation_prompt():
    llm_config.configure(small_model="SMALL-X")
    fake = _FakeAsyncClient(
        [
            _resp("a", finish_reason="length"),
            _resp("b", finish_reason="stop"),
        ]
    )

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        asyncio.run(llm_client.query_llm("hi", continuation_prompt="resume jsonl"))

    assert fake.calls[1]["json"]["messages"][-1] == {
        "role": "user",
        "content": "resume jsonl",
    }


def test_max_continuations_overflow_raises():
    llm_config.configure(small_model="SMALL-X")
    # always returns length -> should raise after max_continuations + 1 calls
    fake = _FakeAsyncClient(
        [_resp("x", finish_reason="length") for _ in range(10)]
    )

    with patch("src.llm_client.httpx.AsyncClient", return_value=fake):
        with pytest.raises(RuntimeError, match="max_continuations"):
            asyncio.run(llm_client.query_llm("hi", max_continuations=2))

    # max_continuations=2 means 1 initial + 2 retries = 3 total calls
    assert len(fake.calls) == 3


# ##################################################################
# ensure_model no-op behaviour


def test_ensure_model_noop_when_keep_models_loaded():
    llm_config.configure(keep_models_loaded=True, model_swap_url="http://swap")

    fake_client_factory = MagicMock()
    with patch("src.model_swap.httpx.AsyncClient", fake_client_factory):
        asyncio.run(model_swap.ensure_model("small"))

    fake_client_factory.assert_not_called()


def test_ensure_model_noop_when_swap_url_unset():
    llm_config.configure(keep_models_loaded=False, model_swap_url="")

    fake_client_factory = MagicMock()
    with patch("src.model_swap.httpx.AsyncClient", fake_client_factory):
        asyncio.run(model_swap.ensure_model("small"))

    fake_client_factory.assert_not_called()


def test_ensure_model_posts_when_configured_and_skips_redundant_swap():
    llm_config.configure(
        keep_models_loaded=False,
        model_swap_url="http://swap.local/load",
        small_model="SMALL-X",
        large_model="LARGE-X",
    )

    fake = _FakeAsyncClient([_resp("a"), _resp("b")])  # payloads unused, only status matters
    with patch("src.model_swap.httpx.AsyncClient", return_value=fake):

        async def _go():
            await model_swap.ensure_model("small")
            # second call with same tier should be skipped
            await model_swap.ensure_model("small")
            # tier change should trigger another POST
            await model_swap.ensure_model("large")

        asyncio.run(_go())

    assert len(fake.calls) == 2
    assert fake.calls[0]["json"] == {"model": "SMALL-X"}
    assert fake.calls[1]["json"] == {"model": "LARGE-X"}
