from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.opencode import OpenCodeProvider


class MockMessage:
    def __init__(self, role, content, reasoning_content=None):
        self.role = role
        self.content = content
        self.reasoning_content = reasoning_content


@pytest.mark.asyncio
async def test_opencode_deepseek_stream_preserves_empty_reasoning_block():
    provider = OpenCodeProvider(ProviderConfig(api_key="test"))
    request = SimpleNamespace(
        model="deepseek-v4-flash-free",
        messages=[MockMessage("user", "use a tool")],
        system=None,
        tools=[],
        tool_choice=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        stop_sequences=None,
        thinking=None,
    )

    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q":"test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason="tool_calls",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [
            event
            async for event in provider.stream_response(request, thinking_enabled=True)
        ]

    assert any(
        "event: content_block_start" in event and '"type": "thinking"' in event
        for event in events
    )
    assert any(
        "event: content_block_start" in event and '"type": "tool_use"' in event
        for event in events
    )
