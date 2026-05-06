from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation.client import ChatGPTClient, ChatGPTResponse


class _FakeDriver:
    def __init__(self, current_url: str) -> None:
        self.current_url = current_url
        self.visited_urls: list[str] = []

    def get(self, url: str) -> None:
        self.visited_urls.append(url)
        self.current_url = url

    def find_elements(self, *_args, **_kwargs) -> list[object]:
        return []


class ChatGPTClientTests(unittest.TestCase):
    def _build_client(self, current_url: str = "https://chatgpt.com/") -> ChatGPTClient:
        client = ChatGPTClient.__new__(ChatGPTClient)
        client.config = SimpleNamespace(
            base_url="https://chatgpt.com/",
            response_timeout_seconds=10.0,
            response_idle_seconds=0.0,
            page_load_timeout=5.0,
            send_button_timeout_seconds=5.0,
            prompt_settle_seconds=0.0,
        )
        client.driver = _FakeDriver(current_url)
        client.selector_catalog = None
        client.wait = None
        client._last_conversation_url = None
        return client

    def test_ensure_logged_in_keeps_existing_chat_open(self) -> None:
        client = self._build_client(current_url="https://chatgpt.com/c/existing-chat")
        client.is_logged_in = mock.Mock(return_value=True)
        client.save_cookies = mock.Mock()

        ChatGPTClient.ensure_logged_in(client, allow_manual_login=True)

        self.assertEqual(client.driver.visited_urls, [])
        client.save_cookies.assert_called_once_with()

    def test_run_prompt_restores_last_conversation_when_reusing_chat(self) -> None:
        client = self._build_client()
        client._last_conversation_url = "https://chatgpt.com/c/existing-chat"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._prepare_clean_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock()
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client.wait_for_response = mock.Mock(
            return_value=ChatGPTResponse(
                message_id="msg-1",
                model_slug="gpt-test",
                text="done",
                url="https://chatgpt.com/c/existing-chat",
            )
        )

        response = ChatGPTClient.run_prompt(
            client,
            prompt="Describe the image.",
            attachments=None,
            new_chat=False,
            allow_manual_login=True,
        )

        self.assertEqual(response.text, "done")
        self.assertEqual(client.driver.visited_urls, ["https://chatgpt.com/c/existing-chat"])
        client.start_new_chat.assert_not_called()

    def test_wait_for_response_waits_until_generation_finishes(self) -> None:
        client = self._build_client(current_url="https://chatgpt.com/c/existing-chat")
        client._page_looks_like_bot_challenge = mock.Mock(return_value=False)
        client._current_assistant_snapshot = mock.Mock(
            side_effect=[
                [("msg-1", "msg-1", "partial", "gpt-test")],
                [("msg-1", "msg-1", "partial", "gpt-test")],
                [("msg-1", "msg-1", "complete", "gpt-test")],
                [("msg-1", "msg-1", "complete", "gpt-test")],
                [("msg-1", "msg-1", "complete", "gpt-test")],
                [("msg-1", "msg-1", "complete", "gpt-test")],
                [("msg-1", "msg-1", "complete", "gpt-test")],
            ]
        )
        client._is_generation_in_progress = mock.Mock(side_effect=[True, False, False, False, False])

        with (
            mock.patch(
                "chatgpt_automation.client.time.time",
                side_effect=[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ),
            mock.patch("chatgpt_automation.client.time.sleep", return_value=None),
        ):
            response = ChatGPTClient.wait_for_response(client, previous_message_ids=set())

        self.assertEqual(response.text, "complete")
        self.assertEqual(response.url, "https://chatgpt.com/c/existing-chat")
        self.assertEqual(client._is_generation_in_progress.call_count, 2)


if __name__ == "__main__":
    unittest.main()
