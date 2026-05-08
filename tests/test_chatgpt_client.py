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

from chatgpt_automation.client import ChatGPTClient, ChatGPTResponse, ElementNotInteractableException


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

    def test_ensure_logged_in_rebuilds_missing_driver(self) -> None:
        client = self._build_client()
        client.driver = None
        rebuilt_driver = _FakeDriver("https://chatgpt.com/")
        client._rebuild_driver = mock.Mock(side_effect=lambda: setattr(client, "driver", rebuilt_driver))
        client.is_logged_in = mock.Mock(return_value=True)
        client.save_cookies = mock.Mock()

        ChatGPTClient.ensure_logged_in(client, allow_manual_login=True)

        client._rebuild_driver.assert_called_once_with()
        client.save_cookies.assert_called_once_with()
        self.assertIs(client.driver, rebuilt_driver)

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

    def test_find_file_input_prefers_current_composer_input(self) -> None:
        client = self._build_client()
        root = object()
        off_input = mock.Mock()
        off_input.is_enabled.return_value = True
        on_input = mock.Mock()
        on_input.is_enabled.return_value = True
        client.driver.find_elements = mock.Mock(return_value=[off_input, on_input])
        client._composer_root = mock.Mock(return_value=root)
        client._file_input_belongs_to_composer = mock.Mock(side_effect=[False, True])
        client._element_attribute = mock.Mock(return_value="")

        result = ChatGPTClient._find_file_input(client)

        self.assertIs(result, on_input)

    def test_attach_files_resets_selected_input_before_upload(self) -> None:
        client = self._build_client()
        file_input = mock.Mock()
        client._prepare_clean_composer = mock.Mock()
        client._dismiss_blocking_ui = mock.Mock()
        client.driver.execute_script = mock.Mock()
        client._find_file_input = mock.Mock(return_value=file_input)
        client._reset_file_input_value = mock.Mock(return_value=True)
        client._wait_for_pending_attachments = mock.Mock(return_value=True)
        client.selector_catalog = mock.Mock()
        client.selector_catalog.find_by_attribute = mock.Mock(return_value=None)

        attachment = ROOT / "README.md"

        with mock.patch("chatgpt_automation.client.time.sleep", return_value=None):
            ChatGPTClient.attach_files(client, [attachment])

        client._prepare_clean_composer.assert_called_once_with()
        client._reset_file_input_value.assert_called_once_with(file_input)
        file_input.send_keys.assert_called_once_with(str(attachment.resolve()))
        client._wait_for_pending_attachments.assert_called_once_with(expected_min=1, timeout=10.0)

    def test_pending_attachment_count_uses_visible_preview_count(self) -> None:
        client = self._build_client()
        client._attachment_preview_count = mock.Mock(return_value=2)
        client._selected_file_input_count = mock.Mock(return_value=1)

        self.assertEqual(ChatGPTClient._pending_attachment_count(client), 2)

    def test_prepare_clean_composer_reloads_when_attachments_remain(self) -> None:
        client = self._build_client(current_url="https://chatgpt.com/c/existing-chat")
        composer = object()
        client._dismiss_blocking_ui = mock.Mock()
        client._focus_composer = mock.Mock(return_value=composer)
        client._clear_composer = mock.Mock()
        client._clear_pending_attachments = mock.Mock()
        client._reset_file_inputs = mock.Mock()
        client._pending_attachment_count = mock.Mock(return_value=1)
        client._reload_conversation_for_clean_composer = mock.Mock()

        ChatGPTClient._prepare_clean_composer(client)

        client._reload_conversation_for_clean_composer.assert_called_once_with()
        self.assertEqual(client._clear_composer.call_count, 2)
        self.assertEqual(client._clear_pending_attachments.call_count, 2)
        self.assertEqual(client._reset_file_inputs.call_count, 2)

    def test_enter_prompt_falls_back_to_js_when_composer_is_not_interactable(self) -> None:
        client = self._build_client()
        composer = mock.Mock()
        composer.send_keys.side_effect = ElementNotInteractableException("blocked")
        client._dismiss_blocking_ui = mock.Mock()
        client._focus_composer = mock.Mock(return_value=composer)
        client._clear_composer = mock.Mock()
        client._set_composer_text_via_js = mock.Mock()
        client._composer_text = mock.Mock(return_value="Describe this trading chart in JSON only.")

        with mock.patch("chatgpt_automation.client.time.sleep", return_value=None):
            ChatGPTClient.enter_prompt(client, "Describe this trading chart in JSON only.")

        client._set_composer_text_via_js.assert_called_once_with(composer, "Describe this trading chart in JSON only.")

    def test_run_prompt_retries_compose_once_after_interaction_failure(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock(side_effect=[RuntimeError("Could not upload image attachments to ChatGPT."), None])
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client.wait_for_response = mock.Mock(
            return_value=ChatGPTResponse(
                message_id="msg-1",
                model_slug="gpt-test",
                text="done",
                url="https://chatgpt.com/c/retried-chat",
            )
        )
        client._reset_before_compose_retry = mock.Mock()

        response = ChatGPTClient.run_prompt(
            client,
            prompt="Describe the image.",
            attachments=[attachment],
            new_chat=True,
            allow_manual_login=True,
        )

        self.assertEqual(response.url, "https://chatgpt.com/c/retried-chat")
        self.assertEqual(client.attach_files.call_count, 2)
        client._reset_before_compose_retry.assert_called_once_with(new_chat=True)

    def test_run_prompt_retries_attachment_preview_failure_in_new_chat(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock(
            side_effect=[
                RuntimeError("Image upload did not appear in the ChatGPT composer after Selenium selected the file."),
                None,
            ]
        )
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client.wait_for_response = mock.Mock(
            return_value=ChatGPTResponse(
                message_id="msg-1",
                model_slug="gpt-test",
                text="done",
                url="https://chatgpt.com/c/retried-chat",
            )
        )
        client._reset_before_compose_retry = mock.Mock()

        response = ChatGPTClient.run_prompt(
            client,
            prompt="Describe the image.",
            attachments=[attachment],
            new_chat=False,
            allow_manual_login=True,
        )

        self.assertEqual(response.url, "https://chatgpt.com/c/retried-chat")
        self.assertEqual(client.attach_files.call_count, 2)
        client._reset_before_compose_retry.assert_called_once_with(new_chat=True)

    def test_run_prompt_keeps_non_upload_attachment_errors_to_two_compose_attempts(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock()
        client.enter_prompt = mock.Mock(side_effect=RuntimeError("Prompt entry failed."))
        client._send_prompt = mock.Mock()
        client._reset_before_compose_retry = mock.Mock()
        client._rebuild_before_compose_retry = mock.Mock()

        with self.assertRaises(RuntimeError):
            ChatGPTClient.run_prompt(
                client,
                prompt="Describe the image.",
                attachments=[attachment],
                new_chat=False,
                allow_manual_login=True,
            )

        self.assertEqual(client.attach_files.call_count, 2)
        self.assertEqual(client.enter_prompt.call_count, 2)
        client._reset_before_compose_retry.assert_called_once_with(new_chat=False)
        client._rebuild_before_compose_retry.assert_not_called()

    def test_run_prompt_rebuilds_browser_after_repeated_attachment_preview_failure(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock(
            side_effect=[
                RuntimeError("Image upload did not appear in the ChatGPT composer after Selenium selected the file."),
                RuntimeError("Image upload did not appear in the ChatGPT composer after Selenium selected the file."),
                None,
            ]
        )
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client.wait_for_response = mock.Mock(
            return_value=ChatGPTResponse(
                message_id="msg-1",
                model_slug="gpt-test",
                text="done",
                url="https://chatgpt.com/c/rebuilt-chat",
            )
        )
        client._reset_before_compose_retry = mock.Mock()
        client._rebuild_before_compose_retry = mock.Mock()

        response = ChatGPTClient.run_prompt(
            client,
            prompt="Describe the image.",
            attachments=[attachment],
            new_chat=False,
            allow_manual_login=True,
        )

        self.assertEqual(response.url, "https://chatgpt.com/c/rebuilt-chat")
        self.assertEqual(client.attach_files.call_count, 3)
        client._reset_before_compose_retry.assert_called_once_with(new_chat=True)
        client._rebuild_before_compose_retry.assert_called_once_with(allow_manual_login=True)

    def test_run_prompt_rebuilds_browser_before_raising_final_attachment_preview_failure(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock(
            side_effect=RuntimeError(
                "Image upload did not appear in the ChatGPT composer after Selenium selected the file."
            )
        )
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client._reset_before_compose_retry = mock.Mock()
        client._rebuild_before_compose_retry = mock.Mock()

        with self.assertRaises(RuntimeError):
            ChatGPTClient.run_prompt(
                client,
                prompt="Describe the image.",
                attachments=[attachment],
                new_chat=False,
                allow_manual_login=True,
            )

        self.assertEqual(client.attach_files.call_count, 3)
        self.assertEqual(client._reset_before_compose_retry.call_count, 1)
        self.assertEqual(client._rebuild_before_compose_retry.call_count, 2)

    def test_run_prompt_preserves_upload_error_when_final_rebuild_fails(self) -> None:
        client = self._build_client()
        attachment = ROOT / "README.md"
        upload_error = RuntimeError(
            "Image upload did not appear in the ChatGPT composer after Selenium selected the file."
        )
        client.ensure_logged_in = mock.Mock()
        client.start_new_chat = mock.Mock()
        client._wait_for_composer = mock.Mock()
        client._current_assistant_snapshot = mock.Mock(return_value=[])
        client.attach_files = mock.Mock(side_effect=upload_error)
        client.enter_prompt = mock.Mock()
        client._send_prompt = mock.Mock()
        client._reset_before_compose_retry = mock.Mock()
        client._rebuild_before_compose_retry = mock.Mock(side_effect=[None, RuntimeError("rebuild failed")])

        with self.assertRaisesRegex(RuntimeError, "Image upload did not appear"):
            ChatGPTClient.run_prompt(
                client,
                prompt="Describe the image.",
                attachments=[attachment],
                new_chat=False,
                allow_manual_login=True,
            )

        self.assertEqual(client.attach_files.call_count, 3)
        self.assertEqual(client._rebuild_before_compose_retry.call_count, 2)


if __name__ == "__main__":
    unittest.main()
