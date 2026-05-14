from __future__ import annotations

import json
import re
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .browser import (
    build_driver_with_fallback,
    launch_browser_for_manual_login,
    load_cookie_payload,
    normalize_cookie_payload,
    require_selenium,
    set_cookie_payload,
)
from .config import ChatGPTAutomationConfig
from .selectors import SelectorCatalog, SelectorEntry

SELENIUM_IMPORT_ERROR: Exception | None = None
try:
    from selenium.common.exceptions import (
        ElementNotInteractableException,
        NoSuchElementException,
        StaleElementReferenceException,
        TimeoutException,
        WebDriverException,
    )
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.remote.webelement import WebElement
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError as exc:  # pragma: no cover - depends on local machine state
    SELENIUM_IMPORT_ERROR = exc
    ElementNotInteractableException = Exception  # type: ignore[assignment]
    NoSuchElementException = Exception  # type: ignore[assignment]
    StaleElementReferenceException = Exception  # type: ignore[assignment]
    TimeoutException = Exception  # type: ignore[assignment]
    WebDriverException = Exception  # type: ignore[assignment]
    class _FallbackBy:
        CSS_SELECTOR = "css selector"
        TAG_NAME = "tag name"
        XPATH = "xpath"

    class _FallbackKeys:
        CONTROL = "\ue009"
        DELETE = "\ue017"
        ESCAPE = "\ue00c"

    By = _FallbackBy  # type: ignore[assignment]
    Keys = _FallbackKeys  # type: ignore[assignment]
    WebElement = object  # type: ignore[assignment]
    EC = object  # type: ignore[assignment]
    WebDriverWait = object  # type: ignore[assignment]


COMPOSER_CSS_CANDIDATES = [
    "div#prompt-textarea[contenteditable='true']",
    "div#prompt-textarea",
    "form div[contenteditable='true'][role='textbox']",
    "form div[contenteditable='true']",
    "textarea[placeholder]",
    "textarea",
]

SEND_BUTTON_CSS_CANDIDATES = [
    "button[data-testid='send-button']",
    "button#composer-submit-button",
]
SEND_BUTTON_ATTRIBUTE_MARKERS = (
    "send-button",
    "composer-submit",
    "submit-button",
    "send prompt",
    "send message",
    "nachricht senden",
    "aufforderung senden",
)

STOP_BUTTON_CSS_CANDIDATES = [
    "button[data-testid='stop-button']",
    "button[aria-label*='Stop']",
    "button[aria-label*='stop']",
    "button[title*='Stop']",
    "button[title*='stop']",
]
STOP_BUTTON_ATTRIBUTE_MARKERS = (
    "stop-button",
    "stop generating",
    "stop streaming",
    "stop response",
    "cancel response",
    "generierung stoppen",
    "antwort stoppen",
    "streaming stoppen",
)

FILE_INPUT_CSS = "input[type='file']"
ASSISTANT_MESSAGE_CSS = "main [data-message-author-role='assistant']"
DIALOG_CSS_CANDIDATES = [
    "[role='dialog']",
    "[aria-modal='true']",
]
SHARE_DIALOG_MARKERS = (
    "copy link",
    "share",
    "link kopieren",
    "linkedin",
    "reddit",
    "chatgpt",
)
CLOSE_BUTTON_ATTRIBUTE_MARKERS = (
    "close",
    "schließen",
    "schliessen",
    "cancel",
    "dismiss",
)
ATTACHMENT_REMOVE_MARKERS = (
    "remove",
    "delete",
    "cancel",
    "entfernen",
    "lÃ¶schen",
    "loeschen",
    "abbrechen",
)
ATTACHMENT_PREVIEW_CSS_CANDIDATES = [
    "img",
    "[data-testid*='attachment']",
    "[data-testid*='upload']",
    "[data-testid*='file']",
    "[data-testid*='preview']",
    "[aria-label*='attachment']",
    "[aria-label*='Attachment']",
    "[aria-label*='Anhang']",
    "[aria-label*='Bild']",
    "[aria-label*='image']",
    "[aria-label*='Image']",
    "[class*='attachment']",
    "[class*='upload']",
    "[class*='preview']",
]
ATTACHMENT_UPLOAD_FAILURE_MARKERS = (
    "could not upload image attachments",
    "could not upload file attachments",
    "file upload",
    "file input",
)


@dataclass(frozen=True)
class ChatGPTResponse:
    message_id: str | None
    model_slug: str | None
    text: str
    url: str


class ChatGPTClient:
    def __init__(
        self,
        config: ChatGPTAutomationConfig,
        selector_catalog: SelectorCatalog | None = None,
        driver: Any | None = None,
    ) -> None:
        require_selenium()
        self.config = config
        self.selector_catalog = selector_catalog or SelectorCatalog.from_files(config.selector_files)
        self.driver = driver or build_driver_with_fallback(config)
        self.wait = WebDriverWait(self.driver, config.page_load_timeout)
        self._last_conversation_url: str | None = None
        self._apply_stealth_overrides()

    def close(self) -> None:
        if self.driver is None:
            return
        try:
            self.driver.quit()
        finally:
            self.driver = None

    def _apply_stealth_overrides(self) -> None:
        if not self.config.stealth:
            return
        try:
            self.driver.execute_cdp_cmd(
                "Page.addScriptToEvaluateOnNewDocument",
                {
                    "source": """
window.chrome = window.chrome || { runtime: {} };
""",
                },
            )
        except Exception:
            return

    def _elements(self, selector_type: str, selector: str) -> list[WebElement]:
        if selector_type == "xpath":
            return self.driver.find_elements(By.XPATH, selector)
        return self.driver.find_elements(By.CSS_SELECTOR, selector)

    def _rebuild_driver(self) -> None:
        self.close()
        self.driver = build_driver_with_fallback(self.config)
        self.wait = WebDriverWait(self.driver, self.config.page_load_timeout)
        self._apply_stealth_overrides()

    def _find_first_entry_element(self, entry: SelectorEntry, visible_only: bool = True) -> WebElement | None:
        for selector_type, selector in entry.iter_candidates():
            try:
                elements = self._elements(selector_type, selector)
            except Exception:
                continue
            for element in elements:
                try:
                    if visible_only and not element.is_displayed():
                        continue
                    return element
                except StaleElementReferenceException:
                    continue
        return None

    def _find_catalog_entry(self, *label_terms: str) -> SelectorEntry | None:
        return self.selector_catalog.find_by_label_terms(*label_terms)

    def _try_click(self, entry: SelectorEntry | None) -> bool:
        if entry is None:
            return False
        element = self._find_first_entry_element(entry, visible_only=True)
        if element is None:
            return False
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            element.click()
            return True
        except Exception:
            try:
                self.driver.execute_script("arguments[0].click();", element)
                return True
            except Exception:
                return False

    def _slugify(self, text: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip()).strip("_").lower()
        return normalized or "debug"

    def _wait_for_composer(self, timeout: float | None = None) -> WebElement:
        deadline = time.time() + (timeout or self.config.page_load_timeout)
        last_error = "composer not found"

        while time.time() < deadline:
            self.dismiss_cookie_banner()
            self._dismiss_share_dialog()

            element = self._find_best_composer_candidate()
            if element is not None:
                return element

            time.sleep(0.5)

        raise TimeoutException(last_error)

    def _raise_runtime_with_snapshot(self, reason: str, message: str, cause: Exception | None = None) -> None:
        snapshot_dir = self._write_debug_snapshot(reason)
        if snapshot_dir is not None:
            message = f"{message} Debug snapshot: {snapshot_dir}."
        if cause is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from cause

    def dismiss_cookie_banner(self) -> None:
        accept_entry = self._find_catalog_entry("akzeptieren")
        if accept_entry is None:
            return

        current_url = self._current_url().lower()
        if "auth" not in current_url and "login" not in current_url:
            try:
                page_text = self.driver.find_element(By.TAG_NAME, "body").text.lower()
            except Exception:
                page_text = ""
            if not any(marker in page_text for marker in ("cookie", "cookies", "consent", "zustimmung", "einwilligung")):
                return

        self._try_click(accept_entry)

    def save_cookies(self) -> int:
        if self.config.cookies_file is None:
            return 0
        try:
            cookies = self.driver.get_cookies()
        except Exception:
            return 0
        if not cookies:
            return 0

        self.config.cookies_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cookies": cookies}
        self.config.cookies_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(cookies)

    def restore_cookies(self) -> int:
        cookies = load_cookie_payload(self.config.cookies_file)
        if not cookies:
            return 0

        self.driver.get(self.config.base_url)
        added = 0
        for cookie in cookies:
            try:
                self.driver.add_cookie(cookie)
                added += 1
            except Exception:
                continue
        self.driver.get(self.config.base_url)
        return added

    def set_cookies(
        self,
        cookies: list[dict[str, Any]],
        persist: bool = True,
        apply_to_browser: bool = True,
        replace: bool = False,
    ) -> int:
        normalized = normalize_cookie_payload(cookies)
        if persist:
            set_cookie_payload(self.config.cookies_file, normalized, replace=replace)
        if not apply_to_browser or not normalized:
            return len(normalized)

        self.driver.get(self.config.base_url)
        added = 0
        for cookie in normalized:
            try:
                self.driver.add_cookie(cookie)
                added += 1
            except Exception:
                continue
        if added:
            self.driver.get(self.config.base_url)
        return added

    def is_logged_in(self) -> bool:
        try:
            self._wait_for_composer(timeout=8.0)
            return True
        except Exception:
            return False

    def _page_looks_like_bot_challenge(self) -> bool:
        challenge_markers = (
            "just a moment",
            "verify you are human",
            "security check",
            "checking your browser",
            "captcha",
            "cloudflare",
            "unusual activity",
        )
        text_parts: list[str] = []

        try:
            title = str(self.driver.title or "").strip()
            if title:
                text_parts.append(title)
        except Exception:
            pass

        try:
            body = self.driver.find_element(By.TAG_NAME, "body").text.strip()
            if body:
                text_parts.append(body[:4000])
        except Exception:
            pass

        try:
            page_source = str(self.driver.page_source or "").strip()
            if page_source:
                text_parts.append(page_source[:4000])
        except Exception:
            pass

        if not text_parts:
            return False

        combined = "\n".join(text_parts).lower()
        return any(marker in combined for marker in challenge_markers)

    def _manual_login_failure_message(self) -> str:
        if self._page_looks_like_bot_challenge():
            return (
                "ChatGPT is showing a bot/security challenge in the automated browser session. "
                "Complete the first login in the separate bootstrap browser window, close that window, "
                "and rerun the enrichment. If the challenge still appears, the browser-based flow is currently blocked "
                "and an API-based integration is the more reliable path."
            )
        return (
            "ChatGPT session is still not authenticated after the manual profile bootstrap. "
            "Complete the login in the opened browser window, close it afterwards, and rerun the enrichment."
        )

    def _run_manual_login_bootstrap(self) -> None:
        print(
            "ChatGPT session not found. Opening a regular browser window with the shared project profile. "
            "Log in there, then close that browser window so automation can continue.",
            flush=True,
        )
        self.close()
        process = launch_browser_for_manual_login(self.config, url=self.config.login_url)
        try:
            process.wait(timeout=self.config.manual_login_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Manual login did not finish within {int(self.config.manual_login_timeout_seconds)} seconds. "
                "Please rerun the enrichment and close the opened browser window after the ChatGPT session is ready."
            ) from exc
        time.sleep(2.0)
        self._rebuild_driver()

    def ensure_logged_in(self, allow_manual_login: bool = True) -> None:
        if self.driver is None:
            self._rebuild_driver()

        if self.is_logged_in():
            self.save_cookies()
            return

        self.driver.get(self.config.base_url)
        self.restore_cookies()
        if self.is_logged_in():
            self.save_cookies()
            return

        if not allow_manual_login or self.config.headless:
            raise RuntimeError(
                "ChatGPT session is not authenticated. Reuse the saved profile or cookies, "
                "or rerun without headless mode so you can log in once manually."
            )
        self._run_manual_login_bootstrap()
        self.driver.get(self.config.base_url)
        if self.is_logged_in():
            self.save_cookies()
            return

        self.restore_cookies()
        if self.is_logged_in():
            self.save_cookies()
            return

        raise RuntimeError(self._manual_login_failure_message())

    def start_new_chat(self) -> None:
        new_chat_entry = self.selector_catalog.find_by_attribute("data-testid", "create-new-chat-button")
        if self._try_click(new_chat_entry):
            self._wait_for_composer()
            self._last_conversation_url = None
            return
        self.driver.get(self.config.base_url)
        self._wait_for_composer()
        self._last_conversation_url = None

    def _element_attribute(self, element: WebElement, name: str) -> str:
        try:
            value = element.get_attribute(name)
        except Exception:
            return ""
        return str(value or "").strip()

    def _element_is_interactable(self, element: WebElement) -> bool:
        try:
            if not element.is_displayed() or not element.is_enabled():
                return False
        except StaleElementReferenceException:
            return False

        for attribute_name in ("aria-disabled", "disabled", "data-disabled"):
            value = self._element_attribute(element, attribute_name).lower()
            if value in {"true", "disabled"}:
                return False
        return True

    def _element_has_ancestor(self, element: WebElement, selector: str) -> bool:
        try:
            return bool(
                self.driver.execute_script(
                    "return !!(arguments[0] && arguments[0].closest(arguments[1]));",
                    element,
                    selector,
                )
            )
        except Exception:
            return False

    def _element_looks_like_composer(self, element: WebElement) -> bool:
        if not self._element_is_interactable(element):
            return False
        if self._element_has_ancestor(element, "[role='dialog'], [aria-modal='true']"):
            return False
        if self._element_has_ancestor(element, "[data-message-author-role='assistant'], [data-message-author-role='user']"):
            return False

        tag_name = self._element_attribute(element, "tagName").lower()
        if not tag_name:
            try:
                tag_name = str(element.tag_name or "").lower()
            except Exception:
                tag_name = ""
        role = self._element_attribute(element, "role").lower()
        contenteditable = self._element_attribute(element, "contenteditable").lower()
        element_id = self._element_attribute(element, "id").lower()
        placeholder = (
            self._element_attribute(element, "data-placeholder").lower()
            or self._element_attribute(element, "placeholder").lower()
        )

        if tag_name == "textarea":
            return True
        if "prompt-textarea" in element_id:
            return True
        if contenteditable == "true" and role in {"textbox", ""}:
            return True
        if placeholder:
            return True
        return False

    def _composer_score(self, element: WebElement) -> int:
        score = 0
        role = self._element_attribute(element, "role").lower()
        contenteditable = self._element_attribute(element, "contenteditable").lower()
        element_id = self._element_attribute(element, "id").lower()
        placeholder = (
            self._element_attribute(element, "placeholder").lower()
            or self._element_attribute(element, "data-placeholder").lower()
        )

        if "prompt-textarea" in element_id:
            score += 100
        if role == "textbox":
            score += 20
        if contenteditable == "true":
            score += 20
        if placeholder:
            score += 10
        if self._element_has_ancestor(element, "main form"):
            score += 40
        if self._element_has_ancestor(element, "header, nav, aside"):
            score -= 50

        try:
            rect = element.rect
            if isinstance(rect, dict):
                y = float(rect.get("y", 0.0) or 0.0)
                height = float(rect.get("height", 0.0) or 0.0)
                if y >= 350:
                    score += 20
                if height >= 20:
                    score += 5
        except Exception:
            pass

        return score

    def _find_best_composer_candidate(self) -> WebElement | None:
        candidates: list[WebElement] = []
        seen: set[str] = set()

        for candidate in COMPOSER_CSS_CANDIDATES:
            try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, candidate)
            except Exception:
                continue
            for element in elements:
                try:
                    key = str(element.id)
                except Exception:
                    key = str(id(element))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(element)

        best_element: WebElement | None = None
        best_score = -10_000
        for element in candidates:
            try:
                if not self._element_looks_like_composer(element):
                    continue
                score = self._composer_score(element)
            except StaleElementReferenceException:
                continue
            if score > best_score:
                best_score = score
                best_element = element
        return best_element

    def _dismiss_blocking_ui(self) -> bool:
        dismissed = self._dismiss_share_dialog()
        if not self._visible_dialogs():
            return dismissed

        try:
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.ESCAPE)
            time.sleep(0.2)
        except Exception:
            pass

        return self._dismiss_share_dialog() or dismissed

    def _looks_like_send_button(self, element: WebElement) -> bool:
        attributes = " ".join(
            self._element_attribute(element, name).lower()
            for name in ("data-testid", "id", "aria-label", "title", "class")
        )
        return any(marker in attributes for marker in SEND_BUTTON_ATTRIBUTE_MARKERS)

    def _looks_like_stop_button(self, element: WebElement) -> bool:
        attributes = " ".join(
            self._element_attribute(element, name).lower()
            for name in ("data-testid", "id", "aria-label", "title", "class")
        )
        return any(marker in attributes for marker in STOP_BUTTON_ATTRIBUTE_MARKERS)

    def _iter_send_button_candidates(self) -> list[WebElement]:
        candidates: list[WebElement] = []
        seen: set[str] = set()

        def _append_unique(elements: list[WebElement]) -> None:
            for element in elements:
                try:
                    key = str(element.id)
                except Exception:
                    key = str(id(element))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(element)

        send_entry = self.selector_catalog.find_by_attribute("data-testid", "send-button")
        if send_entry:
            for selector_type, selector in send_entry.iter_candidates():
                try:
                    _append_unique(self._elements(selector_type, selector))
                except Exception:
                    continue

        for candidate in SEND_BUTTON_CSS_CANDIDATES:
            try:
                _append_unique(self.driver.find_elements(By.CSS_SELECTOR, candidate))
            except Exception:
                continue

        try:
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
        except Exception:
            all_buttons = []
        for element in all_buttons:
            if self._looks_like_send_button(element):
                _append_unique([element])

        return candidates

    def _iter_stop_button_candidates(self) -> list[WebElement]:
        candidates: list[WebElement] = []
        seen: set[str] = set()

        def _append_unique(elements: list[WebElement]) -> None:
            for element in elements:
                try:
                    key = str(element.id)
                except Exception:
                    key = str(id(element))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(element)

        for candidate in STOP_BUTTON_CSS_CANDIDATES:
            try:
                _append_unique(self.driver.find_elements(By.CSS_SELECTOR, candidate))
            except Exception:
                continue

        try:
            all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
        except Exception:
            all_buttons = []
        for element in all_buttons:
            if self._looks_like_stop_button(element):
                _append_unique([element])

        return candidates

    def _find_send_button(self, timeout: float | None = None) -> WebElement:
        deadline = time.time() + (timeout or self.config.send_button_timeout_seconds)

        while time.time() < deadline:
            self.dismiss_cookie_banner()
            self._dismiss_share_dialog()
            for element in self._iter_send_button_candidates():
                if self._element_is_interactable(element):
                    return element

            if self._page_looks_like_bot_challenge():
                raise RuntimeError(self._manual_login_failure_message())
            time.sleep(0.5)

        snapshot_dir = self._write_debug_snapshot("send_button_missing")
        raise NoSuchElementException(
            "Could not find the ChatGPT send button after waiting for the prompt and attachments to become ready. "
            f"Pending attachments: {self._pending_attachment_count()}. Debug snapshot: {snapshot_dir}."
        )

    def _focus_composer(self) -> WebElement:
        self._dismiss_blocking_ui()
        composer = self._wait_for_composer()
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", composer)
            composer.click()
        except Exception:
            pass
        try:
            self.driver.execute_script("arguments[0].focus();", composer)
        except Exception:
            pass
        try:
            active_element = self.driver.switch_to.active_element
            if active_element and self._element_looks_like_composer(active_element):
                return active_element
        except Exception:
            return composer
        return composer

    def _clear_composer(self, composer: WebElement) -> None:
        try:
            self.driver.execute_script(
                """
const element = arguments[0];
if (!element) {
  return;
}
const tag = (element.tagName || '').toLowerCase();
if (tag === 'textarea' || tag === 'input') {
  element.value = '';
  element.dispatchEvent(new Event('input', { bubbles: true }));
  element.dispatchEvent(new Event('change', { bubbles: true }));
  return;
}
if ((element.getAttribute('contenteditable') || '').toLowerCase() === 'true') {
  element.innerHTML = '';
  element.textContent = '';
  element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward', data: null }));
}
""",
                composer,
            )
        except Exception:
            pass
        try:
            composer.send_keys(Keys.CONTROL, "a")
            composer.send_keys(Keys.DELETE)
        except Exception:
            return

    def _composer_text(self, composer: WebElement) -> str:
        try:
            value = composer.get_attribute("value")
        except Exception:
            value = None
        if value:
            return str(value).strip()
        try:
            text = composer.text
        except Exception:
            text = ""
        if text:
            return str(text).strip()
        try:
            raw_text = self.driver.execute_script(
                "return arguments[0] ? ((arguments[0].innerText || arguments[0].textContent || '') + '') : '';",
                composer,
            )
        except Exception:
            raw_text = ""
        return str(raw_text or "").strip()

    def _set_composer_text_via_js(self, composer: WebElement, prompt: str) -> None:
        self.driver.execute_script(
            """
const element = arguments[0];
const value = String(arguments[1] || '');
if (!element) {
  return;
}
element.focus();
const tag = (element.tagName || '').toLowerCase();
if (tag === 'textarea' || tag === 'input') {
  element.value = value;
  element.dispatchEvent(new Event('input', { bubbles: true }));
  element.dispatchEvent(new Event('change', { bubbles: true }));
  return;
}
if ((element.getAttribute('contenteditable') || '').toLowerCase() === 'true') {
  element.innerHTML = '';
  const lines = value.split(/\\r?\\n/);
  for (const line of lines) {
    const p = document.createElement('p');
    p.textContent = line;
    element.appendChild(p);
  }
  element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: value }));
}
""",
            composer,
            prompt,
        )

    def enter_prompt(self, prompt: str) -> None:
        self._dismiss_blocking_ui()
        composer = self._focus_composer()
        self._clear_composer(composer)
        try:
            composer.send_keys(prompt)
        except (ElementNotInteractableException, StaleElementReferenceException) as exc:
            self._dismiss_blocking_ui()
            composer = self._focus_composer()
            self._clear_composer(composer)
            self._set_composer_text_via_js(composer, prompt)
            last_error: Exception | None = exc
        else:
            last_error = None
        time.sleep(self.config.prompt_settle_seconds)
        entered_text = self._composer_text(composer)
        minimum_length = max(10, min(len(prompt.strip()) // 4, 80))
        if prompt.strip() and len(entered_text) < minimum_length:
            self._dismiss_blocking_ui()
            composer = self._focus_composer()
            self._set_composer_text_via_js(composer, prompt)
            time.sleep(self.config.prompt_settle_seconds)
            entered_text = self._composer_text(composer)
        if prompt.strip() and len(entered_text) < minimum_length:
            self._raise_runtime_with_snapshot(
                "prompt_entry_failed",
                "Prompt text did not appear in the ChatGPT composer after typing.",
                cause=last_error,
            )

    def _current_assistant_snapshot(self) -> list[tuple[str, str | None, str, str | None]]:
        snapshots: list[tuple[str, str | None, str, str | None]] = []
        for index, element in enumerate(self.driver.find_elements(By.CSS_SELECTOR, ASSISTANT_MESSAGE_CSS)):
            try:
                if self._element_has_ancestor(element, "[role='dialog'], [aria-modal='true']"):
                    continue
                text = element.text.strip()
                message_id = element.get_attribute("data-message-id")
                model_slug = element.get_attribute("data-message-model-slug")
            except StaleElementReferenceException:
                continue
            if text:
                message_key = str(message_id or "").strip() or f"assistant_index_{index}"
                snapshots.append((message_key, message_id, text, model_slug))
        return snapshots

    def _current_url(self) -> str:
        try:
            return str(self.driver.current_url or "").strip()
        except Exception:
            return ""

    def _normalize_response_text(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in str(text).replace("\r\n", "\n").splitlines()).strip()

    def _is_generation_in_progress(self) -> bool:
        for element in self._iter_stop_button_candidates():
            if self._element_is_interactable(element):
                return True
        return False

    def _visible_dialogs(self) -> list[WebElement]:
        dialogs: list[WebElement] = []
        seen: set[str] = set()
        for selector in DIALOG_CSS_CANDIDATES:
            try:
                candidates = self.driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                continue
            for element in candidates:
                try:
                    if not element.is_displayed():
                        continue
                    key = str(element.id)
                except Exception:
                    key = str(id(element))
                if key in seen:
                    continue
                seen.add(key)
                dialogs.append(element)
        return dialogs

    def _dialog_looks_like_share_popup(self, dialog: WebElement) -> bool:
        try:
            text = dialog.text.lower()
        except Exception:
            return False
        hits = sum(1 for marker in SHARE_DIALOG_MARKERS if marker in text)
        return hits >= 2

    def _dismiss_share_dialog(self) -> bool:
        for dialog in self._visible_dialogs():
            if not self._dialog_looks_like_share_popup(dialog):
                continue
            try:
                buttons = dialog.find_elements(By.TAG_NAME, "button")
            except Exception:
                buttons = []
            for button in buttons:
                if not self._element_is_interactable(button):
                    continue
                attributes = " ".join(
                    self._element_attribute(button, name).lower()
                    for name in ("aria-label", "title", "class", "data-testid")
                )
                text = ""
                try:
                    text = button.text.strip().lower()
                except Exception:
                    text = ""
                if any(marker in attributes for marker in CLOSE_BUTTON_ATTRIBUTE_MARKERS) or text in {"×", "x", "close"}:
                    try:
                        button.click()
                    except Exception:
                        try:
                            self.driver.execute_script("arguments[0].click();", button)
                        except Exception:
                            continue
                    return True
        return False

    def _find_file_input(self) -> WebElement | None:
        try:
            inputs = self.driver.find_elements(By.CSS_SELECTOR, FILE_INPUT_CSS)
        except Exception:
            return None
        if not inputs:
            return None
        root = self._composer_root()
        best_element: WebElement | None = None
        best_score = -1
        for index, element in enumerate(inputs):
            try:
                if not element.is_enabled():
                    continue
            except StaleElementReferenceException:
                continue
            score = index
            if self._file_input_belongs_to_composer(element, root):
                score += 100
            if self._element_attribute(element, "multiple").lower() in {"true", "multiple"}:
                score += 10
            try:
                if element.is_displayed():
                    score += 20
            except StaleElementReferenceException:
                continue
            if score >= best_score:
                best_score = score
                best_element = element
        if best_element is not None:
            return best_element
        return inputs[-1]

    def _file_input_belongs_to_composer(self, file_input: WebElement, root: WebElement | None = None) -> bool:
        if root is None:
            root = self._composer_root()
        if root is None:
            return False
        try:
            return bool(
                self.driver.execute_script(
                    "return !!(arguments[0] && arguments[1] && arguments[0].contains(arguments[1]));",
                    root,
                    file_input,
                )
            )
        except Exception:
            return False

    def _reset_file_input_value(self, file_input: WebElement) -> bool:
        try:
            cleared = self.driver.execute_script(
                """
const input = arguments[0];
if (!input) {
  return false;
}
try {
  input.value = '';
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
} catch (error) {
  return false;
}
const fileCount = input.files ? Number(input.files.length || 0) : 0;
return fileCount === 0 && !String(input.value || '');
""",
                file_input,
            )
        except Exception:
            return False
        return bool(cleared)

    def _reset_file_inputs(self) -> int:
        cleared = 0
        try:
            inputs = self.driver.find_elements(By.CSS_SELECTOR, FILE_INPUT_CSS)
        except Exception:
            return 0
        for file_input in inputs:
            if self._reset_file_input_value(file_input):
                cleared += 1
        return cleared

    def _composer_root(self) -> WebElement | None:
        composer = self._find_best_composer_candidate()
        if composer is None:
            return None
        try:
            root = self.driver.execute_script("return arguments[0] ? (arguments[0].closest('form') || arguments[0]) : null;", composer)
        except Exception:
            root = None
        return root or composer

    def _attachment_preview_count(self) -> int:
        root = self._composer_root()
        if root is None:
            return 0
        try:
            count = self.driver.execute_script(
                """
const root = arguments[0];
const selectors = arguments[1] || [];
if (!root || !selectors.length) {
  return 0;
}
const scopes = [root];
const main = root.closest('main');
if (main && main !== root) {
  scopes.push(main);
}
const nodes = new Set();
const popupSelector = [
  '[role="dialog"]',
  '[aria-modal="true"]',
  '[role="menu"]',
  '[role="menuitem"]',
  '[role="listbox"]',
  '[data-radix-popper-content-wrapper]'
].join(',');
const controlSelector = [
  'button',
  'input',
  'textarea',
  'select',
  'option',
  'label',
  'summary',
  'a[href]',
  '[role="button"]',
  '[role="menuitem"]',
  '[role="option"]'
].join(',');
const previewEvidenceSelector = [
  'img',
  'canvas',
  'video',
  '[data-testid*="attachment"]',
  '[data-testid*="preview"]',
  '[aria-label*="attachment"]',
  '[aria-label*="Attachment"]',
  '[aria-label*="Anhang"]',
  '[class*="attachment"]',
  '[class*="preview"]'
].join(',');
const uploadControlPattern = /\\b(upload|file input|choose file|attach file|datei hochladen|datei ausw|anhang hochladen)\\b/i;
for (const scope of scopes) {
  for (const selector of selectors) {
    for (const node of scope.querySelectorAll(selector)) {
      if (node.closest('[data-message-author-role]') || (node.closest(popupSelector) && !root.contains(node))) {
        continue;
      }
      if (node.matches(controlSelector) || (node.closest(controlSelector) && !node.matches('img, canvas, video'))) {
        continue;
      }
      const style = window.getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      const hidden = style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || 1) === 0;
      if (hidden || (rect.width <= 0 && rect.height <= 0 && node.tagName !== 'IMG')) {
        continue;
      }
      const text = [
        node.getAttribute('aria-label') || '',
        node.getAttribute('title') || '',
        node.getAttribute('data-testid') || '',
        node.textContent || ''
      ].join(' ').trim();
      const hasPreviewEvidence = node.matches(previewEvidenceSelector) || !!node.querySelector(previewEvidenceSelector);
      const looksLikeUploadControl = uploadControlPattern.test(text);
      if (hasPreviewEvidence && !looksLikeUploadControl) {
        nodes.add(node);
      }
    }
  }
}
return nodes.size;
""",
                root,
                ATTACHMENT_PREVIEW_CSS_CANDIDATES,
            )
        except Exception:
            return 0
        try:
            return int(count or 0)
        except Exception:
            return 0

    def _selected_file_input_count(self) -> int:
        try:
            count = self.driver.execute_script(
                """
const selector = arguments[0];
let total = 0;
for (const input of document.querySelectorAll(selector)) {
  if (input.files && input.files.length) {
    total += Number(input.files.length || 0);
  }
}
return total;
""",
                FILE_INPUT_CSS,
            )
        except Exception:
            return 0
        try:
            return int(count or 0)
        except Exception:
            return 0

    def _pending_attachment_count(self) -> int:
        return self._attachment_preview_count()

    def _clear_pending_attachments(self) -> int:
        removed = 0
        for _ in range(6):
            root = self._composer_root()
            if root is None:
                break
            clicked = False
            try:
                buttons = root.find_elements(By.TAG_NAME, "button")
            except Exception:
                buttons = []
            for button in buttons:
                if not self._element_is_interactable(button):
                    continue
                attributes = " ".join(
                    self._element_attribute(button, name).lower()
                    for name in ("aria-label", "title", "class", "data-testid")
                )
                text = ""
                try:
                    text = button.text.strip().lower()
                except Exception:
                    text = ""
                combined = f"{attributes} {text}".strip()
                if not any(marker in combined for marker in ATTACHMENT_REMOVE_MARKERS):
                    continue
                try:
                    button.click()
                except Exception:
                    try:
                        self.driver.execute_script("arguments[0].click();", button)
                    except Exception:
                        continue
                removed += 1
                clicked = True
                time.sleep(0.3)
                break
            if not clicked:
                break
        return removed

    def _reload_conversation_for_clean_composer(self) -> None:
        target_url = self._current_url() or self._last_conversation_url or self.config.base_url
        self.driver.get(target_url)
        self._wait_for_composer()

    def _prepare_clean_composer(self) -> None:
        self._dismiss_blocking_ui()
        composer = self._focus_composer()
        self._clear_composer(composer)
        self._clear_pending_attachments()
        self._reset_file_inputs()
        if self._pending_attachment_count() > 0:
            self._reload_conversation_for_clean_composer()
            self._dismiss_blocking_ui()
            composer = self._focus_composer()
            self._clear_composer(composer)
            self._clear_pending_attachments()
            self._reset_file_inputs()

    def _write_debug_snapshot(self, reason: str) -> Path | None:
        debug_root = self.config.chatgpt_dir.parent / ".runtime" / "chatgpt" / "debug"
        snapshot_dir = debug_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{self._slugify(reason)}"
        try:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "reason": reason,
                "url": self._current_url(),
                "pending_attachment_count": self._pending_attachment_count(),
                "selected_file_input_count": self._selected_file_input_count(),
                "visible_dialog_count": len(self._visible_dialogs()),
            }
            (snapshot_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                (snapshot_dir / "page.html").write_text(self.driver.page_source or "", encoding="utf-8")
            except Exception:
                pass
            try:
                self.driver.save_screenshot(str(snapshot_dir / "page.png"))
            except Exception:
                pass
            return snapshot_dir
        except Exception:
            return None

    def _wait_for_pending_attachments(self, expected_min: int = 1, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        minimum = max(1, expected_min)
        while time.time() < deadline:
            self._dismiss_share_dialog()
            if self._pending_attachment_count() >= minimum:
                return True
            time.sleep(0.3)
        return False

    def _reset_before_compose_retry(self, new_chat: bool) -> None:
        self._dismiss_blocking_ui()
        if new_chat:
            self.start_new_chat()
            return
        self._reload_conversation_for_clean_composer()
        self._dismiss_blocking_ui()

    def _rebuild_before_compose_retry(self, allow_manual_login: bool) -> None:
        self._last_conversation_url = None
        self._rebuild_driver()
        self.ensure_logged_in(allow_manual_login=allow_manual_login)
        self.start_new_chat()

    def _is_attachment_upload_failure(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in ATTACHMENT_UPLOAD_FAILURE_MARKERS)

    def _is_attachment_preview_missing_failure(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "attachment_preview_missing" in message or "image upload did not appear" in message

    def _compose_and_send_prompt(
        self,
        prompt: str,
        attachments: list[Path] | None,
        new_chat: bool,
        allow_manual_login: bool,
    ) -> None:
        last_error: Exception | None = None
        retry_new_chat = new_chat
        rebuild_before_attempt = False
        max_attempts = 2
        attempt = 0
        while attempt < max_attempts:
            if attempt > 0:
                if rebuild_before_attempt:
                    self._rebuild_before_compose_retry(allow_manual_login=allow_manual_login)
                    rebuild_before_attempt = False
                else:
                    self._reset_before_compose_retry(new_chat=retry_new_chat)
            try:
                if attachments:
                    self.attach_files(attachments)
                else:
                    self._prepare_clean_composer()
                self.enter_prompt(prompt)
                self._send_prompt()
                return
            except Exception as exc:
                last_error = exc
                if bool(attachments) and self._is_attachment_preview_missing_failure(exc):
                    raise
                attachment_upload_failure = bool(attachments) and self._is_attachment_upload_failure(exc)
                if attachment_upload_failure:
                    max_attempts = 3
                if attempt < max_attempts - 1:
                    retry_new_chat = new_chat or attachment_upload_failure
                    rebuild_before_attempt = attachment_upload_failure and attempt > 0
                    attempt += 1
                    continue
                if attachment_upload_failure:
                    with suppress(Exception):
                        self._rebuild_before_compose_retry(allow_manual_login=allow_manual_login)
                raise
            attempt += 1
        if last_error is not None:
            raise last_error

    def _restore_last_conversation(self) -> None:
        if not self._last_conversation_url:
            return
        current_url = self._current_url().rstrip("/")
        target_url = self._last_conversation_url.rstrip("/")
        if current_url == target_url:
            return
        self.driver.get(self._last_conversation_url)
        self._wait_for_composer()

    def attach_files(self, attachments: list[Path]) -> None:
        normalized_paths = [str(path.resolve()) for path in attachments]
        if not normalized_paths:
            return

        self._prepare_clean_composer()
        self._dismiss_blocking_ui()
        self.driver.execute_script(
            """
for (const input of document.querySelectorAll(arguments[0])) {
  input.hidden = false;
  input.style.display = 'block';
  input.style.visibility = 'visible';
  input.style.opacity = 1;
  input.style.position = 'fixed';
  input.style.left = '0';
  input.style.top = '0';
  input.style.zIndex = 2147483647;
}
""",
            FILE_INPUT_CSS,
        )

        deadline = time.time() + self.config.page_load_timeout
        file_input = self._find_file_input()
        if file_input is None:
            try:
                self._try_click(self.selector_catalog.find_by_attribute("data-testid", "composer-plus-btn"))
            except Exception:
                pass

        while time.time() < deadline:
            file_input = self._find_file_input()
            if file_input is not None:
                break
            time.sleep(0.3)

        if file_input is None:
            self._raise_runtime_with_snapshot(
                "file_input_missing",
                "Could not find a file input on the ChatGPT page.",
            )

        self._reset_file_input_value(file_input)
        try:
            file_input.send_keys("\n".join(normalized_paths))
        except (ElementNotInteractableException, StaleElementReferenceException) as exc:
            self._dismiss_blocking_ui()
            refreshed_input = self._find_file_input()
            if refreshed_input is None:
                self._raise_runtime_with_snapshot(
                    "file_input_interaction_failed",
                    "File upload input became unavailable before Selenium could attach the image.",
                    cause=exc,
                )
            self._reset_file_input_value(refreshed_input)
            try:
                refreshed_input.send_keys("\n".join(normalized_paths))
            except Exception as retry_exc:
                self._raise_runtime_with_snapshot(
                    "file_upload_failed",
                    "Could not upload image attachments to ChatGPT after retrying the file input.",
                    cause=retry_exc if isinstance(retry_exc, Exception) else exc,
                )
        time.sleep(0.5)
        if not self._wait_for_pending_attachments(expected_min=1, timeout=10.0):
            self._raise_runtime_with_snapshot(
                "attachment_preview_missing",
                "Image upload did not appear in the ChatGPT composer after Selenium selected the file.",
            )

    def _send_prompt(self) -> None:
        try:
            self._dismiss_blocking_ui()
            send_button = self._find_send_button(timeout=self.config.send_button_timeout_seconds)
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", send_button)
            try:
                send_button.click()
            except Exception:
                self.driver.execute_script("arguments[0].click();", send_button)
        except Exception as exc:
            self._raise_runtime_with_snapshot(
                "send_prompt_failed",
                "Could not click the ChatGPT send button.",
                cause=exc,
            )

    def wait_for_response(self, previous_message_ids: set[str]) -> ChatGPTResponse:
        deadline = time.time() + self.config.response_timeout_seconds
        current_message_key: str | None = None
        current_message_id: str | None = None
        current_model_slug: str | None = None
        last_seen_text = ""
        last_change_at: float | None = None
        stable_snapshot_count = 0

        while True:
            now = time.time()
            if now >= deadline:
                break
            self._dismiss_share_dialog()
            snapshots = self._current_assistant_snapshot()
            for message_key, message_id, text, model_slug in reversed(snapshots):
                if message_key in previous_message_ids:
                    continue

                normalized_text = self._normalize_response_text(text)
                if current_message_key != message_key or normalized_text != last_seen_text:
                    current_message_key = message_key
                    current_message_id = message_id
                    current_model_slug = model_slug
                    last_seen_text = normalized_text
                    last_change_at = now
                    stable_snapshot_count = 0
                elif normalized_text and last_change_at is not None:
                    stable_snapshot_count += 1
                    idle_for = now - last_change_at
                    if (
                        stable_snapshot_count >= 2
                        and idle_for >= self.config.response_idle_seconds
                        and not self._is_generation_in_progress()
                    ):
                        return ChatGPTResponse(
                            message_id=current_message_id,
                            model_slug=current_model_slug,
                            text=normalized_text,
                            url=self._current_url(),
                        )
                break
            else:
                if (
                    current_message_key is not None
                    and last_seen_text
                    and last_change_at is not None
                    and stable_snapshot_count >= 2
                    and (now - last_change_at) >= self.config.response_idle_seconds
                    and not self._is_generation_in_progress()
                ):
                    return ChatGPTResponse(
                        message_id=current_message_id,
                        model_slug=current_model_slug,
                        text=last_seen_text,
                        url=self._current_url(),
                    )

            if self._page_looks_like_bot_challenge():
                raise RuntimeError(self._manual_login_failure_message())
            time.sleep(1.0)

        raise TimeoutException("Timed out while waiting for a ChatGPT response.")

    def run_prompt(
        self,
        prompt: str,
        attachments: list[Path] | None = None,
        new_chat: bool = True,
        allow_manual_login: bool = True,
    ) -> ChatGPTResponse:
        self.ensure_logged_in(allow_manual_login=allow_manual_login)
        if new_chat:
            self.start_new_chat()
        else:
            self._restore_last_conversation()

        previous_message_ids = {
            message_key
            for message_key, _message_id, _text, _model_slug in self._current_assistant_snapshot()
        }
        self._compose_and_send_prompt(
            prompt=prompt,
            attachments=attachments,
            new_chat=new_chat,
            allow_manual_login=allow_manual_login,
        )
        response = self.wait_for_response(previous_message_ids)
        if response.url:
            self._last_conversation_url = response.url
        return response
