# -*- coding: utf-8 -*-
"""
Shared LLM and parsing utilities for the skill-memory framework.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except (ModuleNotFoundError, ImportError):  # pragma: no cover - environment dependent
    OpenAI = None

try:
    import openai as legacy_openai
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    legacy_openai = None

from framework.config import API_BASE_URL, AVAILABLE_MODELS, DEFAULT_MODEL


class LLMCallError(RuntimeError):
    """Raised when an LLM-enabled component cannot produce a valid result."""


class LLMCaller:
    """Small wrapper around chat completions with optional image input."""

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: str = DEFAULT_MODEL,
        component_name: str = "LLM",
    ):
        if client is not None:
            self.client = client
            self.legacy_client = None
        elif OpenAI is not None:
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=API_BASE_URL,
            )
            self.legacy_client = None
        else:
            self.client = None
            self.legacy_client = self._init_legacy_client()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Available models: "
                f"{', '.join(sorted(AVAILABLE_MODELS))}"
            )
        self.model_config = AVAILABLE_MODELS[model]
        self.component_name = component_name

    @staticmethod
    def _init_legacy_client() -> Any:
        if legacy_openai is None:
            return None
        legacy_openai.api_key = os.getenv("OPENAI_API_KEY")
        if API_BASE_URL:
            # Different old SDK versions use either api_base or base_url.
            try:
                legacy_openai.api_base = API_BASE_URL
            except Exception:
                pass
            try:
                legacy_openai.base_url = API_BASE_URL
            except Exception:
                pass
        return legacy_openai

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        max_retries: int = 3,
    ) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }]

        if image_path and os.path.exists(image_path):
            image_b64 = self.encode_image(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })

        last_error = None
        if self.client is None and self.legacy_client is None and not self.api_key:
            raise RuntimeError(
                "LLM calls require OPENAI_API_KEY. Set the environment variable "
                "or run without --build_with_llm / with deterministic fallbacks."
            )
        for attempt in range(1, max_retries + 1):
            try:
                if self.client is not None:
                    response = self.client.chat.completions.create(
                        model=self.model_config.api_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()

                if self.legacy_client is not None and hasattr(self.legacy_client, "ChatCompletion"):
                    response = self.legacy_client.ChatCompletion.create(
                        model=self.model_config.api_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return _extract_legacy_content(response)

                return self._call_http(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # pragma: no cover - depends on external API
                last_error = exc
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(
                        f"[{self.component_name}] API call failed "
                        f"(attempt {attempt}/{max_retries}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    print(
                        f"[{self.component_name}] API call failed after "
                        f"{max_retries} attempts: {exc}"
                    )

        raise LLMCallError(f"{self.component_name} API call failed: {last_error}")

    def _call_http(
        self,
        messages: Any,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """SDK-free OpenAI-compatible chat completions call."""
        url = API_BASE_URL.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model_config.api_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} from LLM endpoint: {detail}") from exc
        return str(data["choices"][0]["message"]["content"]).strip()


def _extract_legacy_content(response: Any) -> str:
    """Extract message content from old openai SDK response shapes."""
    try:
        return response["choices"][0]["message"]["content"].strip()
    except Exception:
        pass
    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return str(response).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def normalize_prediction(value: Any, positive_label: str = "harmful") -> int:
    """Parse harmful/harmless style labels while respecting negations."""
    text = str(value or "").lower().strip()
    positive = positive_label.lower()
    negative_markers = [
        "not misogynistic",
        "not harmful",
        "not hateful",
        "harmless",
        "benign",
        "safe",
        "0",
        "false",
    ]
    if any(marker in text for marker in negative_markers):
        return 0
    positive_markers = [
        positive,
        "misogynistic",
        "harmful",
        "hateful",
        "unsafe",
        "1",
        "true",
    ]
    if any(marker in text for marker in positive_markers):
        return 1
    return 0
