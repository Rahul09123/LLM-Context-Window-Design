"""
llm_client.py
=============
Provider-agnostic LLM client.

To switch models/providers, edit the ``llm:`` block in config.yaml only:

    llm:
      provider:    "gemini"           # anthropic | gemini | openai
      model:       "gemini-1.5-flash"
      api_key_env: "GEMINI_API_KEY"   # name of the env-var holding the key
      max_tokens:  1024
      qa_per_conv: 3

All call-sites use:

    from llm_client import build_llm_client
    client = build_llm_client(cfg["llm"])
    text = client.complete(system="...", user="...")

No other file needs to know which provider is active.
"""

import os
import time
import logging
from abc import ABC, abstractmethod

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """Minimal interface every provider adapter must implement."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Send a system + user message and return the assistant text.

        Parameters
        ----------
        system : str
            System / instruction prompt.
        user : str
            User message.
        max_tokens : int
            Maximum tokens in the response.
        max_retries : int
            Retry attempts on transient errors.
        retry_delay : float
            Base seconds to wait between retries (multiplied by attempt #).

        Returns
        -------
        str
            The model's text response, or '' on repeated failure.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic (Claude)
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicClient(LLMClient):
    """Wraps the ``anthropic`` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("Run: pip install anthropic")
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._api_errors = (_anthropic.APIError, _anthropic.RateLimitError)

    def complete(self, system, user, max_tokens=1024, max_retries=3, retry_delay=2.0) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except self._api_errors as exc:
                logger.warning("Anthropic attempt %d/%d: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Google Gemini
# ─────────────────────────────────────────────────────────────────────────────

class GeminiClient(LLMClient):
    """Wraps the ``google-genai`` SDK (v1+)."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai as _genai
            from google.genai import types as _types
        except ImportError:
            raise ImportError("Run: pip install google-genai")
        self._client = _genai.Client(api_key=api_key)
        self._types = _types
        # Strip leading "models/" prefix if present (config may omit it)
        self._model_name = model if "/" not in model else model.split("/", 1)[1]

    def complete(self, system, user, max_tokens=1024, max_retries=3, retry_delay=2.0) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user,
                    config=self._types.GenerateContentConfig(
                        system_instruction=system,
                        max_output_tokens=max_tokens,
                    ),
                )
                return resp.text or ""
            except Exception as exc:
                logger.warning("Gemini attempt %d/%d: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient(LLMClient):
    """Wraps the ``openai`` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI as _OpenAI, APIError, RateLimitError
        except ImportError:
            raise ImportError("Run: pip install openai")
        self._client = _OpenAI(api_key=api_key)
        self._model = model
        self._api_errors = (APIError, RateLimitError)

    def complete(self, system, user, max_tokens=1024, max_retries=3, retry_delay=2.0) -> str:
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                return resp.choices[0].message.content or ""
            except self._api_errors as exc:
                logger.warning("OpenAI attempt %d/%d: %s", attempt, max_retries, exc)
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "gemini":    GeminiClient,
    "openai":    OpenAIClient,
}


def build_llm_client(llm_cfg: dict) -> LLMClient:
    """Instantiate the correct LLMClient from the ``llm:`` config block.

    Parameters
    ----------
    llm_cfg : dict
        The ``llm`` sub-dict from config.yaml, with keys:
        ``provider``, ``model``, ``api_key_env``.

    Returns
    -------
    LLMClient

    Raises
    ------
    ValueError
        If the provider is not recognised.
    EnvironmentError
        If the API-key environment variable is not set.
    """
    provider = llm_cfg["provider"].lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose one of: {list(_PROVIDERS)}"
        )

    env_var = llm_cfg["api_key_env"]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise EnvironmentError(
            f"API key env-var '{env_var}' is not set. "
            f"Run: export {env_var}=<your-key>"
        )

    model = llm_cfg["model"]
    logger.info("LLM provider=%s  model=%s", provider, model)
    return _PROVIDERS[provider](api_key=api_key, model=model)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)
