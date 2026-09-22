from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


Message = Dict[str, str]


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    raw: Optional[Dict[str, Any]] = None


class OllamaClient:
    """Small non-streaming client for the local Ollama HTTP API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/") # rstrip("/") -> removes trailing "/" if there is one
        self.timeout = timeout

    def list_models(self) -> List[str]:
        """ Lists models using GET /api/tags """
        response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def chat(self, model: str, messages: List[Message], temperature: float, max_tokens: int, include_raw: bool = False) -> ChatResult:

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        return ChatResult(
            text              = result.get("message", {}).get("content", ""),
            model             = result.get("model", model),
            prompt_tokens     = result.get("prompt_eval_count", 0),
            completion_tokens = result.get("eval_count", 0),
            raw               = result if include_raw else None,
        )
