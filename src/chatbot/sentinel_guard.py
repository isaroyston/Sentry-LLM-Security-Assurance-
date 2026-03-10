import os
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_SENTINEL_GUARDRAILS: Dict[str, Dict[str, Any]] = {
    "lionguard-2-binary": {}, 
    "off-topic": {},
    "system-prompt-leakage": {},
    "aws/prompt_attack": {},
}

@dataclass
class SentinelResult:
    blocked: bool
    status_code: Optional[int] = None
    response_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    triggering_guardrails: Optional[List[str]] = None # Added this so you know exactly which filter caught it!


class SentinelGuard:
    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        guardrails: Optional[Dict[str, Dict[str, Any]]] = None,
        timeout: int = 15,
        threshold: float = 0.90,
        fail_closed: bool = False,
    ):
        self.api_key = api_key or os.getenv("SENTINEL_API_KEY")
        self.url = "https://sentinel.stg.aiguardian.gov.sg/api/v1/validate" 
        self.guardrails = guardrails or DEFAULT_SENTINEL_GUARDRAILS
        self.timeout = timeout
        self.threshold = threshold
        self.fail_closed = fail_closed

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def validate(self, text: str, messages: Optional[List[Dict[str, str]]] = None) -> SentinelResult:
        if not self.enabled:
            return SentinelResult(blocked=False, error="SENTINEL_API_KEY missing")

        # The API specifically asks for text, messages (optional), and guardrails at the top level
        payload: Dict[str, Any] = {
            "text": text,
            "guardrails": self.guardrails,
        }
        if messages:
            payload["messages"] = messages

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status() # Automatically catches 4xx/5xx errors
            response_json = response.json()
            
        except requests.exceptions.RequestException as exc:
            # If the request fails entirely, fallback to your fail_closed preference
            return SentinelResult(
                blocked=self.fail_closed,
                error=f"Sentinel API Request failed: {exc}",
            )

        # --- THE FIX: Evaluate the scores directly ---
        triggered = []
        results_dict = response_json.get("results", {})
        
        for guardrail_name, data in results_dict.items():
            # Extract the score (defaults to 0.0 if missing)
            score = data.get("score", 0.0)
            
            # If the probability is higher than our threshold (0.95), it's a violation
            if score > self.threshold:
                triggered.append(f"{guardrail_name} ({score})")

        return SentinelResult(
            blocked=len(triggered) > 0, # If any guardrail triggered, block the message
            status_code=response.status_code,
            response_json=response_json,
            triggering_guardrails=triggered
        )