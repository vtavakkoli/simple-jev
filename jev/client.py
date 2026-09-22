"""Synchronous, standard-library client with explicit backend selection."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DecisionError(RuntimeError):
    """Backend or response failure, without request data or secrets in its message."""

    def __init__(self, message: str, *, backend: str | None = None,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.backend = backend
        self.status_code = status_code


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward credentials or private state to a redirect.


def _number(value: Any, low: float, high: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"Invalid numeric {field}")
    if not math.isfinite(value) or not low <= value <= high:
        raise DecisionError(f"Out-of-range {field}")


def _json_value(value: Any) -> bool:
    """Reject Python objects/keys that json.dumps would silently coerce."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False


def validate_request(*, state: Any, questions: Any) -> None:
    """Validate this client's state/question subset without keys or a network call.

    Backend-specific cardinality and context limits remain server-enforced.
    Instructions and criterion descriptions may contain structured JSON.
    """
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be a string, JSON object or JSON array")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty object")
    for name, question in questions.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(question, dict):
            raise ValueError("Question IDs must be nonempty strings and definitions must be objects")
        if set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("Unknown question field; use type, instructions and criteria")
        if "instructions" not in question:
            raise ValueError("Each question must include instructions")
        kind = question.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("Unsupported question type")
        criteria = question.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2
                                 or not all(isinstance(k, str) and k.strip() for k in criteria)):
            raise ValueError("Choice needs at least two nonempty named options")
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            raise ValueError("Score needs at least two ordered levels")
        if kind == "noul" and criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise ValueError("Noul criteria may contain only true and false descriptions")
    try:
        valid = _json_value(state) and _json_value(questions)
    except RecursionError:
        valid = False
    if not valid:
        raise ValueError("Request must contain finite JSON values and string object keys")


def validate_answers(response: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """Check coverage, allowed values and distributions before applications act."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise DecisionError("Response has no answers object")
    answers = response["answers"]
    if set(answers) != set(questions):
        raise DecisionError("Response question IDs do not match the request")
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise DecisionError("Wrong answer type")
        if kind == "noul":
            _number(answer.get("noul"), 0, 1, "noul")
            continue
        criteria = question["criteria"]
        labels = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != labels:
            raise DecisionError("Invalid probability labels")
        for value in probabilities.values():
            _number(value, 0, 1, "probability")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise DecisionError("Probabilities do not sum to one")
        _number(answer.get("confidence"), 0, 1, "confidence")
        # Hosted JEV confidence is independent of the option distribution.
        if kind == "choice":
            if not isinstance(answer.get("choice"), str) or answer["choice"] not in labels:
                raise DecisionError("Unknown choice")
        else:
            _number(answer.get("score"), 0, len(criteria) - 1, "score")
    return response


class DecisionClient:
    """Use `typesafe` for hosted JEV or `local` for the HF classifier endpoint.

    Calls are synchronous, with no retries and no backend fallback. ``timeout``
    is the transport socket timeout, not a total workflow deadline. Local mode
    never reads TYPESAFE_API_KEY. Instances are not promised to be thread-safe.
    """

    def __init__(self, backend: str = "typesafe", *, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 30) -> None:
        if backend not in ("typesafe", "local"):
            raise ValueError("backend must be typesafe or local")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a nonempty string")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
            raise ValueError("base_url must be a nonempty URL")
        self.backend = backend
        self.model = model or ("jev-latest" if backend == "typesafe" else "Qwen/Qwen3.5-0.8B")
        self.base_url = (base_url or ("https://api.typesafe.ai/v1" if backend == "typesafe"
                                    else "http://127.0.0.1:8000/v1")).rstrip("/")
        parsed = urlsplit(self.base_url)
        try:
            parsed.port  # Validate malformed ports before building a request.
        except ValueError:
            raise ValueError("Invalid base_url port") from None
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query or fragment")
        if backend == "typesafe" and self.base_url != "https://api.typesafe.ai/v1":
            raise ValueError("Hosted JEV uses the official TypeSafe endpoint")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or any(c.isspace() for c in self.base_url):
            raise ValueError("Invalid base_url")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS except for a loopback local server")
        self._api_key = (api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")) if backend == "typesafe" else api_key
        if self._api_key is not None and (not isinstance(self._api_key, str) or not self._api_key.strip()
                                         or any(c.isspace() for c in self._api_key)):
            raise ValueError("API key must be nonempty and contain no whitespace")
        if backend == "typesafe" and not self._api_key:
            raise ValueError("Set TYPESAFE_API_KEY to use real hosted JEV")
        self.timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def decide(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Return provider data plus backend/model/latency metadata.

        ValueError indicates invalid input; DecisionError indicates a network,
        HTTP or response-validation failure. The caller owns action execution.
        """
        validate_request(state=state, questions=questions)
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        path = "/systemone" if self.backend == "typesafe" else "/classifier"
        request = Request(self.base_url + path, data=json.dumps(payload, allow_nan=False).encode(),
                          headers=headers, method="POST")
        start = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout) as result:
                response = json.loads(result.read())
        except HTTPError as exc:
            raise DecisionError(f"{self.backend} HTTP {exc.code}; no fallback was used",
                                backend=self.backend, status_code=exc.code) from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise DecisionError(f"{self.backend} request failed; no fallback was used",
                                backend=self.backend) from None
        try:
            validate_answers(response, questions)
        except DecisionError as exc:
            raise DecisionError(str(exc), backend=self.backend) from None
        response["client_metadata"] = {"backend": self.backend, "requested_model": self.model,
                                       "latency_ms": (time.perf_counter() - start) * 1000}
        return response
