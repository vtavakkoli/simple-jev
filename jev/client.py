"""Small synchronous HTTP client. Never substitutes a different backend."""
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DecisionError(RuntimeError):
    """A request failed or a backend returned an unusable decision."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward credentials or state to a redirect target.


def _number(value, low, high, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionError(f"Invalid numeric {field}")
    if not math.isfinite(value) or not low <= value <= high:
        raise DecisionError(f"Out-of-range {field}")


def validate_answers(response, questions):
    """Check answer coverage, allowed values and distributions before acting."""
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise DecisionError("Response has no answers object")
    answers = response["answers"]
    if set(answers) != set(questions):
        raise DecisionError("Response question IDs do not match the request")
    for name, question in questions.items():
        answer = answers[name]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise DecisionError(f"Wrong answer type for {name}")
        if kind == "noul":
            _number(answer.get("noul"), 0, 1, "noul")
            continue
        criteria = question["criteria"]
        labels = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != labels:
            raise DecisionError(f"Invalid probability labels for {name}")
        for value in probabilities.values():
            _number(value, 0, 1, "probability")
        if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise DecisionError(f"Probabilities do not sum to one for {name}")
        _number(answer.get("confidence"), 0, 1, "confidence")
        # Hosted JEV confidence is independent of the option distribution.
        if kind == "choice":
            if not isinstance(answer.get("choice"), str) or answer["choice"] not in labels:
                raise DecisionError(f"Unknown choice for {name}")
        else:
            _number(answer.get("score"), 0, len(criteria) - 1, "score")
    return response


class DecisionClient:
    """Select `typesafe` explicitly for real JEV; `local` for HF logits."""
    def __init__(self, backend="typesafe", *, model=None, base_url=None,
                 api_key=None, timeout=30):
        if backend not in {"typesafe", "local"}:
            raise ValueError("backend must be typesafe or local")
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.backend = backend
        self.model = model or ("jev-latest" if backend == "typesafe" else "Qwen/Qwen3.5-0.8B")
        self.base_url = (base_url or ("https://api.typesafe.ai/v1" if backend == "typesafe"
                                    else "http://127.0.0.1:8000/v1")).rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query or fragment")
        if backend == "typesafe" and self.base_url != "https://api.typesafe.ai/v1":
            raise ValueError("Hosted JEV uses the official TypeSafe endpoint")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Invalid base_url")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS except for a loopback local server")
        self._api_key = (api_key or os.environ.get("TYPESAFE_API_KEY")) if backend == "typesafe" else api_key
        if backend == "typesafe" and not self._api_key:
            raise ValueError("Set TYPESAFE_API_KEY to use real hosted JEV")
        self.timeout = timeout
        self._opener = build_opener(_NoRedirect())

    def decide(self, *, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        for name, question in questions.items():
            if not isinstance(name, str) or not isinstance(question, dict):
                raise ValueError("Question IDs must be strings and definitions must be objects")
            kind = question.get("type")
            if kind not in {"choice", "score", "noul"}:
                raise ValueError("Unsupported question type")
            criteria = question.get("criteria")
            if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2
                                     or not all(isinstance(k, str) for k in criteria)):
                raise ValueError("Choice needs at least two named options")
            if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
                raise ValueError("Score needs at least two ordered levels")
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
            # Avoid echoing response bodies that could include request data.
            raise DecisionError(f"{self.backend} HTTP {exc.code}; no fallback was used") from None
        except (URLError, TimeoutError, OSError, ValueError):
            raise DecisionError(f"{self.backend} request failed; no fallback was used") from None
        validate_answers(response, questions)
        response["client_metadata"] = {"backend": self.backend, "requested_model": self.model,
                                       "latency_ms": (time.perf_counter() - start) * 1000}
        return response
