# Python client and command-line reference

The root `jev-lab` distribution installs the `jev` module and command. It requires Python 3.10+ and has no runtime dependencies. It does not install the local model server. The package is available from this repository; no PyPI publication is claimed.

```bash
python -m pip install -e .
jev-lab --version
# Equivalent without installing, from the checkout:
python -m jev --help
```

## Request files

Supply a JSON object with `state`, `questions` and optional `model`. State is a string, object or array. Every question has `type` and `instructions`; instructions and criterion descriptions may contain structured JSON. The client rejects unknown question fields and non-finite numbers.

| Type | Criteria | Answer |
| --- | --- | --- |
| `choice` | Object with at least two named options | Selected ID, distribution and confidence |
| `score` | At least two ordered rubric levels | Zero-based score, distribution and confidence |
| `noul` | Optional `true` / `false` descriptions | Support value in [0, 1] |

This client deliberately supports the shared state/question subset. It does not expose chat `messages`, server diagnostic options, streaming, async calls or completion generation. Backend-specific cardinality and context limits remain enforced by the server. For the full local server contract, see [API_REFERENCE.md](../hf-server/API_REFERENCE.md).

## Commands

```bash
# Offline; no API key or HTTP request:
jev-lab validate examples/requests/support-triage.json

# Hosted TypeSafe JEV; reads TYPESAFE_API_KEY:
jev-lab run examples/requests/support-triage.json --backend typesafe

# Local server already running:
jev-lab run examples/requests/evidence-check.json --backend local \
  --model Qwen/Qwen3.5-0.8B --base-url http://127.0.0.1:8000/v1

# Save JSON; replaces an existing file only after a successful response:
jev-lab run examples/requests/support-triage.json --backend typesafe --output result.json
```

Use `-` as the request path to read stdin. `--model` overrides the file's model; otherwise the backend default applies. `--timeout` defaults to 30 seconds and is a socket timeout, not a total end-to-end deadline. Result files may contain sensitive provider output: choose where you store them.

| Exit code | Meaning |
| --- | --- |
| 0 | Valid request or successful call |
| 2 | Invalid arguments, input, configuration or file access |
| 3 | HTTP/network failure or invalid backend response |

The CLI requires `--backend` for inference so a command does not accidentally choose a hosted service. The Python client defaults to `typesafe` for compatibility. Hosted calls require account access and consume provider quota. No credential is requested during offline validation.

## Python interface

```python
from jev import DecisionClient, DecisionError, validate_request

questions = {
    "supported": {
        "type": "noul",
        "instructions": "Does the source explicitly state that offline mode exists?",
    }
}
state = "Offline operation is planned for a future release."
validate_request(state=state, questions=questions)  # no network
client = DecisionClient("typesafe", timeout=30)
try:
    response = client.decide(state=state, questions=questions)
except DecisionError as error:
    print(error.backend, error.status_code)  # status_code is None for non-HTTP failures
    raise
```

### Constructor

| Argument | Default | Behavior |
| --- | --- | --- |
| `backend` | `typesafe` | `typesafe` or `local` |
| `model` | Backend default | `jev-latest` hosted; `Qwen/Qwen3.5-0.8B` local |
| `base_url` | Backend default | Official TypeSafe URL is fixed; local supports a custom API base |
| `api_key` | Hosted environment key | Local never reads `TYPESAFE_API_KEY`; explicit local token is supported |
| `timeout` | 30 | Positive finite socket timeout in seconds |

Instances are synchronous and are not promised to be thread-safe. There are no automatic retries, backend substitutions or cached cross-request answers. Choose retry policy at the application layer with quota and action semantics in mind.

### Result validation

`decide()` preserves provider fields and adds `client_metadata` with the selected backend, requested model and measured latency. It checks answer coverage, types, finite numeric ranges, option membership, distribution keys and normalization (absolute tolerance 0.0001). It does not certify correctness or calibration.

TypeSafe's confidence is distinct from the option probability. Local Simple Jev uses the maximum allowed-label probability as confidence. The client does not replace one with the other. Use task-specific evaluation when setting review thresholds.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing `TYPESAFE_API_KEY` | Set the environment variable in the same terminal/runtime; never paste it into a tracked file |
| HTTP 401 / 403 | Provider key, account access and permissions |
| HTTP 429 | Provider quota/rate limit; the client does not retry automatically |
| Local connection failure | Server is running; API base contains `/v1`; selected port is correct |
| Model rejection | Model ID exactly matches the local server or an available hosted model |
| Invalid probability/answer error | Backend response matches the submitted questions and documented contract |
| `jev-lab` command missing | Activate the environment and run `python -m pip install -e .`, or use `python -m jev` |
