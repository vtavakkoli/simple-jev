# Simple-JEV HTTP API reference

This reference describes the standalone Hugging Face implementation in
`hf_server.py`, version 0.1.0. It does not require vLLM. The API evaluates many
questions against one context and returns JSON in one non-streaming response.
It reads selected next-token logits; it does not generate prose answers.

## Endpoints and transport

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/v1/classifier` | Score the supplied questions. |
| POST | `/v1/systemone` | Exact alias of `/v1/classifier`; omitted from generated OpenAPI. |
| GET | `/health` | Returns `{"status":"ready","model":"<loaded model>"}` after service initialization. This does not run an inference probe. |
| GET | `/docs` | Interactive Swagger documentation. |
| GET | `/redoc` | Generated ReDoc documentation. |
| GET | `/openapi.json` | Generated request schema and route definitions. |

Send `Content-Type: application/json`. There are no classifier query parameters
or required custom headers. This standalone server implements no authentication,
API-key management or TLS itself. It has no chat-completions endpoint; `messages`
is an alternative way to supply classifier context.

The generated OpenAPI covers request schemas, but the handwritten response,
error, ignored-field and runtime-limit details below are more complete.

## Quick start

After installation, start the server with a model that supports the reference's
cache and tokenizer requirements:

```bash
simple-jev --model Qwen/Qwen3.5-2B --host 0.0.0.0 --port 8000
```

Use that same model identifier in requests:

```bash
curl --fail-with-body http://localhost:8000/v1/classifier \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3.5-2B",
  "state": "Mia owns a red bicycle. Her dog is named Max.",
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "support": {
      "type": "score",
      "instructions": "How strongly does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Partially supported", "Fully supported"]
    },
    "dog": {
      "type": "noul",
      "instructions": "Is Mia's dog named Max?"
    }
  }
}
JSON
```

Illustrative response (probabilities and usage are examples, not measured output):

```json
{
  "model": "Qwen/Qwen3.5-2B",
  "answers": {
    "color": {
      "type": "choice",
      "choice": "red",
      "confidence": 0.97,
      "probabilities": {"red": 0.97, "blue": 0.03}
    },
    "support": {
      "type": "score",
      "score": 1.9,
      "confidence": 0.92,
      "probabilities": {"0": 0.02, "1": 0.06, "2": 0.92},
      "legend": {"0": "Unsupported", "1": "Partially supported", "2": "Fully supported"}
    },
    "dog": {"type": "noul", "noul": 0.96}
  },
  "usage": {"input_tokens": 600, "output_tokens": 0}
}
```

## Request body

Unknown top-level fields are ignored. Unknown question/option fields are rejected. Field names are case-sensitive. Examples use JSON strings and
numbers; clients should send the documented types rather than rely on Pydantic
coercion.

| Field | Type | Required/default | Behavior |
| --- | --- | --- | --- |
| `model` | string | Required; nonempty | Must match the model ID or local path used to start this server. The HTTP request does not load or switch models. |
| `state` | string, object, array, or null | Supply exactly one non-null `state` or `messages` | Shared context. Objects/arrays are serialized into prompt text; they are not executable state. A top-level number or boolean is not supported. |
| `messages` | array of messages or null | Alternative to `state`; at least one message | Text chat history rendered with the model's chat template. |
| `questions` | object mapping IDs to questions | Required; 1–256 entries at schema level | IDs must be nonempty strings. The server's branch limit is additionally enforced, default 100. |
| `options` | object | Defaults shown below | Response diagnostics; prompt/scoring rules are fixed by v1. |
| `tools` | array of objects or null | Omitted/null | Reserved in the schema; nonempty values are rejected by the HF implementation. |
| `mm_processor_kwargs` | object or null | Omitted/null | Reserved; nonempty values are rejected. |
| `media_io_kwargs` | object of objects or null | Omitted/null | Reserved; nonempty values are rejected. |

Empty reserved containers are accepted but have no effect. Omit them normally.
Explicit `null` does not count as supplied context. An empty string or empty JSON
object/array does count as `state`. Supplying both non-null context fields, or
neither, returns 422.

### Chat messages

For this HF implementation each message contains only:

| Field | Supported value |
| --- | --- |
| `role` | `system`, `developer`, `user`, or `assistant` |
| `content` | String, including an empty string |

The shared schema also describes `tool`/`function` roles, null content, content
part arrays and extra message fields. **The HF compiler rejects these.** Images,
audio, video, tool calls, `name`, and other extra message properties are not
supported. A model's chat template may further restrict roles or their order.

For chat input replace `state` in the example with:

```json
"messages": [
  {"role": "user", "content": "Mia owns a red bicycle."},
  {"role": "assistant", "content": "Her dog is named Max."}
]
```

Classifier instructions and question prompts are assembled around this context
using the default `shared_questions_deliberate` pattern and the tokenizer's
chat template. Clients cannot override that template or pattern through this API.

### Question fields

An **entry** below means a string, JSON object, JSON array, or null. Nested JSON
may contain ordinary JSON scalar values. Bare numbers and booleans are not valid
entries. `instructions` is required even though its value may be null.

| Question type | `instructions` | `criteria` |
| --- | --- | --- |
| `choice` | Required entry describing the question | Required object with 2–50 candidate IDs mapped to entries describing each candidate. Null descriptions are allowed. |
| `score` | Required entry describing what to evaluate | Required ordered array of 2–50 entries, lowest level first. |
| `noul` | Required entry describing a truth/yes-no proposition | Optional object with only `"true"` and/or `"false"` keys mapped to entries; default null. Neither key is required. |

Every question requires `type`, exactly `choice`, `score`, or `noul`. Unknown
question fields are rejected. Candidate IDs are the public choice values. Their
JSON insertion order determines label assignment and breaks exact ties; score
criteria retain array order. Unlike question IDs, empty choice candidate IDs are
not explicitly forbidden by the current schema.

Example truth rubric:

```json
{
  "type": "noul",
  "instructions": "Does the message request a refund?",
  "criteria": {
    "true": "The customer explicitly asks for money back.",
    "false": "The customer makes no refund request."
  }
}
```

### Options — all fields

```json
{"raw_logits": false}
```

`raw_logits` requests selected-token logits, visible only with advanced metrics.
The shared `v1` template fixes choice, score, and Noul behavior. `choice_mode`,
`score_mode`, and `score_format` are rejected. The server selects v1 at the prompt
builder boundary; HTTP request fields cannot override it.

There is no request temperature or sampling step. Softmax uses the selected
logits without temperature scaling. Probabilities are conditional on the
candidate/rating token set, not the entire vocabulary, and are not calibrated
probabilities of correctness.

## Scoring and response fields

Every successful response contains:

| Field | Meaning |
| --- | --- |
| `model` | Request model identifier. |
| `answers` | Object keyed by the supplied question IDs. |
| `usage.input_tokens` | Exact union of token prefixes across the compiled question branches. Shared prefixes count once. Includes classifier instructions, examples, template tokens and suffixes. |
| `usage.output_tokens` | Always 0 for this HF backend: it scores logits without sampling output tokens. |

Usage excludes padding. It is logical unique-prefix accounting, not a measurement
of all actual model work: suffix batches can recompute additional overlap beyond
the common seed prefix. It is not persistent-cache billing across requests.

### Choice

One branch assigns single-token labels `A`–`Z`, then `a`–`x`, for up to 50
candidates. A softmax over candidate logits produces:

| Field | Meaning |
| --- | --- |
| `type` | `choice` |
| `choice` | Candidate ID with the greatest logit. Exact ties select the first candidate. |
| `confidence` | Winning candidate's softmax probability. |
| `probabilities` | Object mapping each candidate ID to its probability; sums approximately to 1. |

### Score

Uses one branch and labels `0`–`9` for up to 10 criteria; above 10 it uses letter
labels internally and maps them back to zero-based criterion indices.

| Field | Meaning |
| --- | --- |
| `type` | `score` |
| `score` | Expected zero-based criterion index: `sum(p[i] * i)`. May be fractional; range 0 to `N-1`. |
| `confidence` | Largest criterion probability, not a confidence interval for the expected score. |
| `probabilities` | Object keyed by numeric strings `"0"` through `"N-1"`, even when internal labels are letters. |
| `legend` | Object mapping those same numeric strings to the original criteria. |

### Noul

Noul always uses one nine-bin rating branch. Convert the expected rating to a
decimal `d` in [0.1, 0.9], then return:

```text
noul = clamp(0.01 + (d - 0.1) * 0.98 / 0.8, 0.01, 0.99)
```

Thus endpoint ratings 0.1 and 0.9 map to 0.01 and 0.99, with midpoint 0.5.
Default fields are `type: "noul"` and `noul`. There is no separate confidence
field. This is a transformed expected rating, not a binary-token softmax.

## Advanced metrics and raw logits

Start the server with `ENABLE_OPEN_JEV_ADVANCED_METRICS=1` to expose these fields.
Values `true`, `yes`, and `on` also enable it, case-insensitively. The environment
variable retains its original name after the Simple-JEV rename. It is read when
the service is constructed, not from each HTTP request.

### Additional answer fields

| Mode | Additional fields |
| --- | --- |
| Direct choice | `margin` (top two probability difference), `ties` (all equal-max-logit IDs), `calibrated: false`, `scoring: "direct_label_logits"`; `logits` keyed by candidate ID if requested. |
| Direct score | `variance` over criterion indices, `calibrated: false`, `scoring: "direct_level_logits"`, `score_mapping: "label_to_zero_based_level"`; `logits` keyed by numeric index strings if requested. |
| Noul | `rating`, `calibrated: false`. |

A Noul `rating` contains `bins` (nine ordered
values), `probabilities` (nine values), `expected_score`, `variance`, and `entropy`
(natural-log units). With `raw_logits: true` it also contains a nine-element
`logits` array. Variance follows the selected integer/decimal units.

### Top-level metadata

| Field | Value/meaning |
| --- | --- |
| `metadata.backend` | `transformers` |
| `metadata.model_revision` | Startup `--revision`, or null. |
| `metadata.template_version` | The resolved shared template version: `v1`. |
| `metadata.calibration` | `not_calibrated` |
| `metadata.usage_accounting` | `unique_token_prefixes_and_engine_leaf_outputs` (legacy identifier). |

### Top-level metrics

| Field | Meaning |
| --- | --- |
| `backend` | `transformers` |
| `prefill_strategy` | `shared_prefix` |
| `prefix_tokens` | Length of the shared prefix actually evaluated once. At least one token is left for each suffix, even for identical prompts. |
| `suffix_batch_sizes` | Number of question/candidate branches in each suffix forward. |
| `engine_forwards` | Prefix forward, if any, plus suffix forwards. These are model calls, not HTTP calls. |
| `branch_prompt_tokens` | Sum of all complete branch lengths, including repeated prefixes. |
| `computed_prompt_tokens` | Prefix tokens actually evaluated for this request plus padded suffix tokens. Persistent-cache hits exclude reused prefix tokens. |
| `computed_prefix_tokens` | Prefix tokens actually evaluated for this request. |
| `logical_prefill_tokens` | Shared prefix length plus unpadded suffix lengths, independent of persistent-cache hits. |
| `padded_suffix_tokens` | Sum of batch size times maximum suffix length for each batch. |
| `persistent_prefix_cache` | `disabled`, `unavailable`, `miss`, or `hit`. |
| `persistent_prefix_tokens` | Exact-token context-independent prefix eligible for persistent reuse. |
| `persistent_prefix_hit_tokens` | Prefix tokens served from a persistent KV entry for this request. |
| `persistent_prefix_cache_entries` | Current number of entries in the backend LRU. |
| `branch_output_tokens` | 0 |
| `scored_positions` | Number of scoring branches. |
| `backend_seconds` | Backend elapsed time inside model lock. |
| `queue_seconds` | Wait for the service's request semaphore. |
| `total_seconds` | Service time through response construction, including queue, compilation and inference; excludes final network transmission. |

## Limits, batching and cancellation

The CLI service runs one model request at a time with up to 16 additional
requests waiting. Questions within a request execute in suffix batches. At
capacity (17 admitted requests), additional requests receive 429.

The default request limit is 100 branches, configurable with
`--max-request-branches`. Every question consumes exactly one branch. The schema
caps questions at 256; choice/score criteria are limited to 50 entries.

Each complete compiled branch, including shared context and appended question
instructions, must fit `--max-model-len`. There is no automatic truncation.
This CLI limit is not automatically clamped to the model's native context limit;
configure it appropriately for the model. Setting it higher does not add model
support for longer sequences.

The backend evaluates the exact common prefix once and copies its Transformers
cache for each suffix batch. Suffixes are sorted by length, batched under
`--max-batch-size` and the padded `--max-batch-tokens` budget, and reordered for
response assembly. A single suffix larger than the token budget returns 422.
The prefix forward itself is not chunked by this budget. Optional
`--prefix-cache-entries N` persists a context-independent exact-token KV seed
across requests. Entries are LRU-bounded, disabled by default, and consume
model-device memory. There is no continuous cross-request batching.

Client disconnects cancel the service task. An in-flight model forward cannot
be immediately interrupted; the backend observes cancellation between forwards
and keeps its model lock until safe to release.

## Errors

| Status | Meaning |
| --- | --- |
| 422 | Invalid JSON/schema, unknown model, invalid context combination, unsupported chat/media/tool input, branch/token limits, or another compiler/backend `ValueError`. |
| 429 | Request queue full; header `Retry-After: 1`, body `{"detail":"Scoring queue is full"}`. |
| 499 | Client disconnected, if a response can still be delivered: `{"detail":"Client disconnected"}`. |
| 500 | Unhandled runtime failure, such as a model execution error. No stable structured error body is guaranteed. |

Example semantic validation error:

```json
{
  "error": {
    "message": "Loaded model is 'Qwen/Qwen3.5-2B'",
    "type": "invalid_request_error",
    "code": 422,
    "param": null,
    "details": []
  }
}
```

Schema errors use the same envelope, with up to ten detail entries containing
`param`, `message`, and `type`. `param` identifies a dotted field path, with array
indices such as `[0]`; union type names can appear in paths. The top-level `param`
is the first detail's path. Extra errors are counted in the summary message.
Semantic errors may have no field path or details. Validation detail entries omit
submitted input values and exception contexts.

## Unknown top-level fields

All undeclared top-level fields are ignored, including completion settings and
custom client fields. For example, `stream: true` still returns ordinary JSON,
and `max_tokens` does not change the number of questions scored. Declared fields
remain validated; misspelled fields inside questions/options are rejected.

## Server startup arguments — exhaustive list

These are process settings, not HTTP request fields. Both `simple-jev` and
`python -m hf_server` accept them.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | HF model ID or local pretrained model directory. Also the accepted request `model` string. |
| `--revision` | Unset | HF revision passed to tokenizer, config and model loading. |
| `--device` | `auto` | Passed as Transformers `device_map`; examples: `auto`, `cpu`, `cuda:0`. ROCm PyTorch also uses CUDA device naming. |
| `--dtype` | `bfloat16` | One of `float32`, `float16`, `bfloat16`. |
| `--max-model-len` | `16384` | Maximum token length of each compiled branch. |
| `--max-batch-size` | `32` | Maximum suffix rows per model forward; must be positive. |
| `--max-batch-tokens` | `32768` | Maximum padded suffix tokens per batch; must be positive. Does not chunk or limit the prefix forward. |
| `--max-request-branches` | `100` | Positive expanded-branch cap per classifier request, subject to schema hard limits. |
| `--prefix-cache-entries` | `0` | Exact context-independent KV prefix entries retained across requests. `0` disables; entries consume model-device memory. |
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | HTTP port. |
| `-h`, `--help` | — | Print argument help and exit. |

No CLI flags are currently provided for authentication, quantization, model
aliases, request queue size, request concurrency, or vision. The service requires
compatible copyable/reorderable Transformers caches and suitable single-token
rating/choice labels; arbitrary HF models are not guaranteed to work.

## Source of truth

- [Shared prompt structure](../common/PROMPT_STRUCTURE_V1.md)
- [Shared prompt builder](../common/prompt_builder.py)
- [Request schema](../common/request_schema.py)
- [HF server: chat rendering, inference, service, HTTP routes, and CLI](hf_server.py)
- [Scoring formulas](../common/response_scoring.py)
