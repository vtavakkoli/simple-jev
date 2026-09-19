# Simple-JEV

Standalone classifier HTTP API using Hugging Face Transformers and PyTorch.
Classifier validation, prompt text, and response scoring come from the sibling
`common/` folder. Keep both folders in the checkout. No Open-JEV or vLLM runtime
is required. The HF package includes `common` when installed/built from this repo.

## Install and run

Use Python 3.12 or newer. Install the appropriate PyTorch build for your CPU,
CUDA or ROCm environment first, then install this package in that environment:

```bash
cd /path/to/simple-jev
python -m venv .venv
source .venv/bin/activate
# Install your hardware-specific PyTorch build here.
pip install -e './hf-server[test]'
simple-jev --model /path/to/downloaded/model --device auto \
  --dtype bfloat16 --max-model-len 32768 \
  --max-batch-size 32 --max-batch-tokens 32768 --port 8000
```

Or run the file directly from the checkout:

```bash
python hf-server/hf_server.py --model /path/to/model --device cpu --dtype float32
```

After installation, `python -m hf_server` accepts the same arguments.
Choose a context limit supported by the model. CPU testing can use
`--device cpu --dtype float32`. Model IDs from Hugging Face are also accepted;
a local model directory avoids downloading weights again.

## API

See the [complete HTTP API reference](API_REFERENCE.md) for all request fields,
options, response formats, errors, limits, metrics and server arguments.

`POST /v1/classifier` and its alias `POST /v1/systemone` return non-streaming JSON.
`GET /health` reports readiness. `/docs` provides the generated API schema.
The request model must match the name/path passed to `--model`.

```json
{
  "model": "/path/to/downloaded/model",
  "messages": [{"role": "user", "content": "Mia owns a red bicycle. Her dog is named Max."}],
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "What color is Mia's bicycle?",
      "criteria": {"red": null, "blue": null}
    },
    "dog": {"type": "noul", "instructions": "Is the dog named Max?"},
    "support": {
      "type": "score",
      "instructions": "How well does the context support that Mia owns a bicycle?",
      "criteria": ["Unsupported", "Supported"]
    }
  }
}
```

Supply exactly one of `messages` or `state`. State accepts text or JSON.
Chat uses the model tokenizer's chat template. Choice and score support up to
50 entries. The server defaults to 100 scoring branches per request;
`--max-request-branches` configures the limit. The shared v1 template uses exactly one branch per question. Legacy choice/score
modes and score-format switches are no longer accepted.
Invalid input returns readable 422 errors; a full queue returns 429.

Responses contain `model`, `answers`, and `usage`. Answers include confidence.
`usage.input_tokens` counts unique token prefixes once, not the entire shared
context once per question. `usage.output_tokens` is zero: this implementation
reads logits without sampling any output tokens. Set
`ENABLE_OPEN_JEV_ADVANCED_METRICS=1` to include detailed timing and metadata.
Standard completion settings such as `max_tokens` and `temperature` are ignored.

## Shared-prefix execution

The compiler calls `common.prepare_prompt(request, version="v1")`, assembles the
returned strings with state/chat roles, and applies the model chat template.
See [the v1 specification](../common/PROMPT_STRUCTURE_V1.md). Compile each
question, find their exact common token prefix, and run that prefix once with
`use_cache=True`. For each suffix batch, copy the prefix cache and repeat its
rows with the Transformers cache API, then score the question suffixes in
parallel. The seed cache remains unchanged. Results return in request order.

Suffixes are grouped by length, bounded by both batch size and padded suffix
token budget. Each suffix selects its own final logit position. Requests execute
serially against the model; parallelism is within each request.

For repeated classifier schemas, `--prefix-cache-entries N` enables a bounded
LRU of exact-token KV seeds across requests. The compiler renders two distinct
synthetic contexts and intersects both token streams with every real branch, so
the reusable prefix cannot depend on caller context. The backend verifies the
candidate against the real request prefix again before use. Cached seeds are
never extended in place; request-specific work starts from a deep copy.

Persistent caching is disabled by default (`0`) because each entry retains KV
state on the model device. Enable a small bounded cache when repeated requests
share the same classifier schema:

```bash
simple-jev --model Qwen/Qwen3.5-2B --prefix-cache-entries 8
```

The normal request-local shared prefix remains active whether or not the
persistent cache is enabled. The prefix itself is not chunked by
`--max-batch-tokens`.

## Scope and validation

This reference currently accepts **text only**, including text messages.
Images, audio, video and tool calls are rejected. A multimodal model loader does
not imply multimodal input support. Models need a compatible Transformers cache
that supports copying and `reorder_cache`, a chat template, and single-token
rating/choice labels. Arbitrary model compatibility is not guaranteed.

The shared v1 prompt and scoring rules are the source of truth for this server.
It does not claim exact numeric equivalence with another inference engine.
Tests compare reused-cache logits against independent full-prompt forwards for
tiny Qwen3, Qwen3.5, Gemma2 and Gemma4 models, and exercise API validation,
confidence, usage accounting and endpoint aliases. They use random local models,
without downloading weights; they do not measure answer quality.

```bash
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```
