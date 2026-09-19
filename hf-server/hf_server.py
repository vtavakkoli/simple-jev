"""Single-file Hugging Face classifier server using the shared common modules.

Run directly from a checkout (the sibling common/ folder is required)::

    python hf-server/hf_server.py --model /path/to/model --device cpu --dtype float32

Or install with pip install -e './hf-server[test]' and run::

    simple-jev --model organization/model --device auto
    python -m hf_server --model organization/model --device auto

Request flow:
    ClassifierRequest -> PromptCompiler -> HFBackend -> common.build_response

common owns validation, versioned classifier wording, label semantics, and answer
math. This file owns text-only role assembly, native chat/tokenizer boundaries,
shared-prefix inference, queue/cancellation controls, usage accounting, HTTP, and
startup. Model weights load only when load_service/main is called.

The sections below follow the data flow and retain the implementation notes for
cache ownership, per-row logit selection, and asynchronous cleanup. Tests live in
tests/; no separate simple_jev package or duplicate classifier template is needed.
"""

import argparse
import asyncio
import copy
import inspect
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

# Direct script execution puts hf-server/, not the checkout root, on sys.path.
# Prefer the sibling common source when running from this repo; an installed
# wheel instead imports its bundled common package through normal resolution.
_checkout_root = Path(__file__).resolve().parent.parent
if (_checkout_root / "common" / "prompt_builder.py").is_file():
    sys.path.insert(0, str(_checkout_root))

from common import (
    ClassifierRequest,
    PromptPlan,
    build_response,
    prepare_prompt,
)
from common.prompt_builder import DEFAULT_TEMPLATE_VERSION, canonical

# Shared plan to native chat and tokens


@dataclass(frozen=True)
class Branch:
    """One compiled question, identified by its plan-local branch ID.

    token_ids encodes the complete prompt through the incomplete assistant answer
    prefix. output_ids contains the next-token vocabulary IDs, in the same order
    as the shared question's output_labels. Messages/prefix remain available for
    inspection; they are not reconstructed from tokens during inference.

    render_only compilation leaves both ID lists empty and is not executable.
    frozen prevents attribute reassignment, not mutation of the contained lists.
    """

    branch_id: str
    token_ids: list[int]
    output_ids: list[int]
    messages: list[dict]
    answer_prefix: str


@dataclass
class CompiledRequest:
    """Bind the semantic prompt plan to its HF-specific executable branches.

    Keep this pair together until response scoring: branch IDs and output order
    must be interpreted against the plan that created them. Branch order initially
    matches plan order; the backend may reorder execution for efficient padding.
    """

    plan: PromptPlan
    branches: list[Branch]
    # Exact token prefix proven independent of caller context. Backends may
    # persist a KV snapshot for this prefix across requests when enabled.
    cache_prefix_ids: list[int] = field(default_factory=list)


def common_prefix(sequences):
    """Return the longest identical token prefix of a nonempty sequence group.

    The caller validates nonempty prompts. An empty shared prefix is valid.
    Comparing actual IDs is essential: matching rendered string fragments alone
    does not prove that the tokenizer produced reusable prefix tokens.
    """
    first = sequences[0]
    end = min(map(len, sequences))
    for other in sequences[1:]:
        for i in range(end):
            if first[i] != other[i]:
                end = i
                break
    return first[:end]


class PromptCompiler:
    """Render shared classifier plans with a model's native tokenizer template."""

    def __init__(self, tokenizer, max_tokens=16384, version=DEFAULT_TEMPLATE_VERSION):
        """Store the renderer, complete-prompt token limit, and shared version.

        Version validation is performed by common.prepare_prompt during compile.
        This class does not load a tokenizer or model on its own.
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.version = version

    @staticmethod
    def _messages(request, system, content, *, context_probe=None):
        """Build one branch, optionally replacing caller context with probe text.

        Probe renderings are used only to locate a conservative token boundary.
        compile() intersects two distinct probes with every real branch before
        exposing a persistent-cache candidate.
        """
        if request.messages is None:
            state = request.state if context_probe is None else context_probe
            return [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"State:\n{canonical(state)}\n\n" + content,
                },
            ]

        messages = []
        for message in request.messages:
            item = message.model_dump(exclude_none=True)
            if context_probe is not None:
                # Preserve roles because chat templates may encode them, while
                # replacing all caller-controlled conversation content.
                item = {"role": item["role"], "content": context_probe}
            messages.append(item)
        if messages[0]["role"] == "system":
            messages[0]["content"] = system + "\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        return messages

    def compile(self, request: ClassifierRequest, *, render_only=False):
        """Validate text input and compile one branch per shared-plan question.

        request may be a ClassifierRequest or an input dictionary. render_only
        returns roles/content and the shared plan for diagnostics/tests; it skips
        chat-template tokenization, token limits, and output-boundary checks.
        Such a result must not be sent to HFBackend.score.

        Unsupported media/tools, malformed token boundaries, and overlong prompts
        raise ValueError. Caller data is preserved when merging system messages.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        # The shared schema allows richer contexts for other integrations. This
        # adapter narrows that contract before constructing text-only messages.
        if request.tools or request.mm_processor_kwargs or request.media_io_kwargs:
            raise ValueError(
                "HF text reference does not support tools or media options"
            )
        if request.messages and any(
            not isinstance(m.content, str)
            or m.model_extra
            or m.role in {"tool", "function"}
            for m in request.messages
        ):
            raise ValueError("HF text reference accepts plain text chat only")

        plan = prepare_prompt(request, version=self.version)
        system = plan.system_prompt_prefix + plan.prefix_instruction
        branches = []
        for question in plan.questions:
            content = plan.suffix_instruction + question.instruction
            # State is JSON-serialized, including quotes around string states.
            # For chat, preserve turn boundaries and merge only a leading system
            # turn; the final selected question is always a new user message.
            messages = self._messages(request, system, content)

            ids, output_ids = [], []
            if not render_only:
                # Render first, then append incomplete JSON to the open assistant
                # position. Do not create a completed assistant message or add
                # a closing brace/EOS before the next-token scoring position.
                text = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    + question.answer_prefix
                )
                # The template already supplies special tokens. Adding another
                # BOS/EOS during encode would alter the intended model input.
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if not ids or len(ids) > self.max_tokens:
                    raise ValueError(
                        f"Branch for {question.question_id!r} must contain 1–{self.max_tokens} tokens"
                    )
                # Derive IDs at the actual rendered boundary, not from isolated
                # label encoding. Check every branch; templates/context can affect it.
                for label in question.output_labels:
                    extended = self.tokenizer.encode(
                        text + label, add_special_tokens=False
                    )
                    if len(extended) != len(ids) + 1 or extended[:-1] != ids:
                        raise ValueError(
                            f"Answer label {label!r} is not single-token stable"
                        )
                    output_ids.append(extended[-1])
                if len(set(output_ids)) != len(output_ids):
                    raise ValueError("Output labels must map to distinct token IDs")
            branches.append(
                Branch(
                    question.branch_id,
                    ids,
                    output_ids,
                    messages,
                    question.answer_prefix,
                )
            )

        cache_prefix_ids = []
        if not render_only and branches:
            # Render two otherwise identical branches with different synthetic
            # contexts. Their token-level intersection cannot depend on the
            # caller's actual content, even when that content itself is empty.
            # Intersect those probes with every real branch as a second safety
            # check. Native chat-template/tokenizer boundaries are therefore
            # part of the proof; no text hash is trusted for reuse.
            first = plan.questions[0]
            candidates = []
            for probe in ("A", "Z"):
                cache_messages = self._messages(
                    request,
                    system,
                    plan.suffix_instruction + first.instruction,
                    context_probe=probe,
                )
                cache_text = (
                    self.tokenizer.apply_chat_template(
                        cache_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    + first.answer_prefix
                )
                candidates.append(
                    self.tokenizer.encode(cache_text, add_special_tokens=False)
                )
            safe_limit = min(len(branch.token_ids) for branch in branches) - 1
            cache_prefix_ids = common_prefix(
                [*candidates, *[branch.token_ids for branch in branches]]
            )[: max(safe_limit, 0)]

        return CompiledRequest(plan, branches, cache_prefix_ids)


# Inference result


@dataclass
class BackendResult:
    """Results keyed by plan-local branch ID, independent of batch execution order.

    HF stores one 1-D CPU float32 tensor per branch, ordered exactly like the
    corresponding shared question's output_labels. The broad Any annotation also
    permits label/float maps from test backends, accepted by common's scorer.

    metrics contains internal timing/token/batch counters. branch_output_tokens
    is zero for HF logits-only inference. It does not supply input usage; the
    service computes the unique token-prefix union from the compiled prompts.
    """

    # Compact tensors in each plan question's output_labels order, keyed by ID.
    logits: dict[str, Any]
    metrics: dict


# Logical input-token accounting


def unique_prompt_tokens(sequences):
    """Return the number of distinct token-tree edges across all sequences.

    Example: [1, 2, 3] and [1, 2, 4] count as four tokens, not six. Duplicate
    paths add nothing; a path that ends inside another adds no new suffix. Empty
    input returns zero. Sorting creates a new list and leaves caller order intact.

    Lexicographic neighbors share the greatest already-counted prefix for each
    next sequence. Subtract that overlap from its length instead of building an
    explicit trie. No global cache or backend warmup information is consulted.
    """
    # Lexicographically adjacent sequences share the maximum previously seen
    # prefix. Count each token-tree edge once, including hierarchical prefixes.
    total = 0
    previous = []
    for sequence in sorted(sequences):
        shared = 0
        for left, right in zip(previous, sequence):
            if left != right:
                break
            shared += 1
        total += len(sequence) - shared
        previous = sequence
    return total


# Shared-prefix model execution


class HFBackend:
    """Own one inference model and serialize all forwards through a thread lock."""

    def __init__(
        self,
        model,
        *,
        max_batch_size=32,
        max_batch_tokens=32768,
        prefix_cache_entries=0,
    ):
        """Set batching bounds and an optional cross-request prefix-cache LRU.

        Persistent caching is disabled by default because every entry retains a
        model KV cache on the model device. Keys are exact token tuples.
        """
        if max_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("Batch limits must be positive")
        if prefix_cache_entries < 0:
            raise ValueError("prefix_cache_entries must be nonnegative")
        self.model = model.eval()
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefix_cache_entries = prefix_cache_entries
        self._prefix_cache = OrderedDict()
        # Protect both model execution and persistent cache mutation.
        self._lock = threading.Lock()
        # Some model classes accept selected sequence positions to avoid
        # materializing all sequence logits. Fall back to full logits otherwise.
        self._last_logits = (
            "logits_to_keep" in inspect.signature(model.forward).parameters
        )

    def _prefix_cache_get(self, token_ids):
        """Return an exact-token seed and promote it in the LRU."""
        if not self.prefix_cache_entries or not token_ids:
            return None
        key = tuple(token_ids)
        cache = self._prefix_cache.pop(key, None)
        if cache is not None:
            self._prefix_cache[key] = cache
        return cache

    def _prefix_cache_put(self, token_ids, cache):
        """Insert one immutable seed and evict least-recently-used entries."""
        if not self.prefix_cache_entries or not token_ids or cache is None:
            return
        key = tuple(token_ids)
        self._prefix_cache.pop(key, None)
        self._prefix_cache[key] = cache
        while len(self._prefix_cache) > self.prefix_cache_entries:
            self._prefix_cache.popitem(last=False)

    async def score(self, compiled):
        """Run blocking model work in a worker thread with cooperative cancellation.

        Cancelling an asyncio task cannot interrupt an in-flight tensor kernel.
        Set a stop flag for the worker to check before its next batch, then let
        cancellation propagate. The model lock remains held until the worker exits.
        """
        stop = threading.Event()
        try:
            return await asyncio.to_thread(self._score, compiled, stop)
        except asyncio.CancelledError:
            stop.set()
            raise

    def _score(self, compiled, stop):
        """Execute one request; all model/cache work stays under the same lock."""
        import torch

        with self._lock, torch.inference_mode():
            if stop.is_set():
                raise asyncio.CancelledError()
            start = time.perf_counter()
            sequences = [b.token_ids for b in compiled.branches]
            if not sequences or any(not ids for ids in sequences):
                raise ValueError("Expected nonempty scoring prompts")
            # Leave at least one suffix token, including for identical prompts.
            prefix = common_prefix(sequences)[: min(map(len, sequences)) - 1]
            # Inputs start on the embedding device; a dispatched/sharded model
            # may move later activations using its own Transformers hooks.
            device = self.model.get_input_embeddings().weight.device
            extra = {"logits_to_keep": 1} if self._last_logits else {}
            cache = None
            forwards = 0
            computed_prefix_tokens = 0
            persistent_status = (
                "disabled" if self.prefix_cache_entries == 0 else "unavailable"
            )
            persistent_hit_tokens = 0

            # Re-check compiler metadata at execution time so custom compilers
            # cannot trigger reuse unless the candidate is an exact prefix of
            # this request's real shared prefix.
            persistent_prefix = list(
                getattr(compiled, "cache_prefix_ids", []) or []
            )
            persistent_safe = (
                bool(prefix)
                and bool(persistent_prefix)
                and len(persistent_prefix) <= len(prefix)
                and prefix[: len(persistent_prefix)] == persistent_prefix
            )

            if prefix and self.prefix_cache_entries and persistent_safe:
                cache = self._prefix_cache_get(persistent_prefix)
                if cache is None:
                    ids = torch.tensor([persistent_prefix], device=device)
                    out = self.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=True,
                        **extra,
                    )
                    cache = out.past_key_values
                    if cache is None or not hasattr(cache, "reorder_cache"):
                        raise ValueError(
                            "Model must expose a reorderable Transformers cache"
                        )
                    self._prefix_cache_put(persistent_prefix, cache)
                    del out
                    forwards += 1
                    computed_prefix_tokens += len(persistent_prefix)
                    persistent_status = "miss"
                else:
                    persistent_status = "hit"
                    persistent_hit_tokens = len(persistent_prefix)

                remainder = prefix[len(persistent_prefix) :]
                if remainder:
                    # Never extend the persistent object itself: keep LRU entries
                    # immutable and request-local mutation on a private copy.
                    request_cache = copy.deepcopy(cache)
                    ids = torch.tensor([remainder], device=device)
                    mask = torch.ones(
                        (1, len(prefix)), dtype=torch.long, device=device
                    )
                    positions = torch.arange(
                        len(persistent_prefix), len(prefix), device=device
                    ).unsqueeze(0)
                    out = self.model(
                        input_ids=ids,
                        attention_mask=mask,
                        position_ids=positions,
                        past_key_values=request_cache,
                        use_cache=True,
                        **extra,
                    )
                    cache = out.past_key_values
                    if cache is None or not hasattr(cache, "reorder_cache"):
                        raise ValueError(
                            "Model must expose a reorderable Transformers cache"
                        )
                    del out, request_cache
                    forwards += 1
                    computed_prefix_tokens += len(remainder)
            elif prefix:
                # Default behavior: one request-local shared-prefix prefill.
                ids = torch.tensor([prefix], device=device)
                out = self.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=True,
                    **extra,
                )
                cache = out.past_key_values
                if cache is None or not hasattr(cache, "reorder_cache"):
                    raise ValueError(
                        "Model must expose a reorderable Transformers cache"
                    )
                del out
                forwards += 1
                computed_prefix_tokens += len(prefix)
            lengths = [len(ids) - len(prefix) for ids in sequences]
            if max(lengths) > self.max_batch_tokens:
                raise ValueError("A question suffix exceeds max_batch_tokens")
            # Longest first reduces padding. Results carry branch IDs, so map
            # insertion/execution order need not equal the original question order.
            pending = sorted(
                range(len(sequences)), key=lambda i: lengths[i], reverse=True
            )
            results = {}
            sizes = []
            padded_tokens = 0
            while pending:
                if stop.is_set():
                    raise asyncio.CancelledError()
                # Every row uses width slots, including right padding. Account
                # for padded work, not just the sum of unpadded suffix lengths.
                width = lengths[pending[0]]
                limit = min(self.max_batch_size, self.max_batch_tokens // width)
                batch, pending = pending[:limit], pending[limit:]
                # Forward mutates its cache. Never let a branch mutate the seed.
                branch_cache = copy.deepcopy(cache)
                if branch_cache is not None:
                    # The seed has one row. Repeated index zero broadcasts that
                    # row to the batch via the cache's supported reorder API.
                    branch_cache.reorder_cache(
                        torch.zeros(len(batch), dtype=torch.long, device=device)
                    )
                # Padding follows the scored position, never enters a real token's
                # causal context, and is excluded from attention. Discard this cache.
                ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
                mask = torch.zeros(
                    (len(batch), len(prefix) + width), dtype=torch.long, device=device
                )
                for row, i in enumerate(batch):
                    ids[row, : lengths[i]] = torch.tensor(
                        sequences[i][len(prefix) :], device=device
                    )
                    mask[row, : len(prefix) + lengths[i]] = 1
                # Position IDs continue after the shared prefix. Attention masks
                # include both prefix and suffix; real tokens cannot attend to
                # right padding. Padded token ID zero is just unused storage.
                positions = (
                    torch.arange(len(prefix), len(prefix) + width, device=device)
                    .unsqueeze(0)
                    .expand(len(batch), -1)
                )
                last = torch.tensor([lengths[i] - 1 for i in batch], device=device)
                # Unequal lengths mean different final positions per row.
                # logits_to_keep accepts one position set shared across rows;
                # inverse maps each row's last position into that returned set.
                keep, inverse = torch.unique(last, sorted=True, return_inverse=True)
                branch_extra = {"logits_to_keep": keep} if self._last_logits else {}
                out = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=branch_cache,
                    use_cache=True,
                    **branch_extra,
                )
                selected = out.logits[
                    torch.arange(len(batch), device=device),
                    inverse if self._last_logits else last,
                ]
                # Gather each branch's permitted tokens before leaving the batch.
                # Compact CPU tensors avoid retaining full vocabulary/GPU buffers.
                for row, i in enumerate(batch):
                    branch = compiled.branches[i]
                    results[branch.branch_id] = (
                        selected[row, branch.output_ids].float().cpu()
                    )
                del out, branch_cache, selected
                forwards += 1
                sizes.append(len(batch))
                padded_tokens += len(batch) * width
            return BackendResult(
                results,
                {
                    "backend": "transformers",
                    "prefill_strategy": "shared_prefix",
                    "prefix_tokens": len(prefix),
                    "suffix_batch_sizes": sizes,
                    "engine_forwards": forwards,
                    # These are distinct accounting views, not interchangeable:
                    # branch_prompt_tokens repeats shared context per branch;
                    # computed_prompt_tokens includes padding but prefixes once;
                    # logical_prefill_tokens omits padding, still counting overlap
                    # beyond the one shared prefix separately per branch.
                    "branch_prompt_tokens": sum(map(len, sequences)),
                    "computed_prompt_tokens": computed_prefix_tokens + padded_tokens,
                    "computed_prefix_tokens": computed_prefix_tokens,
                    "logical_prefill_tokens": len(prefix) + sum(lengths),
                    "padded_suffix_tokens": padded_tokens,
                    "persistent_prefix_cache": persistent_status,
                    "persistent_prefix_tokens": (
                        len(persistent_prefix) if persistent_safe else 0
                    ),
                    "persistent_prefix_hit_tokens": persistent_hit_tokens,
                    "persistent_prefix_cache_entries": len(self._prefix_cache),
                    "branch_output_tokens": 0,
                    "scored_positions": len(sequences),
                    "backend_seconds": time.perf_counter() - start,
                },
            )


# Request admission and response assembly


class OverloadedError(Exception):
    """Admission capacity is exhausted; the HTTP layer translates this to 429."""


class DecisionService:
    """Manage one configured model, its compiler/backend, and request lifecycle.

    Use on one asyncio event loop: the admission counter is intentionally updated
    without awaiting between its capacity check and increment. Compiler/backend
    dependencies allow service tests to run without model weights.
    """

    def __init__(
        self,
        model,
        compiler,
        backend,
        *,
        metadata=None,
        concurrency=4,
        queue_size=16,
        max_request_branches=100,
        model_aliases=(),
        advanced_metrics=None,
    ):
        """Configure admission and diagnostic output for a loaded model.

        concurrency counts active coroutine slots; queue_size adds waiting slots.
        Supply a positive concurrency and nonnegative queue size. model_aliases
        permits additional names for the same loaded model, not dynamic loading.
        advanced_metrics explicitly overrides the environment flag when provided;
        otherwise 1/true/yes/on enable ENABLE_OPEN_JEV_ADVANCED_METRICS.
        """
        if max_request_branches < 1:
            raise ValueError("max_request_branches must be positive")
        self.max_request_branches = max_request_branches
        self.model_aliases = {model, *model_aliases}
        self.model = model
        self.compiler = compiler
        self.backend = backend
        self.metadata = metadata or {}
        self.advanced_metrics = (
            os.environ.get("ENABLE_OPEN_JEV_ADVANCED_METRICS", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if advanced_metrics is None
            else advanced_metrics
        )
        self._semaphore = asyncio.Semaphore(concurrency)
        self._capacity = concurrency + queue_size
        self._inflight = 0

    async def classify(self, request):
        """Return a complete response or raise validation/overload/cancellation.

        Validate and admit before expensive work. Admission is released in finally
        on success, failure, or cancellation. The queue timer ends when the active
        slot is acquired; total_seconds spans admitted work through response build.
        These timings use a monotonic clock and are not distributed trace spans.
        """
        if not isinstance(request, ClassifierRequest):
            request = ClassifierRequest.model_validate(request)
        if request.model not in self.model_aliases:
            raise ValueError(f"Loaded model is {self.model!r}")
        # v1 has exactly one inference branch per question; candidate count no
        # longer expands requests. The shared schema separately caps 256 questions.
        branches = len(request.questions)
        if branches > self.max_request_branches:
            raise ValueError(
                f"Request has {branches} scoring branches; maximum is {self.max_request_branches}"
            )
        if self._inflight >= self._capacity:
            raise OverloadedError("Scoring queue is full")
        # No await between this check/increment pair: other tasks on the same
        # loop cannot interleave admission and oversubscribe the capacity.
        self._inflight += 1
        start = time.perf_counter()
        try:
            async with self._semaphore:
                queued = time.perf_counter() - start
                # Tokenization can be expensive and must not block cancellation/HTTP.
                if hasattr(self.compiler, "compile_async"):
                    compiled = await self.compiler.compile_async(request)
                else:
                    if request.messages and any(
                        not isinstance(m.content, str) or m.role in {"tool", "function"}
                        for m in request.messages
                    ):
                        raise ValueError(
                            "Multimodal/tool chat requires a native renderer; this backend accepts text chat only"
                        )
                    compiled = await asyncio.to_thread(self.compiler.compile, request)
                # The compiler retains the shared plan alongside executable IDs.
                # Never reconstruct label meaning from model output or batch order.
                result = await self.backend.score(compiled)
                response = build_response(
                    compiled.plan,
                    result.logits,
                    # Logical prefix-union accounting excludes batch padding and
                    # repeated context, not a sum of the backend's forward sizes.
                    input_tokens=unique_prompt_tokens(
                        [b.token_ids for b in compiled.branches]
                    ),
                    output_tokens=result.metrics.get("branch_output_tokens", 0),
                    advanced=self.advanced_metrics,
                )
                # Common handles answer filtering and authoritative version data;
                # HF only adds backend-specific metadata and execution timings.
                if self.advanced_metrics:
                    response["metadata"] = {
                        **self.metadata,
                        **response["metadata"],
                        "usage_accounting": "unique_token_prefixes_and_engine_leaf_outputs",
                    }
                    response["metrics"] = {
                        **result.metrics,
                        "queue_seconds": queued,
                        "total_seconds": time.perf_counter() - start,
                    }
                return response
        finally:
            # Release service admission even if a cancelled worker is finishing.
            # HFBackend's lock still prevents overlapping access to the model.
            self._inflight -= 1


# HTTP routes and errors


def validation_response(message=None, errors=()):
    """Format explicit text or Pydantic errors without returning input payloads.

    Keep at most ten structured details and report the count of additional
    errors. Convert location tuples into readable dotted paths with array indices,
    omitting the leading transport-specific 'body' component. The first location
    becomes the top-level param; errors without a location use null.
    """
    details = []
    for error in errors[:10]:
        path = ""
        for part in error.get("loc", ()):
            if part == "body" and not path:
                continue
            if isinstance(part, int):
                path += f"[{part}]"
            else:
                path += ("." if path else "") + str(part)
        details.append(
            {"param": path or None, "message": error["msg"], "type": error["type"]}
        )
    if message is None:
        message = (
            "; ".join(
                f"{e['param']}: {e['message']}" if e["param"] else e["message"]
                for e in details
            )
            or "Invalid classifier request"
        )
        if len(errors) > len(details):
            message += f"; {len(errors) - len(details)} additional validation errors"
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": 422,
                "param": details[0]["param"] if details else None,
                "details": details,
            }
        },
    )


class ClassifierRoute(APIRoute):
    """Normalize errors only for classifier routes, leaving host handlers alone.

    Request parsing can fail before the endpoint function runs, so normalization
    belongs around FastAPI's generated route handler as well as in the endpoint.
    """

    def get_route_handler(self):
        """Wrap FastAPI parsing/execution while preserving non-422 exceptions."""
        handler = super().get_route_handler()

        async def validated(request):
            """Intercept classifier validation errors before they leave this route."""
            try:
                return await handler(request)
            except RequestValidationError as exc:
                # Exclude input values and exception contexts from public errors.
                return validation_response(errors=exc.errors())
            except HTTPException as exc:
                if exc.status_code == 422:
                    return validation_response(message=str(exc.detail))
                raise

        return validated


def create_app(service):
    """Build a standalone app around an already-created service.

    The CLI loads the model before calling this function. Readiness therefore
    reports that configured service, not a new inference probe on every request.
    """
    app = FastAPI(title="Simple-JEV", version="0.1.0")

    @app.get("/health")
    async def health():
        """Return the configured model identifier without invoking inference."""
        return {"status": "ready", "model": service.model}

    attach_routes(app, lambda request: service)
    return app


def attach_routes(app, get_service):
    """Attach classifier endpoints; the alias stays out of generated OpenAPI.

    get_service is synchronous and request-scoped, allowing an embedding app to
    select its service without changing the classifier handler's implementation.
    """
    router = APIRouter(route_class=ClassifierRoute)

    @router.post("/v1/classifier")
    @router.post("/v1/systemone", include_in_schema=False)
    async def classify(body: ClassifierRequest, request: Request):
        """Run one service task and cancel it if the HTTP client disconnects."""
        task = asyncio.create_task(get_service(request).classify(body))
        try:
            # Wait with a timeout rather than awaiting task directly, so a long
            # tokenization/model pass does not prevent disconnect checks.
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "Client disconnected")
            return await task
        except OverloadedError as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ValidationError as exc:
            return validation_response(errors=exc.errors())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            # Always observe the child task's completion/exception. Cancelling
            # its coroutine does not forcibly stop an active model worker thread;
            # HFBackend implements cooperative stopping and a model lock.
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app.include_router(router)


# Model loading and command-line entry point


def load_service(
    model_name,
    *,
    revision=None,
    device="auto",
    dtype="bfloat16",
    max_model_len=16384,
    max_batch_size=32,
    max_batch_tokens=32768,
    max_request_branches=100,
    prefix_cache_entries=0,
):
    """Load a model and return a ready-to-use service, without starting HTTP.

    device is passed to Transformers as device_map; dtype selects a torch dtype.
    max_model_len limits each complete compiled prompt. max_batch_tokens limits
    padded suffix tokens per batch, not shared-prefix prefill or total KV memory.
    max_request_branches caps questions admitted in a single request.

    The loader sets service concurrency to one: separate requests are serialized,
    while branches within a request are batched. The backend's thread lock also
    prevents overlap if cancellation releases admission before a forward ends.
    """
    # Heavy dependencies are local to loading, so CLI help and source inspection
    # do not initialize a model or import the Transformers model classes.
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    config = AutoConfig.from_pretrained(model_name, revision=revision)
    # These checkpoint families use the image/text auto-loader even for text
    # scoring. This selection does not enable image input: the compiler remains
    # text-only and rejects unsupported media/tool requests.
    loader = (
        AutoModelForImageTextToText
        if config.model_type in {"gemma4", "qwen3_5", "qwen3_5_moe"}
        else AutoModelForCausalLM
    )
    model = loader.from_pretrained(
        model_name,
        revision=revision,
        dtype=getattr(torch, dtype),
        device_map=device,
    )
    # PromptCompiler's default comes from common.DEFAULT_TEMPLATE_VERSION.
    # Keep one compiler/backend pair for the service's loaded model/tokenizer.
    compiler = PromptCompiler(tokenizer, max_tokens=max_model_len)
    backend = HFBackend(
        model,
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        prefix_cache_entries=prefix_cache_entries,
    )
    return DecisionService(
        model_name,
        compiler,
        backend,
        concurrency=1,
        max_request_branches=max_request_branches,
        metadata={"backend": "transformers", "model_revision": revision},
    )


def main():
    """Parse process settings, load the service, then run its ASGI application.

    host/port are Uvicorn settings; the other arguments configure loading and
    inference. Model loading happens before the listener starts accepting work.
    --help exits during parsing and therefore does not load any weights.
    """
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--max-request-branches", type=int, default=100)
    parser.add_argument(
        "--prefix-cache-entries",
        type=int,
        default=0,
        help="LRU entries for exact cross-request KV prefixes; 0 disables",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    host, port, model = args.pop("host"), args.pop("port"), args.pop("model")
    uvicorn.run(create_app(load_service(model, **args)), host=host, port=port)


if __name__ == "__main__":
    main()
