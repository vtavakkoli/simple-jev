"""Numerical and HTTP checks using tiny randomly initialized local HF models.

No pretrained weights are downloaded. Cached/batched logits are compared with
independent complete-prompt forwards, with tolerances for floating-point kernel
differences. The families exercise full, sliding, and hybrid attention caches.
These tests establish execution equivalence, not answer quality or throughput.
The module skips when torch or Transformers is absent.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="Install HF optional dependencies",
)


@pytest.mark.parametrize("family", ["qwen3", "gemma2", "qwen3_5", "gemma4"])
async def test_cached_batches_match_independent_full_prompts(family):
    """Compare unequal suffix batches with full forwards and repeat to detect cache mutation."""
    import torch
    from hf_server import Branch, CompiledRequest, HFBackend
    from transformers import (
        Gemma2Config,
        Gemma2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    torch.set_num_threads(2)
    # Seed model initialization so numerical regressions are reproducible.
    torch.manual_seed(7)
    common = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 256,
    }
    if family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**common)).eval()
    elif family == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        model = Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(
                **common,
                linear_key_head_dim=8,
                linear_value_head_dim=8,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                layer_types=["linear_attention", "full_attention"],
                rope_parameters={
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "mrope_section": [1, 1, 2],
                },
            )
        ).eval()
    elif family == "gemma4":
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        model = Gemma4ForCausalLM(
            Gemma4TextConfig(
                **common,
                sliding_window=4,
                hidden_size_per_layer_input=0,
                layer_types=["sliding_attention", "full_attention"],
            )
        ).eval()
    else:
        model = Gemma2ForCausalLM(Gemma2Config(**common, sliding_window=4)).eval()
    # Unequal lengths force sorting, padding, and multiple cache-copy batches.
    sequences = [
        list(range(1, 13)) + suffix
        for suffix in ([20, 21], [22, 23], [24, 25], [26], [27, 28, 29])
    ]
    branches = [
        Branch(str(i), ids, [30, 32, 34], [], "") for i, ids in enumerate(sequences)
    ]
    compiled = CompiledRequest(None, branches)
    backend = HFBackend(model, max_batch_size=2)
    calls = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    result = await backend.score(compiled)
    hook.remove()
    # One prefix forward, then three length-grouped suffix forwards.
    assert calls == [(1, 12), (2, 3), (2, 2), (1, 1)]
    with torch.inference_mode():
        for i, ids in enumerate(sequences):
            logits = result.logits[str(i)]
            expected = model(torch.tensor([ids])).logits[0, -1, [30, 32, 34]]
            np.testing.assert_allclose(logits, expected.numpy(), atol=2e-6, rtol=2e-5)
    # A second call must not inherit any mutated per-request cache state.
    again = await backend.score(compiled)
    np.testing.assert_allclose(
        list(result.logits.values()), list(again.logits.values())
    )
    assert result.metrics["branch_output_tokens"] == 0
    assert result.metrics["logical_prefill_tokens"] == 22
    assert result.metrics["computed_prompt_tokens"] == 23


async def test_persistent_prefix_cache_reuses_exact_context_independent_tokens():
    """Reuse a static KV seed across requests without changing selected logits."""
    import torch
    from hf_server import Branch, CompiledRequest, HFBackend
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    backend = HFBackend(model, prefix_cache_entries=2)
    static = [1, 2, 3, 4, 5]

    first_sequences = [
        static + [10, 11, 20],
        static + [10, 11, 21],
    ]
    second_sequences = [
        static + [12, 13, 22],
        static + [12, 13, 23],
    ]

    def compile_sequences(sequences):
        return CompiledRequest(
            None,
            [
                Branch(str(i), ids, [30, 31], [], "")
                for i, ids in enumerate(sequences)
            ],
            cache_prefix_ids=static,
        )

    calls = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    first = await backend.score(compile_sequences(first_sequences))
    second = await backend.score(compile_sequences(second_sequences))
    hook.remove()

    # Miss: persistent seed + request-local extension + suffix batch.
    # Hit: request-local extension + suffix batch; the static forward disappears.
    assert calls == [(1, 5), (1, 2), (2, 1), (1, 2), (2, 1)]
    assert first.metrics["persistent_prefix_cache"] == "miss"
    assert first.metrics["persistent_prefix_hit_tokens"] == 0
    assert first.metrics["computed_prefix_tokens"] == 7
    assert first.metrics["computed_prompt_tokens"] == 9
    assert second.metrics["persistent_prefix_cache"] == "hit"
    assert second.metrics["persistent_prefix_hit_tokens"] == 5
    assert second.metrics["computed_prefix_tokens"] == 2
    assert second.metrics["computed_prompt_tokens"] == 4
    assert second.metrics["logical_prefill_tokens"] == 9
    assert second.metrics["persistent_prefix_cache_entries"] == 1

    # Cached execution must remain numerically equivalent to a complete prompt.
    with torch.inference_mode():
        for result, sequences in [
            (first, first_sequences),
            (second, second_sequences),
        ]:
            for i, ids in enumerate(sequences):
                expected = model(torch.tensor([ids])).logits[0, -1, [30, 31]]
                np.testing.assert_allclose(
                    result.logits[str(i)],
                    expected.numpy(),
                    atol=2e-6,
                    rtol=2e-5,
                )


async def test_identical_prompts_and_single_branch():
    """Reserve the final scoring position even for identical prompts or a single leaf."""
    from hf_server import Branch, CompiledRequest, HFBackend
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()
    backend = HFBackend(model)
    for count in [1, 3]:
        compiled = CompiledRequest(
            None, [Branch(str(i), [1, 2, 3], [1, 2], [], "") for i in range(count)]
        )
        result = await backend.score(compiled)
        assert result.metrics["prefix_tokens"] == 2
        assert result.metrics["suffix_batch_sizes"] == [count]
        for row in result.logits.values():
            np.testing.assert_allclose(row, result.logits["0"], atol=2e-6, rtol=2e-5)


async def test_hf_http_aliases_and_text_validation():
    """Exercise compiler, real tiny-model backend, shared scoring, and both HTTP routes."""
    import httpx
    import torch
    from hf_server import DecisionService, HFBackend, PromptCompiler, create_app
    from transformers import Qwen3Config, Qwen3ForCausalLM

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(text.encode("ascii"))

        def apply_chat_template(self, messages, **kwargs):
            return "\n".join(m["content"] for m in messages) + "\nAssistant: "

    torch.set_num_threads(2)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=8192,
        )
    ).eval()
    compiler = PromptCompiler(Tokenizer())
    service = DecisionService("test", compiler, HFBackend(model), advanced_metrics=True)
    payload = {
        "model": "test",
        "state": "A red bicycle.",
        "questions": {
            "color": {
                "type": "choice",
                "instructions": "Color?",
                "criteria": {"red": None, "blue": None},
            },
            "yes": {"type": "noul", "instructions": "Red?"},
            "level": {
                "type": "score",
                "instructions": "Red?",
                "criteria": ["no", "yes"],
            },
        },
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test"
    ) as client:
        for route in ["/v1/classifier", "/v1/systemone"]:
            response = await client.post(route, json=payload)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["usage"]["output_tokens"] == 0
            assert body["metrics"]["scored_positions"] == 3
            assert "confidence" in body["answers"]["level"]
            assert "confidence" in body["answers"]["color"]
            assert 0.01 <= body["answers"]["yes"]["noul"] <= 0.99
        payload["tools"] = [{"type": "function"}]
        assert (await client.post("/v1/classifier", json=payload)).status_code == 422
