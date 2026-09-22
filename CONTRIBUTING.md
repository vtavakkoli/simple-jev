# Contributing to JEV Lab

JEV Lab is an independent fork of Featherless Simple Jev. Contributions should make decision workflows easier to run, inspect and evaluate while preserving attribution and the distinction between hosted JEV and local HF inference.

## Development setup

Use Python 3.10+ for the client and Python 3.12+ for the HF server. Node 22 runs the website tests. From the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m unittest discover -s tests -v
node --test website/tests/*.test.mjs
python scripts/check_repository.py
```

Client tests need no API key or model download. Loopback transport tests use an ephemeral local port. For inference changes, also run:

```bash
python -m pip install -e './hf-server[test]'
python -m pytest -c hf-server/pyproject.toml common/tests hf-server/tests -q
```

The root `jev-lab` package installs only `jev/`; the local inference server remains a separate installation. Do not add PyTorch or a model download to the lightweight client install.

## A focused pull request

1. Branch from current `main` and describe the observed problem.
2. Add a minimal fix and meaningful regression coverage for changed behavior.
3. Update runnable examples or reference documentation when the public interface changes.
4. Report the exact checks you ran, plus checks you could not run.
5. Include screenshots for layout changes and measured results for performance claims.

Maintain request/response compatibility where possible. Changes to upstream `common/`, `hf-server/` or `RFDT/` need explicit explanation because these components share a prompt/scoring contract. Never rename a heuristic as JEV inference or report synthetic fixtures as model results.

## Notebooks and secrets

- Put credentials in environment variables or Colab Secrets, never in tracked source or outputs.
- Clear outputs from new notebooks before committing them.
- Describe the backend, installation requirements, billing implications and controller mode near the top.
- Label direct model actions, heuristic supervision and baselines separately.
- Document pinned dependencies and record the model/revision in real experiments.

## Licensing and attribution

This repository currently lacks a root license grant. Do not assume that the fork has relicensed upstream code, images, model weights or datasets. Retain asset-specific attribution; discuss licensing with the maintainers before introducing third-party material. This guide does not establish a contributor license agreement.
