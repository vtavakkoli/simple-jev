# Roadmap

This is a prioritized work list, not a promise of delivery dates or parity with TypeSafe's model.

| Priority | Work | Acceptance evidence |
| --- | --- | --- |
| Now | Run authenticated JEV smoke test | Redacted request metadata, actual model ID and successful mixed-question response |
| Now | Verify browser layout | Desktop/mobile screenshots, navigation and keyboard checks |
| Now | Clarify repository licensing | Explicit upstream license grant or documented permission |
| Next | Reproducible task evaluation | Versioned held-out inputs, model/revision, latency distribution and task metrics |
| Next | Direct CarRacing evaluation | Multiple seeds; raw model decisions; distinguish assisted mode; report failures |
| Next | Confidence threshold evaluation | Held-out coverage/error curves, separate results for each backend |
| Later | Consider async and batch client APIs | Rate-limit behavior and cancellation tests; measured benefit |
| Later | Review upstream backend changes | Focused compatibility review and regression results before importing |

Current implemented features are described in the [README](../README.md) and [changelog](../CHANGELOG.md). Actual TypeSafe weights, proprietary architecture and RLCD training are outside the scope of this integration toolkit.
