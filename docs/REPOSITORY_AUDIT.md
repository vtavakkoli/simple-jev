# Repository audit — 22 September 2026

## Findings addressed

| Finding | Change |
| --- | --- |
| Homepage claimed a full JEV implementation while the local server uses HF logits | Describe the actual backends and add an official TypeSafe integration path |
| Hosted API usage confined to specialized control notebooks | Add `jev.DecisionClient`, mixed-question CLI workflow and a short hosted notebook |
| No shared guard against incomplete/invalid decisions in new workflows | Validate answer coverage, types, finite ranges, choice membership and distributions |
| Provider confidence can differ from maximum option probability | Preserve both; demonstrate review routing based on provider confidence |
| Broken README link to a missing Qwen walker notebook | Link the actual hosted walker and describe its heuristic supervision honestly |
| Existing hosted CarRacing notebook was hard to discover | Add an engine-labelled notebook catalog and website guide |
| Fork homepage and source navigation presented upstream identity as its own | New SVG identity, original architecture diagram and explicit upstream attribution |
| Inherited deploy workflow targeted upstream production configuration | Require explicit enablement and repository-specific Cloudflare variables |
| Website/client regressions were not in PR framework checks | Add lightweight Python client and website tests to CI |

## Still missing / not claimed

- TypeSafe model weights, internal architecture and RLCD training are not supplied by this repository. The real JEV path is an authenticated provider API integration.
- This change does not demonstrate live TypeSafe inference: no account key was supplied for testing. Transport fixtures and validation tests are offline.
- No local pretrained model benchmark was run. Confidence calibration, task accuracy, latency percentiles and control quality require real evaluation.
- BipedalWalker is a gait-speed supervisor over a heuristic; it is not generic model-driven joint control. CarRacing's direct mode requires live evaluation.
- Upstream currently includes additional work, including a Laya backend, not imported wholesale here. This update does not claim upstream feature parity.
- There is no root LICENSE file. Licensing needs upstream clarification; no license has been invented for inherited content.
- The public website playground remains the Featherless API demo. It does not accept TypeSafe keys or impersonate the official JEV service.

## Design and compatibility

Existing `/v1/classifier` and `/v1/systemone` local server behavior is unchanged. The new client is a checkout-level standard-library module; it does not add dependencies to the HF inference server. New visual assets are editable SVG. Original upstream images remain available for historical links. Deployment is not performed by this update.
