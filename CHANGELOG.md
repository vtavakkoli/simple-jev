# Changelog

Changes are recorded when prepared for review. “Unreleased” does not mean published to PyPI or deployed.

## Unreleased — 0.2.0

### Added

- Installable `jev-lab` client package (Python 3.10+, no runtime dependencies).
- `jev-lab validate` for offline JSON request checks and `jev-lab run` for explicit backend calls.
- File/stdin requests, model override, configurable timeout and JSON result output.
- Runnable support-triage and evidence-check request files.
- Client reference, contributor guide, security reporting guidance and prioritized roadmap.
- Loopback HTTP integration tests and CLI regression tests.
- Python 3.10/3.12/3.13 client CI, wheel smoke checks and repository consistency checks.

### Improved

- Reject malformed questions, non-finite JSON, invalid configuration and header whitespace before requests.
- Preserve HTTP status and backend fields on `DecisionError` for application handling.
- Make timeout semantics, backend limits and installation boundaries explicit.

## 2026-09-22 — JEV integration and visual refresh

- Added official TypeSafe API integration alongside local Simple Jev inference.
- Introduced JEV Lab branding, SVG banner and architecture diagram.
- Reorganized README and notebook navigation; corrected a broken walker link.
- Added hosted quickstart, offline client tests and repository audit.
- Replaced inherited upstream deployment targets with opt-in hosting configuration.
