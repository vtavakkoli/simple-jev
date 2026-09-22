"""Fast consistency checks used by CI and contributors."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

for path in [ROOT / "README.md", ROOT / "docs" / "CLIENT.md", ROOT / "CONTRIBUTING.md"]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        errors.append(f"missing or empty required document: {path.relative_to(ROOT)}")

for path in (ROOT / "examples" / "requests").glob("*.json"):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        continue
    if not isinstance(data, dict) or "state" not in data or "questions" not in data:
        errors.append(f"request file lacks state/questions: {path.relative_to(ROOT)}")

# Catch the most common stale notebook and site links without requiring a browser.
for path in [ROOT / "README.md", ROOT / "docs" / "CLIENT.md", ROOT / "docs" / "ROADMAP.md"]:
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if "://" in target or target.startswith("mailto:") or target.startswith("#"):
            continue
        candidate = (path.parent / target).resolve()
        if not candidate.exists():
            errors.append(f"broken documentation link in {path.relative_to(ROOT)}: {target}")

if errors:
    print("Repository checks failed:")
    print("\n".join(f"- {error}" for error in errors))
    sys.exit(1)
print("Repository checks passed")
