"""Run from the repository root: python -m examples.triage --backend typesafe."""
import argparse
import json
from jev import DecisionClient

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Invoices and payments", "technical": "Product faults"}},
    "impact": {"type": "score", "instructions": "Rate the reported operational impact.",
               "criteria": ["No disruption", "Work slowed", "Work blocked"]},
    "refund": {"type": "noul", "instructions": "Does the sender explicitly request a refund?"},
}


def route(answers, threshold=0.8):
    """Keep confidence separate from probability; thresholds need task evaluation."""
    team = answers["team"]
    return team["choice"] if team["confidence"] >= threshold else "human_review"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["typesafe", "local"], default="typesafe")
    parser.add_argument("--model")
    parser.add_argument("--state", default="Our account was billed twice. Please refund the duplicate payment.")
    args = parser.parse_args()
    client = DecisionClient(args.backend, model=args.model)
    result = client.decide(state=args.state, questions=QUESTIONS)
    result["workflow"] = {"route": route(result["answers"]), "threshold": 0.8}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
