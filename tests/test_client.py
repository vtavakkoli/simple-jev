import copy
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from jev import DecisionClient, DecisionError, validate_answers
from examples.triage import QUESTIONS, route

RESPONSE = {"model": "fixture", "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": .6,
             "probabilities": {"billing": .9, "technical": .1}},
    "impact": {"type": "score", "score": 1.5, "confidence": .8,
               "probabilities": {"0": 0, "1": .5, "2": .5}},
    "refund": {"type": "noul", "noul": 1.0}}}


class ClientTests(unittest.TestCase):
    def test_hosted_contract_and_confidence(self):
        client = DecisionClient(api_key="test-only")
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(RESPONSE).encode())) as call:
            result = client.decide(state={"text": "test"}, questions=QUESTIONS)
        request = call.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only")
        self.assertEqual(json.loads(request.data)["model"], "jev-latest")
        self.assertEqual(result["client_metadata"]["backend"], "typesafe")
        self.assertEqual(route(result["answers"]), "human_review")

    def test_local_does_not_inherit_hosted_key(self):
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "hosted-secret"}):
            client = DecisionClient("local")
        with patch.object(client._opener, "open", return_value=io.BytesIO(json.dumps(RESPONSE).encode())) as call:
            client.decide(state="test", questions=QUESTIONS)
        req = call.call_args.args[0]
        self.assertTrue(req.full_url.endswith("/classifier"))
        self.assertIsNone(req.get_header("Authorization"))

    def test_http_error_never_falls_back_or_leaks(self):
        client = DecisionClient(api_key="test-only")
        with patch.object(client._opener, "open", side_effect=HTTPError("url", 401, "secret", {}, None)) as call:
            with self.assertRaisesRegex(DecisionError, "HTTP 401; no fallback"):
                client.decide(state="test", questions=QUESTIONS)
        self.assertEqual(call.call_count, 1)

    def test_invalid_answers(self):
        for field, value in [("choice", "invented"), ("confidence", float("nan")), ("confidence", True),
                             ("probabilities", {"billing": .1, "technical": .1})]:
            response = copy.deepcopy(RESPONSE)
            response["answers"]["team"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(DecisionError):
                validate_answers(response, QUESTIONS)
        response = copy.deepcopy(RESPONSE)
        del response["answers"]["refund"]
        with self.assertRaises(DecisionError):
            validate_answers(response, QUESTIONS)

    def test_endpoint_and_key_validation(self):
        for kwargs in [{"backend": "unknown"}, {"base_url": "https://example.com", "api_key": "test"},
                       {"backend": "local", "base_url": "http://example.com"}, {"timeout": float("inf")}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecisionClient(**kwargs)
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(ValueError):
            DecisionClient()

    def test_malformed_json(self):
        client = DecisionClient("local")
        with patch.object(client._opener, "open", return_value=io.BytesIO(b"not json")):
            with self.assertRaises(DecisionError):
                client.decide(state="test", questions=QUESTIONS)


if __name__ == "__main__":
    unittest.main()
