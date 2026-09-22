"""Exercise real loopback transport and the CLI without external inference."""
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from jev import DecisionClient, DecisionError, validate_request
from jev.cli import main
from test_client import QUESTIONS, RESPONSE


@contextlib.contextmanager
def server(status=200):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers['Content-Length']))),
                          "authorization": self.headers.get("Authorization")})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if status == 302:
                self.send_header("Location", "/unexpected-redirect")
            self.end_headers()
            self.wfile.write(json.dumps(RESPONSE).encode())

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as httpd:
        worker = threading.Thread(target=httpd.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{httpd.server_port}/v1", calls
        finally:
            httpd.shutdown()
            worker.join(timeout=2)


class WorkflowTests(unittest.TestCase):
    def test_real_transport_roundtrip(self):
        with server() as (url, calls):
            result = DecisionClient("local", base_url=url, model="test-model").decide(state="fixture", questions=QUESTIONS)
        self.assertEqual(calls[0]["path"], "/v1/classifier")
        self.assertEqual(calls[0]["body"]["model"], "test-model")
        self.assertEqual(result["answers"], RESPONSE["answers"])
        self.assertGreater(result["client_metadata"]["latency_ms"], 0)

    def test_redirects_and_errors_are_not_retried(self):
        for status in (302, 429, 500):
            with self.subTest(status=status), server(status) as (url, calls):
                with self.assertRaises(DecisionError) as caught:
                    DecisionClient("local", base_url=url).decide(state="fixture", questions=QUESTIONS)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.backend, "local")
                self.assertEqual(len(calls), 1)

    def test_cli_run_writes_json_and_respects_model(self):
        with tempfile.TemporaryDirectory() as tmp, server() as (url, calls):
            request = Path(tmp)/"request.json"
            request.write_text(json.dumps({"model": "file-model", "state": "fixture", "questions": QUESTIONS}))
            output = Path(tmp)/"result.json"
            result = subprocess.run([sys.executable, "-m", "jev", "run", str(request), "--backend", "local",
                                     "--base-url", url, "--model", "override", "--output", str(output)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(json.loads(output.read_text())["answers"], RESPONSE["answers"])
            self.assertEqual(calls[0]["body"]["model"], "override")

    def test_cli_errors_do_not_overwrite_results(self):
        with tempfile.TemporaryDirectory() as tmp, server(500) as (url, calls):
            output = Path(tmp)/"result.json"
            output.write_text("previous result")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                code = main(["run", "examples/requests/support-triage.json", "--backend", "local",
                             "--base-url", url, "--output", str(output)])
            self.assertEqual(code, 3)
            self.assertIn("HTTP 500", err.getvalue())
            self.assertEqual(output.read_text(), "previous result")

    def test_offline_validation_from_stdin(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.stdin", io.StringIO(json.dumps({"state": "fixture", "questions": QUESTIONS}))):
            with patch("jev.client.build_opener", side_effect=AssertionError("No network setup allowed")):
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(main(["validate", "-"]), 0)
        self.assertFalse(json.loads(out.getvalue())["network_used"])

    def test_missing_key_is_actionable(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["run", "examples/requests/support-triage.json", "--backend", "typesafe"]), 2)
        self.assertIn("TYPESAFE_API_KEY", err.getvalue())

    def test_invalid_questions_stop_before_transport(self):
        bad = [None, [], {}, {"": QUESTIONS["refund"]}, {"x": {"type": "noul"}},
               {"x": {"type": "noul", "instructions": "test", "typo": True}},
               {"x": {"type": "noul", "instructions": "test", "criteria": {"yes": "test"}}}]
        client = DecisionClient("local")
        with patch.object(client._opener, "open", side_effect=AssertionError("Must not send")):
            for questions in bad:
                with self.subTest(questions=questions), self.assertRaises(ValueError):
                    client.decide(state="fixture", questions=questions)

    def test_non_json_values_and_cycles_are_rejected(self):
        cyclic = []; cyclic.append(cyclic)
        for state in (None, True, 4, {1: "nonstring key"}, {"number": float("nan")}, [object()], cyclic):
            with self.subTest(type=type(state)), self.assertRaises(ValueError):
                validate_request(state=state, questions=QUESTIONS)

    def test_structured_instructions_are_supported(self):
        questions = copy.deepcopy(QUESTIONS)
        questions["team"]["instructions"] = {"task": "route", "metadata": [1, True, None]}
        validate_request(state={"ticket": "fixture"}, questions=questions)

    def test_config_validation_rejects_bool_timeout_and_header_injection(self):
        for kwargs in ({"timeout": True}, {"api_key": "key\nInjected: value"}, {"model": ""},
                       {"base_url": "http://127.0.0.1:bad/v1"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DecisionClient("local", **kwargs)

    def test_broken_json_is_reported_without_traceback(self):
        with patch("sys.stdin", io.StringIO('{"state": "private-content", bad')), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["validate", "-"]), 2)
        self.assertIn("valid JSON", err.getvalue())
        self.assertNotIn("private-content", err.getvalue())


if __name__ == "__main__":
    unittest.main()
