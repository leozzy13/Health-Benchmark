#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wait for an OpenAI-compatible server to become ready.")
    parser.add_argument("--base-url", required=True, help="Base URL such as http://127.0.0.1:8000/v1")
    parser.add_argument("--expected-model", help="Optional exact model id expected from /models.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    return parser


def _matches_expected_model(payload: dict[str, object], expected_model: str | None) -> bool:
    if not expected_model:
        return True
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        if isinstance(model_id, str) and model_id == expected_model:
            return True
    return False


def _is_ready(base_url: str, expected_model: str | None = None) -> bool:
    for suffix in ("/models", "/health"):
        url = f"{base_url.rstrip('/')}{suffix}"
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status != 200:
                    continue
                if suffix == "/models":
                    payload = json.loads(response.read().decode("utf-8"))
                    return isinstance(payload, dict) and "data" in payload and _matches_expected_model(
                        payload,
                        expected_model,
                    )
                return True
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            json.JSONDecodeError,
            http.client.BadStatusLine,
            ConnectionError,
            OSError,
        ):
            continue
    return False


def main() -> int:
    args = build_parser().parse_args()
    deadline = time.time() + int(args.timeout_seconds)
    while time.time() < deadline:
        if _is_ready(args.base_url, args.expected_model):
            return 0
        time.sleep(max(1, int(args.sleep_seconds)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
