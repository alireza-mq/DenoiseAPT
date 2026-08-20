"""End-to-end API smoke test against a running local server."""

from __future__ import annotations

import argparse
import json
import urllib.request


def request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument(
        "--case",
        help="Prepared case id to exercise; by default prefer a non-synthetic case.",
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    health = request(f"{base}/api/health")
    assert health["status"] == "ok" and health["automatic_runtime_ready"]
    catalog = request(f"{base}/api/cases")
    cases = catalog["cases"]
    assert cases, "No prepared cases were exposed."
    if args.case:
        preferred = next((case for case in cases if case["id"] == args.case), None)
        if preferred is None:
            available = ", ".join(case["id"] for case in cases)
            raise SystemExit(f"Unknown case {args.case!r}. Available cases: {available}")
    else:
        default_case_id = catalog.get("default_case_id")
        preferred = next(
            (case for case in cases if case["id"] == default_case_id),
            next(
                (
                    case
                    for case in cases
                    if not str(case.get("domain", "")).lower().startswith("synthetic")
                ),
                cases[0],
            ),
        )

    corruption = (
        {
            "family": preferred["default_family"],
            "severity": preferred["default_severity"],
            "replicate": preferred["default_replicate"],
        }
        if preferred.get("benchmark_replay")
        else {"family": "impulse", "severity": 0.35, "seed": 17}
    )

    analysis = request(
        f"{base}/api/analyze",
        "POST",
        {
            "case_id": preferred["id"],
            "corruption": corruption,
            "window": {"start": 0, "length": min(512, preferred["length"])},
        },
    )
    n = len(analysis["series"]["observed"])
    assert n >= 64
    if preferred.get("benchmark_replay"):
        assert set(analysis["series"]) == {
            "reference",
            "observed",
            "median",
            "wavelet",
            "noisereduce",
            "rins_t",
            "our_model",
        }
        assert analysis["automatic_control"]["mode"] == "heldout_benchmark_replay"
        assert analysis["automatic_control"]["certificate"]["passed"] is True
        start, end = analysis["meta"]["suggested_expert_interval"]
        intervention = request(
            f"{base}/api/intervene",
            "POST",
            {
                "session_id": analysis["session_id"],
                "action": "blend",
                "start": start,
                "end": end,
                "beta": 0.75,
                "expected_revision": analysis["revision"],
            },
        )
        assert len(intervention["series"]["our_model"]) == n
        assert intervention["history_depth"] == 1
        restored = request(
            f"{base}/api/intervene",
            "POST",
            {
                "session_id": analysis["session_id"],
                "action": "restore_automatic",
                "expected_revision": intervention["revision"],
            },
        )
        assert restored["series"]["our_model"] == analysis["series"]["our_model"]
        revert = request(
            f"{base}/api/intervene",
            "POST",
            {
                "session_id": analysis["session_id"],
                "action": "revert",
                "expected_revision": restored["revision"],
            },
        )
        assert revert["series"]["our_model"] == intervention["series"]["our_model"]
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "case": preferred["name"],
                    "time_points": n,
                    "session": analysis["session_id"],
                    "controller_mode": analysis["automatic_control"]["mode"],
                    "intervention": "blend, restore automatic, then revert",
                },
                indent=2,
            )
        )
        return
    for key in (
        "classical",
        "cgan",
        "denoiseapt",
        "automatic",
        "hybrid",
        "approved",
    ):
        assert len(analysis["series"][key]) == n
    assert analysis["series"]["approved"] == analysis["series"]["automatic"]
    assert analysis["automatic_control"]["mode"] in {
        "witness_certificate",
        "review_only",
    }
    assert analysis["hybrid_control"]["mode"] in {
        "witness_certificate",
        "review_only",
    }
    assert isinstance(analysis["hybrid_control"]["auto_committed"], bool)
    assert analysis["metrics"]["hybrid"]["latency_ms"] >= 0
    assert analysis["metrics"]["hybrid"]["latency_ms"] >= analysis[
        "hybrid_control"
    ]["routing_latency_ms"]
    if preferred.get("automatic_certificate_available"):
        assert analysis["automatic_control"]["certification_eligible"] is True
        assert isinstance(analysis["automatic_control"]["certificate"]["passed"], bool)
        assert analysis["automatic_control"]["certificate"]["passed"] is True
        assert analysis["automatic_control"]["repair_intervals"], (
            "The reproducible guided profile should exercise automatic repair."
        )
    else:
        assert analysis["automatic_control"]["mode"] == "review_only"
        assert analysis["automatic_control"]["auto_committed"] is False
    assert len(analysis["concern"]["values"]) == n
    assert analysis["metrics"]["denoiseapt"]["latency_ms"] >= 0

    interval = analysis.get("anomaly_intervals") or [{"start": n // 3, "end": n // 2}]
    start, end = interval[0]["start"], interval[0]["end"]
    intervention = request(
        f"{base}/api/intervene",
        "POST",
        {
            "session_id": analysis["session_id"],
            "action": "blend",
            "start": start,
            "end": end,
            "beta": 0.65,
            "expected_revision": analysis["revision"],
        },
    )
    assert len(intervention["series"]["approved"]) == n
    assert intervention["history_depth"] >= 1
    restored = request(
        f"{base}/api/intervene",
        "POST",
        {
            "session_id": analysis["session_id"],
            "action": "restore_automatic",
            "expected_revision": intervention["revision"],
        },
    )
    assert restored["series"]["approved"] == analysis["series"]["automatic"]
    revert = request(
        f"{base}/api/intervene",
        "POST",
        {
            "session_id": analysis["session_id"],
            "action": "revert",
            "expected_revision": restored["revision"],
        },
    )
    assert revert["series"]["approved"] == intervention["series"]["approved"]
    print(
        json.dumps(
            {
                "status": "PASS",
                "case": preferred["name"],
                "samples": n,
                "session": analysis["session_id"],
                "soft_generator_latency_ms": analysis["metrics"]["denoiseapt"]["latency_ms"],
                "controller_mode": analysis["automatic_control"]["mode"],
                "controller_decision": analysis["automatic_control"]["decision"],
                "hybrid_mode": analysis["hybrid_control"]["mode"],
                "hybrid_decision": analysis["hybrid_control"]["decision"],
                "intervention": "blend, restore automatic, then revert",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
