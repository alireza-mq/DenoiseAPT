#!/usr/bin/env python3
"""Download a traceable subset of TSB-AD-U or create the offline fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import sys
from urllib.error import URLError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from denoiseapt.datasets import (
    GUIDED_TSB_AD_MEMBER,
    TSB_AD_U_URL,
    TSB_AD_U_SHA256,
    builtin_demo_case,
    download_tsb_ad_u,
    write_case_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=PACKAGE_ROOT / "data" / "TSB-AD-U")
    parser.add_argument("--cache-dir", type=Path, default=PACKAGE_ROOT / "data" / "raw")
    parser.add_argument(
        "--subset",
        action="append",
        default=None,
        help="case-insensitive path fragment to retain; repeat for multiple domains",
    )
    parser.add_argument("--max-files", type=int, default=1, help="maximum CSVs to extract (default: 1)")
    parser.add_argument("--url", default=TSB_AD_U_URL)
    parser.add_argument(
        "--sha256",
        default=TSB_AD_U_SHA256,
        help="trusted expected archive SHA-256 (defaults to the pinned 2026-05-10 release)",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--allow-insecure",
        action="store_true",
        help="explicitly disable TLS verification; unsafe, use only with trusted SHA-256 verification",
    )
    parser.add_argument(
        "--offline-fallback",
        action="store_true",
        help="write the deterministic synthetic fixture if the network download fails",
    )
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="skip the network and write only the deterministic synthetic fixture",
    )
    return parser


def write_fallback(destination: Path, reason: str) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    fixture_path = write_case_csv(builtin_demo_case(), destination / "builtin_demo_case.csv")
    provenance = {
        "dataset": "DenoiseAPT deterministic synthetic fixture",
        "synthetic": True,
        "path": str(fixture_path.resolve()),
        "fallback_reason": reason,
        "notice": "This fixture supports smoke tests only and must not be reported as a TSB-AD result.",
    }
    with (destination / "PROVENANCE.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    return provenance


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture_only:
        print(json.dumps(write_fallback(args.destination, "fixture-only mode"), indent=2))
        return 0
    try:
        provenance = download_tsb_ad_u(
            args.destination,
            cache_dir=args.cache_dir,
            subset=args.subset or [GUIDED_TSB_AD_MEMBER],
            max_files=args.max_files,
            url=args.url,
            expected_sha256=args.sha256,
            force_download=args.force_download,
            allow_insecure=args.allow_insecure,
        )
    except (OSError, ValueError, URLError, ssl.SSLError) as exc:
        if args.offline_fallback:
            provenance = write_fallback(args.destination, f"{type(exc).__name__}: {exc}")
        else:
            print(f"Dataset acquisition failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                print(
                    "The host certificate could not be verified. Do not disable verification silently. "
                    "You may explicitly pass --allow-insecure together with a trusted --sha256, "
                    "or use --offline-fallback for a smoke test.",
                    file=sys.stderr,
                )
            return 2
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
