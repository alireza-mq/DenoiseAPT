#!/usr/bin/env python3
"""Prepare a labelled reference/observation pair for the demo interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from denoiseapt.datasets import (
    GUIDED_TSB_AD_MEMBER,
    TSB_AD_U_SHA256,
    builtin_demo_case,
    discover_tsb_csv,
    load_csv,
    prepare_case,
    secure_extract_zip,
    sha256_file,
    write_case_npz,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="input CSV")
    source.add_argument("--dataset-root", type=Path, help="select a CSV below this directory")
    source.add_argument(
        "--ensure-demo-case",
        action="store_true",
        help="always build the synthetic NPZ and build the guided TSB-AD NPZ when its archive is cached",
    )
    parser.add_argument(
        "--match",
        default=Path(GUIDED_TSB_AD_MEMBER).name,
        help="filename fragment used with --dataset-root (defaults to the guided UCR Medical case)",
    )
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "data" / "prepared" / "demo_case.npz")
    parser.add_argument("--corruption", choices=["none", "gaussian", "impulse", "drift", "mixed"], default="mixed")
    parser.add_argument("--severity", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-length", type=int, default=512)
    parser.add_argument("--start", type=int)
    return parser


def choose_case(args: argparse.Namespace):
    if args.input:
        return load_csv(args.input)
    if args.dataset_root:
        candidates = discover_tsb_csv(args.dataset_root)
        if args.match:
            needle = args.match.casefold()
            candidates = [path for path in candidates if needle in str(path).casefold()]
        if not candidates:
            raise FileNotFoundError("no matching CSV files found below --dataset-root")
        return load_csv(candidates[0])
    return builtin_demo_case()


def write_selected_case(case, output: Path, args: argparse.Namespace) -> Path:
    selected = prepare_case(
        case,
        corruption=args.corruption,
        severity=args.severity,
        seed=args.seed,
        window_length=args.window_length,
        start=args.start,
    )
    if output.suffix.casefold() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(selected.to_dict(), handle, indent=2)
            handle.write("\n")
    elif output.suffix.casefold() == ".npz":
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(selected.source.metadata)
        metadata.update(
            {
                "name": selected.source.name,
                "prepared_observation_available": True,
                "corruption": selected.corruption,
                "severity": selected.severity,
                "seed": selected.seed,
            }
        )
        np.savez_compressed(
            output,
            signal=selected.source.signal.astype(np.float64),
            observation=selected.observation.astype(np.float64),
            labels=selected.source.labels.astype(np.int8),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    else:
        raise ValueError("--output must end in .npz or .json")
    return output


def ensure_demo_cases(args: argparse.Namespace) -> list[Path]:
    prepared_root = PACKAGE_ROOT / "data" / "prepared"
    prepared_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    # An optional benchmark case is never treated as a redistributable asset.
    # Remove a stale derivative when its verified source archive is absent.
    official_path = prepared_root / "tsb_ad_ucr_medical_guided.npz"

    synthetic = builtin_demo_case()
    synthetic_path = prepared_root / "synthetic_guided_case.npz"
    write_case_npz(
        synthetic,
        synthetic_path,
        {
            "name": "Synthetic Low-Amplitude Anomaly",
            "domain": "Synthetic sensor",
            "source": "DenoiseAPT deterministic packaged fixture",
            "benchmark_case": False,
            "sample_rate": None,
        },
    )
    outputs.append(synthetic_path)

    archive = PACKAGE_ROOT / "data" / "raw" / "TSB-AD-U.zip"
    if archive.exists():
        digest = sha256_file(archive)
        if digest.casefold() != TSB_AD_U_SHA256.casefold():
            raise ValueError(
                f"cached TSB-AD-U archive checksum mismatch: expected {TSB_AD_U_SHA256}, got {digest}"
            )
        extracted_root = PACKAGE_ROOT / "data" / "TSB-AD-U"
        extracted = secure_extract_zip(archive, extracted_root, members=[GUIDED_TSB_AD_MEMBER])
        if len(extracted) != 1:
            raise RuntimeError(f"guided member was not extracted from {archive}")
        official = load_csv(extracted[0])
        selected = prepare_case(
            official,
            corruption="none",
            severity=0.0,
            seed=args.seed,
            window_length=512,
        ).source
        local_intervals = []
        padded = np.pad(selected.labels.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            local_intervals.append([int(begin), int(end)])
        write_case_npz(
            selected,
            official_path,
            {
                "name": "TSB-AD UCR Medical Guided Case",
                "domain": "Medical",
                "source": "TSB-AD-U / UCR",
                "source_file": GUIDED_TSB_AD_MEMBER,
                "archive_member": GUIDED_TSB_AD_MEMBER,
                "archive_sha256": digest,
                "source_samples": len(official.signal),
                "source_labelled_interval_half_open": [4187, 4199],
                "window_labelled_intervals_half_open": local_intervals,
                "benchmark_case": True,
                "license_notice": "TSB-AD lists no upstream license for UCR; consult the original source before redistribution.",
            },
        )
        outputs.append(official_path)
    else:
        official_path.unlink(missing_ok=True)
        print(f"Official archive not cached at {archive}; created the synthetic case only.")
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ensure_demo_case:
        outputs = ensure_demo_cases(args)
        print("Prepared demo cases:")
        for output in outputs:
            print(f"  {output.resolve()}")
        return 0
    output = write_selected_case(choose_case(args), args.output, args)
    print(f"Prepared case at {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
