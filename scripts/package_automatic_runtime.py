#!/usr/bin/env python3
"""Materialize the live automatic-preservation runtime bundle.

Only protocol-v1 model artifacts and the development-frozen controller
configuration are copied.  Protocol-v2 confirmation inputs and results are
neither read nor packaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs" / "protocol_v1_controlled"
DEFAULT_CONTROLLER = (
    ROOT
    / "runs"
    / "protocol_v2_controller_benchmark"
    / "frozen_controller_config.json"
)
DEFAULT_OUTPUT = ROOT / "checkpoints" / "automatic_preservation"
SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_source_hash(
    run: Path, manifest: Mapping[str, Any], relative: str
) -> Path:
    normalized = relative.replace("\\", "/")
    path = run / Path(normalized)
    expected = str(manifest["artifact_hashes"].get(normalized, ""))
    actual = sha256(path)
    if not expected or expected != actual:
        raise ValueError(f"Protocol-v1 source hash mismatch for {normalized}.")
    return path


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--controller-artifact", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.protocol_run.resolve()
    controller_source = args.controller_artifact.resolve()
    output = args.output.resolve()
    if output == run or run in output.parents:
        raise ValueError("Runtime output must not modify the sealed protocol-v1 run.")
    if output == controller_source.parent or controller_source.parent in output.parents:
        raise ValueError("Runtime output must not modify the controller development run.")

    protocol = read_json(run / "protocol_manifest.json")
    if protocol.get("status") != "complete" or protocol.get("protocol_version") != "1.0":
        raise ValueError("Expected the completed protocol-v1 controlled run.")
    selection = dict(protocol.get("validation_selection") or {})
    selected_model = str(selection.get("selected_model") or "")
    if not selected_model:
        raise ValueError("Protocol-v1 manifest has no validation-selected model.")
    seed_record = dict(protocol["seed_records"][str(SEED)])
    selected_relative = str(seed_record["checkpoints"][selected_model]["file"])
    ordinary_relative = str(seed_record["checkpoints"]["ordinary_cgan"]["file"])
    detector_relative = str(seed_record["detector_checkpoint"]["file"])
    selected_source = validate_source_hash(run, protocol, selected_relative)
    ordinary_source = validate_source_hash(run, protocol, ordinary_relative)
    detector_source = validate_source_hash(run, protocol, detector_relative)

    controller = read_json(controller_source)
    development = dict(controller.get("development_metadata") or {})
    if development.get("development_only") is not True:
        raise ValueError("Controller artifact is not explicitly development-frozen.")
    if controller.get("confirmation_results") is not None:
        raise ValueError("Refusing to package a controller artifact containing confirmation results.")
    if development.get("base_model") != selected_model:
        raise ValueError("Controller base model does not match protocol-v1 selection.")

    destinations = {
        "soft_generator": output / "seed17_denoiseapt_soft.pt",
        "ordinary_generator": output / "seed17_ordinary_cgan.pt",
        "detectors": output / "seed17_detectors_A_B.pt",
        "controller": output / "frozen_controller_config.json",
    }
    for source, key in (
        (selected_source, "soft_generator"),
        (ordinary_source, "ordinary_generator"),
        (detector_source, "detectors"),
        (controller_source, "controller"),
    ):
        atomic_copy(source, destinations[key])

    guided_case_path = ROOT / "data" / "prepared" / "tsb_ad_ucr_medical_guided.npz"
    if not guided_case_path.is_file():
        raise FileNotFoundError(
            "The allowlisted UCR-Medical guided case must be prepared before packaging."
        )
    # The explicit source identity is stable and already documented by the
    # prepared-case data card.  The helper intentionally does not open any
    # protocol-v2 data artifact to derive this allowlist.
    eligible_source = "TSB-AD-U/442_UCR_id_140_Medical_tr_1875_1st_4187.csv"
    runtime_manifest = {
        "schema_version": 1,
        "seed": SEED,
        "selected_model": selected_model,
        "selection_source": "protocol-v1 validation_selection; detector A event-evidence distortion with RMSE non-inferiority",
        "source_protocol_manifest_sha256": sha256(run / "protocol_manifest.json"),
        "source_validation_selection_sha256": str(
            protocol.get("validation_selection_sha256", "")
        ),
        "files": {
            key: {
                "path": path.name,
                "sha256": sha256(path),
            }
            for key, path in destinations.items()
        },
        "detector_state_sha256": {
            "A_causal_mlp": str(seed_record["scorer_A_state_sha256"]),
            "B_causal_conv": str(seed_record["scorer_B_state_sha256"]),
        },
        "thresholds": {
            key: float(seed_record["thresholds"][key])
            for key in ("A_causal_mlp", "B_causal_conv")
        },
        "threshold_scope": {
            "threshold_source": "protocol-v1 clean validation reference normal timestamps",
            "calibration_split": "protocol-v1 validation",
            "calibration_domain": "TSB-AD-U UCR Medical",
            "normalization": "per-window observation median and max(IQR/1.349, standard deviation, 1e-4)",
            "window_length": 512,
            "eligible_case_ids": ["tsb_ad_ucr_medical_guided"],
            "eligible_source_files": [eligible_source],
            "scope_note": "Witness-bound A/B threshold provenance only; not physical anomaly truth or unseen-domain validity.",
        },
        "prepared_case": {
            "case_id": "tsb_ad_ucr_medical_guided",
            "distribution": "download-prepared optional benchmark input; not bundled in the release ZIP",
            "path": str(guided_case_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(guided_case_path),
        },
        "controller": {
            "artifact_sha256": str(controller["artifact_sha256"]),
            "controller_config_sha256": str(controller["controller_config_sha256"]),
            "development_only": True,
            "confirmation_results_packaged": False,
        },
        "prepared_case_bundled": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "runtime_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    temporary.replace(manifest_path)
    print(f"PASS: packaged automatic-preservation runtime at {output}")
    print(f"Selected soft model: {selected_model}; seed: {SEED}")
    print("Confirmation artifacts accessed: no")


if __name__ == "__main__":
    main()
