"""Live inference adapter for the frozen automatic-preservation artifacts.

The adapter deliberately sits outside :mod:`denoiseapt.evidence_controller`.
It loads the protocol-v1 seed-17 generators and detector witnesses, validates
their packaged hashes, performs one deterministic normalized forward pass, and
then invokes the frozen evidence controller only when the request is within the
declared threshold-calibration scope.

The returned certificate is witness-bound.  It is not a statement that an
event is physically genuine, detector independent, or deployment safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch

from .concern import ConcernConfig, ConcernCues, compute_concern_cues
from .evidence_controller import (
    ControlledSignalSession,
    EvidencePreservationController,
    ProjectionResult,
    VerificationResult,
    WitnessSpec,
    array_sha256,
    load_frozen_config,
)
from .experiment_models import (
    CausalConvConfig,
    CausalConvScorer,
    state_dict_sha256,
    torch_detector_scores,
)
from .inference import WindowNormalization
from .hybrid import (
    HybridCandidateResult,
    LIVE_HYBRID_CONFIG,
    build_evidence_gated_candidate,
    reflect_moving_average,
)
from .models import (
    CausalForecasterScorer,
    ForecasterConfig,
    GeneratorConfig,
    TemporalUNetGenerator,
)


RUNTIME_SCHEMA_VERSION = 1
DEFAULT_ARTIFACT_DIRECTORY = Path("checkpoints") / "automatic_preservation"
LIVE_CONCERN_CONFIG = ConcernConfig(
    gamma_score=0.625,
    gamma_morphology=0.375,
    gamma_uncertainty=0.0,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signal(values: ArrayLike, name: str) -> NDArray[np.float32]:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or result.size < 32 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least 32 finite samples.")
    return np.ascontiguousarray(result)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _artifact_path(directory: Path, record: Mapping[str, Any], name: str) -> Path:
    relative = Path(str(record.get("path", "")))
    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid packaged path for {name}.")
    path = (directory / relative).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError(f"Packaged path for {name} leaves the artifact directory.")
    if not path.is_file():
        raise FileNotFoundError(f"Missing automatic-preservation artifact: {path}")
    expected = str(record.get("sha256", "")).lower()
    actual = file_sha256(path)
    if not expected or expected != actual:
        raise ValueError(
            f"SHA-256 mismatch for {name}: expected {expected or '<missing>'}, got {actual}."
        )
    return path


def _load_generator(path: Path) -> TemporalUNetGenerator:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "denoiseapt-protocol-v1-generator":
        raise ValueError(f"Unsupported generator checkpoint format in {path}.")
    model = TemporalUNetGenerator(GeneratorConfig.from_dict(payload["generator_config"]))
    model.load_state_dict(payload["generator_state"])
    return model.cpu().eval()


@dataclass(frozen=True)
class CertificationEligibility:
    eligible: bool
    mode: str
    reason: str
    threshold_scope: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "mode": self.mode,
            "reason": self.reason,
            "threshold_scope": dict(self.threshold_scope),
        }


@dataclass(frozen=True)
class AutomaticRunResult:
    observation: NDArray[np.float32]
    soft_candidate: NDArray[np.float32]
    ordinary_candidate: NDArray[np.float32]
    automatic: NDArray[np.float32]
    normalization: WindowNormalization
    scores: Mapping[str, Mapping[str, NDArray[np.float32]]]
    concern: ConcernCues
    eligibility: CertificationEligibility
    projection: ProjectionResult | None
    soft_generator_latency_ms: float
    ordinary_generator_latency_ms: float
    runtime_provenance: Mapping[str, Any]
    controller: EvidencePreservationController | None

    def new_session(self) -> "AutomaticRuntimeSession":
        controlled = None
        if self.controller is not None and self.projection is not None:
            controlled = self.controller.new_session(
                self.observation, self.soft_candidate, self.projection
            )
        return AutomaticRuntimeSession(self, controlled)

    def automatic_payload(self) -> dict[str, Any]:
        if self.projection is not None:
            result = self.projection.to_dict(include_arrays=False)
            result.update(
                {
                    "mode": self.eligibility.mode,
                    "certification_eligible": True,
                    "eligibility_reason": self.eligibility.reason,
                    "beta": self.projection.beta.tolist(),
                    "repair_intervals": [
                        interval.to_dict() for interval in self.projection.intervals
                    ],
                    "runtime_provenance": dict(self.runtime_provenance),
                }
            )
            return result
        return {
            "mode": self.eligibility.mode,
            "certification_eligible": False,
            "eligibility_reason": self.eligibility.reason,
            "decision": "review_only",
            "auto_committed": False,
            "fallback_reason": "threshold_provenance_out_of_scope",
            "beta": np.zeros(self.observation.size, dtype=np.float32).tolist(),
            "intervals": [],
            "repair_intervals": [],
            "certificate": _review_only_certificate(self.eligibility.reason),
            "controller_latency_ms": 0.0,
            "generator_latency_ms": self.soft_generator_latency_ms,
            "total_latency_ms": self.soft_generator_latency_ms,
            "audit": {"events": [], "decision_hash": None},
            "provenance": {},
            "runtime_provenance": dict(self.runtime_provenance),
        }


@dataclass(frozen=True)
class HybridRunResult:
    """One experimental classical/DenoiseAPT hybrid result.

    ``auto_committed`` can be true only inside the existing frozen threshold
    scope and after a fresh whole-window A/B recheck.  Review-only requests
    still expose the routed candidate for comparison, but never call it
    certified or make it the editable approved baseline.
    """

    observation: NDArray[np.float32]
    classical_candidate: NDArray[np.float32]
    denoiseapt_repair_source: NDArray[np.float32]
    denoiseapt_repair_source_kind: str
    hybrid_candidate: NDArray[np.float32]
    output: NDArray[np.float32]
    routing: HybridCandidateResult
    scores: Mapping[str, NDArray[np.float32]]
    eligibility: CertificationEligibility
    projection: ProjectionResult | None
    independent_verification: VerificationResult
    independent_recheck_exact_match: bool
    auto_committed: bool
    fallback_reason: str | None
    routing_latency_ms: float
    hybrid_latency_ms: float
    runtime_provenance: Mapping[str, Any]

    def control_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": (
                "witness_certificate"
                if self.eligibility.eligible
                else "review_only"
            ),
            "certification_eligible": self.eligibility.eligible,
            "eligibility_reason": self.eligibility.reason,
            "decision": (
                self.projection.decision
                if self.projection is not None and self.auto_committed
                else ("observation_fallback" if self.eligibility.eligible else "review_only")
            ),
            "auto_committed": self.auto_committed,
            "fallback_reason": self.fallback_reason,
            "denoiseapt_repair_source_kind": self.denoiseapt_repair_source_kind,
            "independent_recheck_exact_match": self.independent_recheck_exact_match,
            "routing_latency_ms": self.routing_latency_ms,
            "hybrid_latency_ms": self.hybrid_latency_ms,
            "routing_weight": self.routing.routing_weight.tolist(),
            "routing": self.routing.routing_payload(),
            "runtime_provenance": dict(self.runtime_provenance),
        }
        if self.eligibility.eligible:
            payload["certificate"] = self.independent_verification.to_dict(
                include_scores=False
            )
            payload["projection"] = (
                None
                if self.projection is None
                else self.projection.to_dict(include_arrays=False)
            )
        else:
            payload["certificate"] = _review_only_certificate(
                self.eligibility.reason
            )
            payload["diagnostic_witness_check"] = (
                self.independent_verification.to_dict(include_scores=False)
            )
            payload["projection"] = None
        return payload


class _NormalizedTorchWitness:
    """Score raw signals with one immutable observation normalization."""

    def __init__(self, model: torch.nn.Module, normalization: WindowNormalization) -> None:
        self.model = model
        self.normalization = normalization
        self._cache: dict[str, NDArray[np.float32]] = {}

    def __call__(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        values = np.ascontiguousarray(signal, dtype=np.float32)
        key = array_sha256(values)
        if key not in self._cache:
            normalized = self.normalization.normalize(values)
            result = torch_detector_scores(self.model, normalized[None, :], device="cpu")[0]
            self._cache[key] = np.ascontiguousarray(result, dtype=np.float32)
        return self._cache[key].copy()


class AutomaticPreservationRuntime:
    """Frozen protocol-v1 models plus the development-frozen controller."""

    def __init__(self, root: Path, artifact_directory: Path | None = None) -> None:
        self.root = root.resolve()
        self.artifact_directory = (
            artifact_directory.resolve()
            if artifact_directory is not None
            else (self.root / DEFAULT_ARTIFACT_DIRECTORY).resolve()
        )
        manifest_path = self.artifact_directory / "runtime_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Automatic-preservation runtime bundle is missing. Run "
                "scripts/package_automatic_runtime.py before starting the demo."
            )
        self.manifest = _load_json(manifest_path)
        if int(self.manifest.get("schema_version", -1)) != RUNTIME_SCHEMA_VERSION:
            raise ValueError("Unsupported automatic-preservation runtime manifest schema.")
        files = self.manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Runtime manifest has no files map.")
        self.soft_path = _artifact_path(
            self.artifact_directory, files["soft_generator"], "soft_generator"
        )
        self.ordinary_path = _artifact_path(
            self.artifact_directory, files["ordinary_generator"], "ordinary_generator"
        )
        self.detector_path = _artifact_path(
            self.artifact_directory, files["detectors"], "detectors"
        )
        self.controller_path = _artifact_path(
            self.artifact_directory, files["controller"], "controller"
        )

        self.soft_generator = _load_generator(self.soft_path)
        self.ordinary_generator = _load_generator(self.ordinary_path)
        detector_payload = torch.load(
            self.detector_path, map_location="cpu", weights_only=True
        )
        if detector_payload.get("format") != "denoiseapt-protocol-v1-detectors":
            raise ValueError("Unsupported detector checkpoint format.")
        self.scorer_a = CausalForecasterScorer(
            ForecasterConfig.from_dict(detector_payload["scorer_a_config"])
        )
        self.scorer_a.load_state_dict(detector_payload["scorer_a_state"])
        self.scorer_a = self.scorer_a.cpu().freeze()
        self.scorer_b = CausalConvScorer(
            CausalConvConfig.from_dict(detector_payload["scorer_b_config"])
        )
        self.scorer_b.load_state_dict(detector_payload["scorer_b_state"])
        self.scorer_b = self.scorer_b.cpu().freeze()
        self.thresholds = {
            str(key): float(value) for key, value in detector_payload["thresholds"].items()
        }
        self._validate_detector_provenance()

        controller_payload = _load_json(self.controller_path)
        if controller_payload.get("confirmation_results") is not None:
            raise ValueError(
                "The live runtime accepts only the development-frozen controller artifact; "
                "confirmation results must remain outside the demo bundle."
            )
        self.controller_config = load_frozen_config(controller_payload)
        self.controller_artifact_sha256 = str(controller_payload["artifact_sha256"])
        self.threshold_scope = dict(self.manifest.get("threshold_scope") or {})
        # The guided benchmark case is intentionally not redistributed.  A
        # valid manifest record is part of the runtime contract, but the core
        # generators and witnesses must remain usable for review-only inputs
        # when that optional file has not yet been download-prepared.
        self._prepared_case_record()
        self._lock = threading.Lock()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    @property
    def ready(self) -> bool:
        return True

    @property
    def guided_certificate_ready(self) -> bool:
        """Whether the optional allowlisted case passes its exact hash check."""

        return bool(self.guided_certificate_status["ready"])

    @property
    def guided_certificate_status(self) -> dict[str, Any]:
        """Return fail-closed readiness for the narrow guided certificate path."""

        try:
            path = self._validate_prepared_case()
        except (OSError, ValueError) as exc:
            return {
                "ready": False,
                "reason": str(exc),
                "case_id": "tsb_ad_ucr_medical_guided",
            }
        return {
            "ready": True,
            "reason": "The optional guided case matches its frozen SHA-256 allowlist.",
            "case_id": "tsb_ad_ucr_medical_guided",
            "path": str(path.relative_to(self.root)).replace("\\", "/"),
        }

    @property
    def selected_model(self) -> str:
        return str(self.manifest.get("selected_model", ""))

    @property
    def provenance(self) -> dict[str, Any]:
        guided_status = self.guided_certificate_status
        return {
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
            "seed": int(self.manifest["seed"]),
            "selected_model": self.selected_model,
            "selection_source": self.manifest.get("selection_source"),
            "soft_generator_sha256": file_sha256(self.soft_path),
            "ordinary_generator_sha256": file_sha256(self.ordinary_path),
            "detector_checkpoint_sha256": file_sha256(self.detector_path),
            "controller_artifact_sha256": self.controller_artifact_sha256,
            "controller_configuration_scope": "development-frozen; confirmation data not accessed",
            "threshold_scope": dict(self.threshold_scope),
            "guided_certificate_ready": bool(guided_status["ready"]),
            "guided_certificate_reason": str(guided_status["reason"]),
        }

    def assess_eligibility(
        self,
        *,
        case_id: str,
        metadata: Mapping[str, Any],
        is_upload: bool,
        window_length: int,
    ) -> CertificationEligibility:
        scope = dict(self.threshold_scope)
        if is_upload:
            return CertificationEligibility(
                False,
                "review_only",
                "Uploaded data have no frozen calibration-domain provenance.",
                scope,
            )
        eligible_cases = {str(value) for value in scope.get("eligible_case_ids", [])}
        if case_id not in eligible_cases:
            return CertificationEligibility(
                False,
                "review_only",
                "This packaged domain is outside the frozen detector-threshold scope.",
                scope,
            )
        # Recheck the packaged case on every certificate-eligible request. This
        # keeps a long-running server fail-closed if the local fixture changes
        # after runtime initialization.
        try:
            self._validate_prepared_case()
        except (OSError, ValueError) as exc:
            return CertificationEligibility(
                False,
                "review_only",
                f"The allowlisted packaged case failed its frozen hash check: {exc}",
                scope,
            )
        expected_length = int(scope.get("window_length", 0))
        if expected_length <= 0 or window_length != expected_length:
            return CertificationEligibility(
                False,
                "review_only",
                f"Certification requires the frozen {expected_length}-sample window contract.",
                scope,
            )
        eligible_sources = {str(value) for value in scope.get("eligible_source_files", [])}
        source_file = str(metadata.get("source_file") or "")
        if not eligible_sources or source_file not in eligible_sources:
            return CertificationEligibility(
                False,
                "review_only",
                "The case source file is not recorded in the frozen threshold scope.",
                scope,
            )
        return CertificationEligibility(
            True,
            "witness_certificate",
            "A/B thresholds have frozen UCR-Medical validation provenance for this 512-sample contract.",
            scope,
        )

    def run(
        self,
        observation: ArrayLike,
        *,
        case_id: str,
        metadata: Mapping[str, Any],
        is_upload: bool,
    ) -> AutomaticRunResult:
        # The local server is threaded.  Serialize the complete model/controller
        # path so strict repeated-score checks cannot interleave with another
        # request using the same frozen PyTorch modules.
        with self._lock:
            return self._run_locked(
                observation,
                case_id=case_id,
                metadata=metadata,
                is_upload=is_upload,
            )

    def run_hybrid(self, inference: AutomaticRunResult) -> HybridRunResult:
        """Route MA9 through DenoiseAPT only where A/B evidence is at risk.

        The method reuses an already-computed ``AutomaticRunResult`` so neither
        generator is run a second time.  It is serialized with the rest of the
        live runtime because it uses the same frozen witness modules.
        """

        with self._lock:
            started = time.perf_counter()
            observed = _signal(inference.observation, "observation")
            classical = reflect_moving_average(
                observed, LIVE_HYBRID_CONFIG.classical_width
            )
            repair_source = _signal(inference.automatic, "denoiseapt_repair_source")
            if inference.projection is None:
                repair_source_kind = "denoiseapt_soft_review_source"
            elif inference.projection.auto_committed:
                repair_source_kind = "denoiseapt_automatic"
            else:
                repair_source_kind = "observation_fallback_from_denoiseapt_automatic"
            controller = EvidencePreservationController(
                self._witnesses(inference.normalization), self.controller_config
            )
            routing = build_evidence_gated_candidate(
                observed,
                classical,
                repair_source,
                controller,
                config=LIVE_HYBRID_CONFIG,
            )

            projection: ProjectionResult | None = None
            proposed = routing.candidate
            if inference.eligibility.eligible:
                projection = controller.project(
                    observed,
                    routing.candidate,
                    base_provenance={
                        "hybrid_algorithm_version": LIVE_HYBRID_CONFIG.algorithm_version,
                        "hybrid_config_sha256": LIVE_HYBRID_CONFIG.sha256,
                        "classical_filter": LIVE_HYBRID_CONFIG.classical_filter,
                        "denoiseapt_soft_generator_sha256": file_sha256(self.soft_path),
                        "routing_repair_source_kind": routing.repair_source_kind,
                        "upstream_denoiseapt_source_kind": repair_source_kind,
                    },
                    generator_latency_ms=inference.soft_generator_latency_ms,
                )
                proposed = projection.automatic_signal

            # Fresh wrappers force a new scorer inference for the final signal;
            # this check does not reuse the routing controller's signal cache.
            independent_controller = EvidencePreservationController(
                self._witnesses(inference.normalization), self.controller_config
            )
            independent = independent_controller.verify(observed, proposed)
            exact_match = (
                False
                if projection is None
                else _verification_scores_equal(projection.certificate, independent)
            )
            auto_committed = bool(
                inference.eligibility.eligible
                and projection is not None
                and projection.auto_committed
                and projection.certificate.passed
                and independent.passed
                and exact_match
            )
            fallback_reason: str | None = None
            output = np.ascontiguousarray(proposed, dtype=np.float32)
            if inference.eligibility.eligible and not auto_committed:
                output = observed.copy()
                if projection is None:
                    fallback_reason = "projection_missing"
                elif not projection.auto_committed:
                    fallback_reason = projection.fallback_reason or "projection_abstained"
                elif not independent.passed:
                    fallback_reason = "independent_postcheck_failed"
                elif not exact_match:
                    fallback_reason = "independent_postcheck_score_mismatch"
                else:
                    fallback_reason = "hybrid_postcondition_failed"
                independent = EvidencePreservationController(
                    self._witnesses(inference.normalization), self.controller_config
                ).verify(observed, output)
            elif not inference.eligibility.eligible:
                fallback_reason = "threshold_provenance_out_of_scope"

            by_id = {item.witness_id: item.output_scores for item in independent.witnesses}
            scores = {
                witness_id: np.ascontiguousarray(by_id[witness_id], dtype=np.float32)
                for witness_id in LIVE_HYBRID_CONFIG.required_witness_ids
                if witness_id in by_id
            }
            if set(scores) != set(LIVE_HYBRID_CONFIG.required_witness_ids):
                # A scoring failure cannot produce an apparently usable hybrid.
                # Fall back to the exact observation and its scores that were
                # already computed successfully by the primary runtime path.
                output = observed.copy()
                auto_committed = False
                fallback_reason = "hybrid_scores_unavailable"
                scores = {
                    witness_id: np.ascontiguousarray(
                        inference.scores[witness_id]["observation"], dtype=np.float32
                    )
                    for witness_id in LIVE_HYBRID_CONFIG.required_witness_ids
                }

            routing_latency = (time.perf_counter() - started) * 1000.0
            base_latency = (
                float(inference.projection.total_latency_ms or 0.0)
                if inference.projection is not None
                else float(inference.soft_generator_latency_ms)
            )
            total_latency = base_latency + routing_latency
            provenance = dict(self.provenance)
            provenance.update(
                {
                    "hybrid_algorithm_version": LIVE_HYBRID_CONFIG.algorithm_version,
                    "hybrid_config_sha256": LIVE_HYBRID_CONFIG.sha256,
                    "classical_filter": LIVE_HYBRID_CONFIG.classical_filter,
                    "normalization": {
                        "method": "observation median and max(IQR/1.349, standard deviation, 1e-4)",
                        "center": inference.normalization.center,
                        "scale": inference.normalization.scale,
                    },
                    "witnesses": controller.provenance["witnesses"],
                    "controller_config_sha256": self.controller_config.sha256,
                    "observation_sha256": array_sha256(observed),
                    "classical_sha256": array_sha256(classical),
                    "denoiseapt_repair_source_sha256": array_sha256(repair_source),
                    "denoiseapt_repair_source_kind": repair_source_kind,
                    "routing_weight_sha256": array_sha256(routing.routing_weight),
                    "hybrid_candidate_sha256": array_sha256(routing.candidate),
                    "hybrid_output_sha256": array_sha256(output),
                    "historical_automatic_unchanged": True,
                    "review_only_outside_threshold_scope": True,
                    "development_only": True,
                    "confirmation_accessed": False,
                }
            )
            return HybridRunResult(
                observation=observed,
                classical_candidate=classical,
                denoiseapt_repair_source=repair_source,
                denoiseapt_repair_source_kind=repair_source_kind,
                hybrid_candidate=routing.candidate,
                output=np.ascontiguousarray(output, dtype=np.float32),
                routing=routing,
                scores=scores,
                eligibility=inference.eligibility,
                projection=projection,
                independent_verification=independent,
                independent_recheck_exact_match=exact_match,
                auto_committed=auto_committed,
                fallback_reason=fallback_reason,
                routing_latency_ms=routing_latency,
                hybrid_latency_ms=total_latency,
                runtime_provenance=provenance,
            )

    def _run_locked(
        self,
        observation: ArrayLike,
        *,
        case_id: str,
        metadata: Mapping[str, Any],
        is_upload: bool,
    ) -> AutomaticRunResult:
        observed = _signal(observation, "observation")
        normalization = WindowNormalization.fit(observed)
        normalized = normalization.normalize(observed)
        tensor = torch.from_numpy(normalized)[None, None, :]
        soft_started = time.perf_counter()
        soft_normalized = self._forward(self.soft_generator, tensor)
        soft_latency = (time.perf_counter() - soft_started) * 1000.0
        ordinary_started = time.perf_counter()
        ordinary_normalized = self._forward(self.ordinary_generator, tensor)
        ordinary_latency = (time.perf_counter() - ordinary_started) * 1000.0

        soft = normalization.restore(soft_normalized)
        ordinary = normalization.restore(ordinary_normalized)
        witnesses = self._witnesses(normalization)
        scores = {
            "A_causal_mlp": {
                "observation": witnesses[0].scorer(observed),
                "soft_candidate": witnesses[0].scorer(soft),
                "ordinary_cgan": witnesses[0].scorer(ordinary),
            },
            "B_causal_conv": {
                "observation": witnesses[1].scorer(observed),
                "soft_candidate": witnesses[1].scorer(soft),
                "ordinary_cgan": witnesses[1].scorer(ordinary),
            },
        }
        concern = compute_concern_cues(
            normalized,
            soft_normalized,
            soft_normalized[None, :],
            scores["A_causal_mlp"]["observation"],
            scores["A_causal_mlp"]["soft_candidate"],
            LIVE_CONCERN_CONFIG,
        )
        eligibility = self.assess_eligibility(
            case_id=case_id,
            metadata=metadata,
            is_upload=is_upload,
            window_length=observed.size,
        )
        projection: ProjectionResult | None = None
        controller: EvidencePreservationController | None = None
        automatic = soft.copy()
        if eligibility.eligible:
            controller = EvidencePreservationController(witnesses, self.controller_config)
            projection = controller.project(
                observed,
                soft,
                base_provenance={
                    "selected_model": self.selected_model,
                    "soft_generator_sha256": file_sha256(self.soft_path),
                    "normalization": {
                        "method": "observation median and max(IQR/1.349, standard deviation, 1e-4)",
                        "center": normalization.center,
                        "scale": normalization.scale,
                    },
                },
                generator_latency_ms=soft_latency,
            )
            automatic = projection.automatic_signal.copy()
        for witness_id, witness in zip(("A_causal_mlp", "B_causal_conv"), witnesses):
            scores[witness_id]["automatic"] = witness.scorer(automatic)
        return AutomaticRunResult(
            observation=observed,
            soft_candidate=soft,
            ordinary_candidate=ordinary,
            automatic=np.ascontiguousarray(automatic, dtype=np.float32),
            normalization=normalization,
            scores=scores,
            concern=concern,
            eligibility=eligibility,
            projection=projection,
            soft_generator_latency_ms=soft_latency,
            ordinary_generator_latency_ms=ordinary_latency,
            runtime_provenance=self.provenance,
            controller=controller,
        )

    def score_signal(
        self,
        signal: ArrayLike,
        normalization: WindowNormalization,
        *,
        witness_id: str = "A_causal_mlp",
    ) -> NDArray[np.float32]:
        if witness_id not in {"A_causal_mlp", "B_causal_conv"}:
            raise ValueError(f"Unknown witness: {witness_id}")
        with self._lock:
            witness = self._witnesses(normalization)[
                0 if witness_id == "A_causal_mlp" else 1
            ]
            return witness.scorer(_signal(signal, "signal"))

    @staticmethod
    def _forward(
        model: TemporalUNetGenerator, tensor: torch.Tensor
    ) -> NDArray[np.float32]:
        model.eval()
        with torch.inference_mode():
            output = model(tensor)
        return np.ascontiguousarray(output[0, 0].cpu().numpy(), dtype=np.float32)

    def _witnesses(self, normalization: WindowNormalization) -> tuple[WitnessSpec, ...]:
        checkpoint_hash = file_sha256(self.detector_path)
        threshold_source = str(self.threshold_scope.get("threshold_source", ""))
        state_hashes = dict(self.manifest.get("detector_state_sha256") or {})
        return (
            WitnessSpec(
                witness_id="A_causal_mlp",
                scorer=_NormalizedTorchWitness(self.scorer_a, normalization),
                threshold=self.thresholds["A_causal_mlp"],
                model_sha256=str(state_hashes["A_causal_mlp"]),
                threshold_source=threshold_source,
                threshold_source_sha256=checkpoint_hash,
                context_left=int(self.scorer_a.config.context_length),
            ),
            WitnessSpec(
                witness_id="B_causal_conv",
                scorer=_NormalizedTorchWitness(self.scorer_b, normalization),
                threshold=self.thresholds["B_causal_conv"],
                model_sha256=str(state_hashes["B_causal_conv"]),
                threshold_source=threshold_source,
                threshold_source_sha256=checkpoint_hash,
                context_left=int(self.scorer_b.config.warmup),
            ),
        )

    def _validate_detector_provenance(self) -> None:
        expected_thresholds = dict(self.manifest.get("thresholds") or {})
        for witness_id in ("A_causal_mlp", "B_causal_conv"):
            if witness_id not in expected_thresholds:
                raise ValueError(f"Runtime manifest lacks {witness_id} threshold provenance.")
            if float(expected_thresholds[witness_id]) != self.thresholds[witness_id]:
                raise ValueError(f"Frozen threshold mismatch for {witness_id}.")
        expected_states = dict(self.manifest.get("detector_state_sha256") or {})
        actual_states = {
            "A_causal_mlp": state_dict_sha256(self.scorer_a),
            "B_causal_conv": state_dict_sha256(self.scorer_b),
        }
        if actual_states != expected_states:
            raise ValueError("Detector state hashes do not match the runtime manifest.")

    def _prepared_case_record(self) -> tuple[dict[str, Any], Path]:
        record = dict(self.manifest.get("prepared_case") or {})
        relative = Path(str(record.get("path", "")))
        expected_hash = str(record.get("sha256", "")).lower()
        if (
            record.get("case_id") != "tsb_ad_ucr_medical_guided"
            or not relative.name
            or relative.is_absolute()
            or ".." in relative.parts
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("Runtime manifest has no valid allowlisted prepared-case record.")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ValueError("The allowlisted prepared-case path leaves the package root.")
        return record, path

    def _validate_prepared_case(self) -> Path:
        record, path = self._prepared_case_record()
        if not path.is_file():
            raise FileNotFoundError(
                "The optional guided case is not installed; run the documented "
                "download-and-prepare step to enable certificate mode."
            )
        if file_sha256(path) != str(record["sha256"]).lower():
            raise ValueError("The allowlisted prepared case does not match its frozen hash.")
        return path


class AutomaticRuntimeSession:
    """Reversible human edits around an immutable automatic baseline."""

    def __init__(
        self,
        result: AutomaticRunResult,
        controlled: ControlledSignalSession | None,
    ) -> None:
        self.result = result
        self.controlled = controlled
        self.current = result.automatic.copy()
        self._states: list[NDArray[np.float32]] = []
        self._revision = 0
        self._records: list[dict[str, Any]] = []
        self._verification: VerificationResult | None = (
            result.projection.certificate if result.projection is not None else None
        )

    @property
    def revision(self) -> int:
        return self._revision

    def apply(
        self,
        action: str,
        start: int | None = None,
        end: int | None = None,
        *,
        beta: float = 0.5,
        expected_revision: int | None = None,
    ) -> None:
        if action == "revert":
            self.revert(expected_revision=expected_revision)
            return
        if expected_revision is not None and int(expected_revision) != self._revision:
            raise RuntimeError(
                f"Stale intervention revision: expected {expected_revision}, current {self._revision}."
            )
        length = self.current.size
        start_index = 0 if start is None else int(start)
        end_index = length if end is None else int(end)
        if action == "restore_automatic":
            start_index, end_index = 0, length
        if not 0 <= start_index < end_index <= length:
            raise ValueError("Intervention interval must satisfy 0 <= start < end <= length.")
        if action == "blend" and (not np.isfinite(beta) or not 0.0 <= beta <= 1.0):
            raise ValueError("Blend beta must lie in [0, 1].")
        previous = self.current.copy()
        self._states.append(previous)
        try:
            if self.controlled is not None:
                mapped = "accept_candidate" if action == "accept" else action
                self.current, self._verification = self.controlled.apply(
                    mapped,
                    start_index,
                    end_index,
                    beta=beta,
                    expected_revision=self.controlled.revision,
                )
            else:
                self.current = self._review_only_edit(
                    action, start_index, end_index, beta=beta
                )
        except Exception:
            self._states.pop()
            raise
        self._revision += 1
        self._records.append(
            {
                "revision": self._revision,
                "action": action,
                "start": start_index,
                "end": end_index,
                "beta": float(beta) if action == "blend" else None,
                "signal_sha256": array_sha256(self.current),
            }
        )

    def revert(self, *, expected_revision: int | None = None) -> None:
        if expected_revision is not None and int(expected_revision) != self._revision:
            raise RuntimeError(
                f"Stale intervention revision: expected {expected_revision}, current {self._revision}."
            )
        if not self._states:
            raise RuntimeError("No human intervention is available to revert.")
        if self.controlled is not None:
            self.current, self._verification = self.controlled.revert(
                expected_revision=self.controlled.revision
            )
        else:
            self.current = self._states[-1].copy()
        self._states.pop()
        self._revision += 1
        self._records.append(
            {
                "revision": self._revision,
                "action": "revert",
                "start": 0,
                "end": int(self.current.size),
                "beta": None,
                "signal_sha256": array_sha256(self.current),
            }
        )

    def _review_only_edit(
        self, action: str, start: int, end: int, *, beta: float
    ) -> NDArray[np.float32]:
        output = self.current.copy()
        if action == "restore_automatic":
            return self.result.automatic.copy()
        if action == "accept":
            output[start:end] = self.result.soft_candidate[start:end]
        elif action == "protect":
            output[start:end] = self.result.observation[start:end]
        elif action == "blend":
            if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
                raise ValueError("Blend beta must lie in [0, 1].")
            output[start:end] = (
                beta * self.result.observation[start:end]
                + (1.0 - beta) * self.result.soft_candidate[start:end]
            )
        else:
            raise ValueError(f"Unsupported intervention action: {action!r}")
        return np.ascontiguousarray(output, dtype=np.float32)

    def certification_payload(self) -> dict[str, Any]:
        if self.controlled is None:
            return _review_only_certificate(self.result.eligibility.reason)
        assert self._verification is not None
        return self._verification.to_dict(include_scores=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.tolist(),
            "history_depth": len(self._states),
            "revision": self._revision,
            "actions": list(self._records),
            "certificate": self.certification_payload(),
            "mode": self.result.eligibility.mode,
        }


def _review_only_certificate(reason: str) -> dict[str, Any]:
    return {
        "status": "unverified",
        "passed": False,
        "witnesses": [],
        "limitations": [
            "No frozen calibrated-threshold provenance applies to this request.",
            "Model scores and concern cues are available only for human review.",
            "The output is not an anomaly-preservation certificate.",
        ],
        "error": reason,
    }


def _verification_scores_equal(
    left: VerificationResult, right: VerificationResult
) -> bool:
    """Require byte-equal full-window witness scores for an independent recheck."""

    if left.status != right.status or left.passed != right.passed:
        return False
    left_by_id = {item.witness_id: item.output_scores for item in left.witnesses}
    right_by_id = {item.witness_id: item.output_scores for item in right.witnesses}
    return set(left_by_id) == set(right_by_id) and all(
        np.array_equal(left_by_id[witness_id], right_by_id[witness_id])
        for witness_id in left_by_id
    )
