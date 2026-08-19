from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_external_denoiser_benchmark_v1 as benchmark


def test_config_is_explicitly_retrospective() -> None:
    config = benchmark._load_json(benchmark.CONFIG_PATH)
    benchmark._validate_config(config)
    assert config["scientific_scope"]["confirmation_claim_eligible"] is False
    assert config["scientific_scope"]["state_of_the_art_claim_eligible"] is False
    assert tuple(config["methods"]) == benchmark.METHODS


def test_median_filter_uses_edge_padding() -> None:
    values = np.zeros(512, dtype=np.float32)
    values[:4] = [9.0, 1.0, 8.0, 2.0]
    output = benchmark.median_filter_w3(values)
    assert output.shape == values.shape
    assert np.array_equal(output[:5], np.asarray([9.0, 8.0, 2.0, 2.0, 0.0], dtype=np.float32))


def test_wavelet_and_noisereduce_outputs_are_finite() -> None:
    x = np.linspace(0.0, 4.0 * np.pi, 512, dtype=np.float32)
    values = np.asarray(np.sin(x) + 0.05 * np.cos(13.0 * x), dtype=np.float32)
    wavelet = benchmark.wavelet_shrinkage(
        values, wavelet="db4", threshold_mode="soft", threshold_multiplier=1.0
    )
    spectral = benchmark.noisereduce_filter(
        values, stationary=True, prop_decrease=0.5, n_fft=32
    )
    assert wavelet.shape == spectral.shape == values.shape
    assert np.all(np.isfinite(wavelet))
    assert np.all(np.isfinite(spectral))


def test_development_selection_is_one_window_per_group() -> None:
    reference, observation, groups = benchmark._development_conditions()
    assert reference.shape == observation.shape == (732, 512)
    assert len(set(groups)) == 61
    assert all(int(np.sum(groups == group)) == 12 for group in set(groups))


def test_rins_t_single_window_is_deterministic() -> None:
    x = np.linspace(0.0, 8.0 * np.pi, 512, dtype=np.float32)
    values = np.asarray(np.sin(x) + 0.08 * np.cos(17.0 * x), dtype=np.float32)
    first = benchmark.run_rins_t(values[None, :], workers=1)[0]
    second = benchmark.run_rins_t(values[None, :], workers=1)[0]
    assert first.shape == values.shape
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_group_balanced_endpoint_uses_group_means() -> None:
    reference = np.zeros((3, 512), dtype=np.float32)
    observation = np.ones_like(reference)
    output = np.stack(
        [
            np.full(512, 1.0, dtype=np.float32),
            np.full(512, 3.0, dtype=np.float32),
            np.full(512, 2.0, dtype=np.float32),
        ]
    )
    groups = np.asarray(["g1", "g1", "g2"])
    # Observation scale is floored to one for these constant windows.
    assert benchmark._group_balanced_os(reference, observation, output, groups) == 20000.0
