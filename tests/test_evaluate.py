from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from scipy.spatial.distance import cdist

from mbbench.evaluate import _mean_distance, evaluate_generation


def test_identical_tables_have_zero_distribution_distance(tmp_path):
    frame = pd.DataFrame({"sample_id": ["a", "b"], "g1": [0.8, 0.2], "g2": [0.2, 0.8]})
    path = tmp_path / "abundance.csv"
    frame.to_csv(path, index=False)
    metrics = evaluate_generation(
        real_abundance=path,
        synthetic_abundance=path,
        output_dir=tmp_path / "evaluation",
    )
    assert abs(metrics["bray_curtis_energy"]) < 1e-12
    assert abs(metrics["aitchison_energy"]) < 1e-12
    assert metrics["aitchison_pseudocount"] > 0

    in_memory = evaluate_generation(
        real_abundance=path,
        synthetic_abundance=path,
    )
    assert in_memory == metrics


def test_blockwise_distance_matches_dense_distance():
    rng = np.random.default_rng(42)
    left = rng.normal(size=(13, 5))
    right = rng.normal(size=(17, 5))
    expected = float(cdist(left, right).mean())
    assert _mean_distance(left, right, "euclidean", block_size=4) == pytest.approx(
        expected
    )
