from __future__ import annotations

import numpy as np

from mbbench.benchmark_effects import (
    global_offsets,
    redistribute_target_mass,
    scalar_offsets,
)
from mbbench.generation_core import decode_composition, sample_counts


def test_shared_support_and_count_observation_are_reproducible():
    abundance = np.array([[1.0, 0.0, -1.0], [0.2, -0.1, 0.4]])
    support = np.array([[2.0, -2.0, 0.0], [1.0, 0.5, -1.0]])
    uniforms = np.random.default_rng(7).random(support.shape)
    first = decode_composition(abundance, support, uniforms=uniforms)
    second = decode_composition(abundance, support, uniforms=uniforms)
    np.testing.assert_array_equal(first.support_mask, second.support_mask)
    np.testing.assert_allclose(first.composition, second.composition)
    counts = sample_counts(
        first.composition,
        np.array([100, 120]),
        model="dirichlet-multinomial",
        concentrations=np.array([500.0, 800.0]),
        seed=11,
    )
    np.testing.assert_array_equal(counts.sum(axis=1), np.array([100, 120]))


def test_strict_target_redistribution_preserves_declared_invariants():
    abundance = np.log(np.array([[0.4, 0.3, 0.2, 0.1], [0.2, 0.2, 0.3, 0.3]]))
    support = np.full_like(abundance, 10.0)
    stage = decode_composition(abundance, support, mode="deterministic")
    coefficients = np.array([np.log(2), -np.log(2), 0.0, 0.0])
    changed = redistribute_target_mass(stage, np.array([0.0, 1.0]), coefficients)
    np.testing.assert_allclose(
        changed.composition[:, 2:], stage.composition[:, 2:], atol=1e-14
    )
    np.testing.assert_allclose(
        changed.composition[:, :2].sum(axis=1),
        stage.composition[:, :2].sum(axis=1),
        atol=1e-12,
    )
    np.testing.assert_array_equal(changed.support_mask, stage.support_mask)


def test_batch_offsets_are_sample_weight_centered():
    batch = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    reference = np.random.default_rng(3).normal(size=(40, 8))
    rng = np.random.default_rng(9)
    offsets = global_offsets(reference, batch, 3, 0.5, rng)
    weights = np.bincount(batch, minlength=3) / len(batch)
    np.testing.assert_allclose(weights @ offsets, 0.0, atol=1e-12)
    scalar = scalar_offsets(batch, 3, 0.2, rng)
    assert abs(float(weights @ scalar)) < 1e-12
