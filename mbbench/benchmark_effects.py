from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .generation_core import CompositionStage


def _batch_weights(batch_indices: np.ndarray, n_batches: int) -> np.ndarray:
    counts = np.bincount(batch_indices, minlength=n_batches).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("Every batch must contain at least one sample.")
    return counts / counts.sum()


def batch_coordinates(
    batch_indices: np.ndarray,
    n_batches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_batches < 2:
        raise ValueError("Batch effects require at least two batches.")
    weights = _batch_weights(batch_indices, n_batches)
    coordinates = rng.normal(size=(n_batches, n_batches - 1))
    coordinates -= weights @ coordinates
    coordinates *= rng.lognormal(mean=0.0, sigma=0.45, size=n_batches - 1)
    if np.linalg.matrix_rank(coordinates, tol=1e-10) != n_batches - 1:
        raise ValueError("Sampled batch coordinates are rank deficient.")
    return coordinates


def _scale_offsets(
    offsets: np.ndarray,
    batch_indices: np.ndarray,
    strength: float,
) -> np.ndarray:
    weights = _batch_weights(batch_indices, len(offsets))
    centered = offsets - weights @ offsets
    rms = float(np.sqrt(np.sum(weights * np.mean(np.square(centered), axis=1))))
    if rms <= np.finfo(float).eps:
        raise ValueError("Batch offsets have zero weighted RMS.")
    return centered * (float(strength) / rms)


def global_offsets(
    reference_logits: np.ndarray,
    batch_indices: np.ndarray,
    n_batches: int,
    strength: float,
    rng: np.random.Generator,
    *,
    n_components: int = 3,
) -> np.ndarray:
    reference = np.asarray(reference_logits, dtype=np.float64)
    clr_like = reference - reference.mean(axis=1, keepdims=True)
    clr_like -= clr_like.mean(axis=0, keepdims=True)
    requested = min(n_components, n_batches - 1)
    maximum_rank = min(clr_like.shape)
    if maximum_rank > 64 and requested < maximum_rank:
        _, singular_values, right = randomized_svd(
            clr_like,
            n_components=requested,
            n_iter=7,
            random_state=0,
        )
    else:
        _, singular_values, right = np.linalg.svd(clr_like, full_matrices=False)
    available = int(np.count_nonzero(singular_values > singular_values[0] * 1e-10))
    components = requested
    if available < components:
        raise ValueError(
            f"Reference logits provide {available} usable global directions; {components} are required."
        )
    loadings = right[:components]
    loadings -= loadings.mean(axis=1, keepdims=True)
    loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)
    coordinates = batch_coordinates(batch_indices, n_batches, rng)[:, :components]
    return _scale_offsets(coordinates @ loadings, batch_indices, strength)


def sparse_offsets(
    eligible: np.ndarray,
    batch_indices: np.ndarray,
    n_batches: int,
    fraction: float,
    strength: float,
    rng: np.random.Generator,
    *,
    selected: np.ndarray | None = None,
    center_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    eligible_indices = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if selected is None:
        count = max(
            2 if center_features else 1,
            int(round(float(fraction) * len(eligible_indices))),
        )
        count = min(count, len(eligible_indices))
        selected = np.sort(rng.choice(eligible_indices, size=count, replace=False))
    else:
        selected = np.sort(np.asarray(selected, dtype=np.int64))
        if len(selected) == 0 or not np.isin(selected, eligible_indices).all():
            raise ValueError("Specified sparse batch targets are not eligible.")
    coordinates = batch_coordinates(batch_indices, n_batches, rng)
    loadings = rng.normal(size=(coordinates.shape[1], len(selected)))
    if center_features:
        loadings -= loadings.mean(axis=1, keepdims=True)
    offsets = np.zeros((n_batches, len(eligible)), dtype=np.float64)
    offsets[:, selected] = coordinates @ loadings
    return _scale_offsets(offsets, batch_indices, strength), selected


def scalar_offsets(
    batch_indices: np.ndarray,
    n_batches: int,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    coordinates = batch_coordinates(batch_indices, n_batches, rng)
    values = coordinates[:, 0]
    weights = _batch_weights(batch_indices, n_batches)
    values -= weights @ values
    rms = float(np.sqrt(np.sum(weights * np.square(values))))
    if rms <= np.finfo(float).eps:
        raise ValueError("Scalar batch offsets have zero weighted RMS.")
    return values * (float(strength) / rms)


def redistribute_target_mass(
    stage: CompositionStage,
    sample_scores: np.ndarray,
    coefficients: np.ndarray,
) -> CompositionStage:
    before = np.asarray(stage.composition, dtype=np.float64)
    scores = np.asarray(sample_scores, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    target = coefficients != 0
    if target.sum() < 2 or len(scores) != len(before):
        raise ValueError(
            "Strict redistribution requires matching scores and at least two targets."
        )
    after = before.copy()
    target_mass = before[:, target].sum(axis=1)
    weighted = before[:, target] * np.exp(np.outer(scores, coefficients[target]))
    active = target_mass > 0
    after[np.ix_(active, target)] = (
        weighted[active]
        / weighted[active].sum(axis=1, keepdims=True)
        * target_mass[active, None]
    )
    if not np.allclose(after[:, ~target], before[:, ~target], atol=1e-14, rtol=0):
        raise AssertionError("Strict redistribution changed a non-target feature.")
    if not np.allclose(after[:, target].sum(axis=1), target_mass, atol=1e-12, rtol=0):
        raise AssertionError("Strict redistribution changed target-set mass.")
    if not np.array_equal(after > 0, stage.support_mask):
        raise AssertionError("Strict redistribution changed the support mask.")
    logits = stage.abundance_logits.copy()
    logits[stage.support_mask] = np.log(after[stage.support_mask])
    return CompositionStage(
        abundance_logits=logits,
        support_logits=stage.support_logits.copy(),
        support_probability=stage.support_probability.copy(),
        support_mask=stage.support_mask.copy(),
        composition=after,
    )
