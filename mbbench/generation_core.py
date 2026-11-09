from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompositionStage:
    abundance_logits: np.ndarray
    support_logits: np.ndarray
    support_probability: np.ndarray
    support_mask: np.ndarray
    composition: np.ndarray


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def support_probability(
    logits: np.ndarray,
    *,
    mode: str,
    threshold: float,
    temperature: float,
) -> np.ndarray:
    if mode not in {"deterministic", "bernoulli", "temperature-bernoulli"}:
        raise ValueError(f"Unknown support mode {mode!r}.")
    if not 0 < threshold < 1 or temperature <= 0:
        raise ValueError(
            "Support threshold must lie in (0, 1) and temperature must be positive."
        )
    values = np.asarray(logits, dtype=np.float64)
    if mode != "temperature-bernoulli":
        return sigmoid(values)
    center = np.log(threshold / (1.0 - threshold))
    return sigmoid((values - center) / temperature)


def decode_composition(
    abundance_logits: np.ndarray,
    support_logits: np.ndarray,
    *,
    mode: str = "temperature-bernoulli",
    threshold: float = 0.5,
    temperature: float = 0.6,
    uniforms: np.ndarray | None = None,
) -> CompositionStage:
    abundance = np.asarray(abundance_logits, dtype=np.float64)
    support = np.asarray(support_logits, dtype=np.float64)
    if abundance.ndim != 2 or abundance.shape != support.shape:
        raise ValueError("Abundance and support logits must be matching matrices.")
    if not np.isfinite(abundance).all() or not np.isfinite(support).all():
        raise ValueError("Abundance and support logits must be finite.")
    probability = support_probability(
        support,
        mode=mode,
        threshold=threshold,
        temperature=temperature,
    )
    if mode == "deterministic":
        mask = probability >= threshold
    else:
        if uniforms is None or np.shape(uniforms) != probability.shape:
            raise ValueError(
                "Stochastic support decoding requires matching shared uniforms."
            )
        mask = np.asarray(uniforms, dtype=np.float64) < probability
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty] = False
        mask[np.flatnonzero(empty), np.argmax(probability[empty], axis=1)] = True
    shifted = np.where(mask, abundance, -np.inf)
    shifted -= np.max(shifted, axis=1, keepdims=True)
    values = np.where(mask, np.exp(shifted), 0.0)
    composition = values / values.sum(axis=1, keepdims=True)
    if not np.isfinite(composition).all() or np.any(composition < 0):
        raise ValueError("Decoded composition is invalid.")
    return CompositionStage(
        abundance_logits=abundance.copy(),
        support_logits=support.copy(),
        support_probability=probability,
        support_mask=mask,
        composition=composition,
    )


def sample_counts(
    composition: np.ndarray,
    depths: np.ndarray,
    *,
    model: str,
    concentrations: np.ndarray | None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    values = np.asarray(composition, dtype=np.float64)
    depth_values = np.asarray(depths)
    if values.ndim != 2 or len(values) != len(depth_values):
        raise ValueError(
            "Composition and depths must contain the same number of samples."
        )
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Composition rows must sum to one before count observation.")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("Composition must be finite and nonnegative.")
    numeric_depths = depth_values.astype(np.float64)
    if not np.array_equal(numeric_depths, np.rint(numeric_depths)) or np.any(
        numeric_depths <= 0
    ):
        raise ValueError("Observation depths must be positive integers.")
    if model not in {"multinomial", "dirichlet-multinomial"}:
        raise ValueError(
            "Observation model must be multinomial or dirichlet-multinomial."
        )
    if model == "dirichlet-multinomial":
        if concentrations is None:
            raise ValueError(
                "Dirichlet-multinomial observation requires concentrations."
            )
        concentration_values = np.asarray(concentrations, dtype=np.float64)
        if concentration_values.shape != (len(values),) or np.any(
            concentration_values <= 0
        ):
            raise ValueError(
                "Observation concentrations must be positive and match samples."
            )
    else:
        concentration_values = np.ones(len(values), dtype=np.float64)
    if (seed is None) == (rng is None):
        raise ValueError("Provide exactly one of seed or rng for count observation.")
    rng = rng if rng is not None else np.random.default_rng(int(seed))
    counts = np.zeros(values.shape, dtype=np.int64)
    for index, probability in enumerate(values):
        draw_probability = probability
        if model == "dirichlet-multinomial":
            active = probability > 0
            draw_probability = np.zeros_like(probability)
            draw_probability[active] = rng.dirichlet(
                np.maximum(probability[active] * concentration_values[index], 1e-9)
            )
        counts[index] = rng.multinomial(int(numeric_depths[index]), draw_probability)
    return counts
