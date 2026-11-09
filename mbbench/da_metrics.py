from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def boolean_series(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if values.notna().all() and numeric.notna().all() and numeric.isin((0, 1)).all():
        return numeric.astype(bool)
    converted = values.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if converted.isna().any():
        raise ValueError(f"{name} must contain only Boolean values.")
    return converted.astype(bool)


def align_da_input(
    frame: pd.DataFrame, taxa: list[str], *, method: str
) -> pd.DataFrame:
    if "taxon" not in frame or "significant" not in frame:
        raise ValueError("DA input requires taxon and significant columns.")
    aligned = frame.copy()
    aligned["taxon"] = aligned["taxon"].astype(str)
    if aligned["taxon"].duplicated().any():
        duplicate = aligned.loc[aligned["taxon"].duplicated(), "taxon"].iloc[0]
        raise ValueError(f"DA input contains duplicate taxon {duplicate!r}.")
    unknown = sorted(set(aligned["taxon"]) - set(taxa))
    missing = sorted(set(taxa) - set(aligned["taxon"]))
    if unknown:
        raise ValueError(f"DA input contains unknown taxa: {', '.join(unknown[:5])}.")
    if missing:
        raise ValueError(
            "DA input must include every benchmark taxon; missing: "
            + ", ".join(missing[:5])
            + "."
        )
    aligned["significant"] = boolean_series(aligned["significant"], "significant")
    if "tested" in aligned:
        aligned["tested"] = boolean_series(aligned["tested"], "tested")
    else:
        aligned["tested"] = True
    if (aligned["significant"] & ~aligned["tested"]).any():
        raise ValueError("A taxon cannot be significant when tested is false.")
    for column in ("p_value", "q_value"):
        if column in aligned:
            original = aligned[column]
            aligned[column] = pd.to_numeric(original, errors="coerce")
            malformed = original.notna() & aligned[column].isna()
            if malformed.any():
                raise ValueError(f"{column} must contain numeric values or missing values.")
            invalid = aligned[column].notna() & ~aligned[column].between(0, 1)
            if invalid.any():
                raise ValueError(f"{column} must be missing or lie in [0, 1].")
    if "effect" in aligned:
        original = aligned["effect"]
        aligned["effect"] = pd.to_numeric(original, errors="coerce")
        if (original.notna() & aligned["effect"].isna()).any():
            raise ValueError("effect must contain numeric values or missing values.")
        if np.isinf(aligned["effect"].dropna()).any():
            raise ValueError("effect must contain finite values or missing values.")
    aligned = pd.DataFrame({"taxon": taxa}).merge(
        aligned, on="taxon", how="left", validate="one_to_one"
    )
    aligned.insert(0, "method", method)
    return aligned


def evaluate_calls(
    calls: pd.DataFrame,
    feature_truth: pd.DataFrame,
    applicability: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    truth = feature_truth.copy()
    truth["taxon"] = truth["taxon"].astype(str)
    abundance = pd.to_numeric(
        truth.get(
            "biological_abundance_coefficient",
            pd.Series(0.0, index=truth.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    support = pd.to_numeric(
        truth.get(
            "biological_support_coefficient", pd.Series(0.0, index=truth.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    if "is_direct_biological_target" in truth:
        direct = boolean_series(
            truth["is_direct_biological_target"], "is_direct_biological_target"
        )
    else:
        direct = abundance.ne(0) | support.ne(0)
    truth = pd.DataFrame(
        {
            "taxon": truth["taxon"],
            "direct_target": direct,
            "truth_abundance_coefficient": abundance,
            "truth_support_coefficient": support,
            "truth_direction": np.sign(np.where(abundance.ne(0), abundance, support)).astype(int),
        }
    )
    table = calls.merge(truth, on="taxon", how="left", validate="one_to_one")
    significant = table["significant"].astype(bool)
    target = table["direct_target"].astype(bool)
    tp = int((significant & target).sum())
    outside = int((significant & ~target).sum())
    fn = int((~significant & target).sum())
    tn = int((~significant & ~target).sum())
    discoveries = int(significant.sum())
    targets = int(target.sum())
    nulls = int((~target).sum())
    strict = bool(applicability.get("da_fdr", False))
    power_supported = bool(applicability.get("da_power", False)) and targets > 0
    truth_mode = str(applicability.get("da_truth_mode", "unsupported"))
    table["outcome"] = np.select(
        [significant & target, significant & ~target, ~significant & target],
        ["true_positive", "outside_target_discovery", "false_negative"],
        default="not_discovered_non_target",
    )
    if strict:
        table.loc[significant & ~target, "outcome"] = "false_positive"
        table.loc[~significant & ~target, "outcome"] = "true_negative"
    signed = significant & target & table["truth_direction"].ne(0)
    sign_accuracy = None
    if "effect" in table and signed.any():
        estimated = np.sign(pd.to_numeric(table.loc[signed, "effect"], errors="coerce"))
        valid = estimated.notna() & estimated.ne(0)
        if valid.any():
            sign_accuracy = float(
                (
                    estimated.loc[valid].astype(int)
                    == table.loc[signed, "truth_direction"].loc[valid].astype(int)
                ).mean()
            )
    outside_fraction = float(outside / discoveries) if discoveries else 0.0
    metrics: dict[str, Any] = {
        "da_truth_mode": truth_mode,
        "strict_non_target_null": strict,
        "n_taxa": int(len(table)),
        "n_tested": int(table["tested"].sum()),
        "coverage": float(table["tested"].mean()),
        "n_direct_targets": targets,
        "n_strict_null_taxa": nulls if strict else None,
        "n_significant": discoveries,
        "direct_target_discoveries": tp,
        "outside_target_discoveries": outside,
        "missed_direct_targets": fn,
        "outside_target_fraction_descriptive": outside_fraction,
        "direct_target_power": float(tp / targets) if power_supported else None,
        "type_ii_error_rate": float(fn / targets) if power_supported else None,
        "direct_target_precision": float(tp / discoveries) if discoveries else None,
        "direct_target_sign_accuracy": sign_accuracy,
        "true_positives": tp if strict else None,
        "false_positives": outside if strict else None,
        "true_negatives": tn if strict else None,
        "false_negatives": fn if strict and targets else None,
        "type_i_error_rate": float(outside / nulls) if strict and nulls else None,
        "specificity": float(tn / nulls) if strict and nulls else None,
        "fdp": float(outside / max(discoveries, 1)) if strict else None,
        "precision": float(tp / discoveries) if strict and discoveries else None,
        "fwer_indicator": bool(outside > 0) if strict else None,
    }
    return metrics, table
