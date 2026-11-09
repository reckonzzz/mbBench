from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    masked_logits = logits.masked_fill(mask <= 0, float("-inf"))
    probs = torch.softmax(masked_logits, dim=dim)
    probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
    totals = probs.sum(dim=dim, keepdim=True)
    return probs / totals.clamp_min(eps)


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = ((1.0 - pt) ** gamma) * bce
    per_sample = focal.mean(dim=-1)
    if sample_weights is None:
        return per_sample.mean()
    weights = sample_weights.to(per_sample).clamp_min(0)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def abundance_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support_mask: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    support = support_mask.to(elementwise.dtype)
    per_sample = (elementwise * support).sum(dim=-1) / support.sum(dim=-1).clamp_min(
        1.0
    )
    valid = support.sum(dim=-1) > 0
    if not valid.any():
        return elementwise.sum() * 0.0
    if sample_weights is None:
        return per_sample[valid].mean()
    weights = sample_weights.to(per_sample).clamp_min(0) * valid.to(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def clr_transform(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clipped = values.clamp_min(eps)
    geometric_mean = clipped.log().mean(dim=-1, keepdim=True)
    return clipped.log() - geometric_mean


def masked_compositional_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    support = mask > 0
    sample_losses: list[torch.Tensor] = []
    sample_indices: list[int] = []
    for sample_index in range(prediction.shape[0]):
        active = support[sample_index]
        if int(active.sum().item()) < 2:
            continue
        pred = prediction[sample_index, active]
        tgt = target[sample_index, active]
        pred = pred / pred.sum().clamp_min(eps)
        tgt = tgt / tgt.sum().clamp_min(eps)
        pred_clr = clr_transform(pred.unsqueeze(0), eps=eps)
        tgt_clr = clr_transform(tgt.unsqueeze(0), eps=eps)
        sample_losses.append(F.mse_loss(pred_clr, tgt_clr))
        sample_indices.append(sample_index)
    if not sample_losses:
        return prediction.sum() * 0.0
    losses = torch.stack(sample_losses)
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.to(losses)[
        torch.as_tensor(sample_indices, device=losses.device)
    ].clamp_min(0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def kl_beta_schedule(epoch_index: int, warmup_epochs: int, beta_max: float) -> float:
    if warmup_epochs <= 0:
        return float(beta_max)
    progress = min(max((epoch_index + 1) / warmup_epochs, 0.0), 1.0)
    return float(beta_max * progress)
