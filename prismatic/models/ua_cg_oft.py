"""Lightweight UA-CG-OFT modules for uncertainty-aware and counterfactual grounded fine-tuning."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UACGOFTOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]
    uncertainty: Optional[torch.Tensor] = None
    grounding_logits: Optional[torch.Tensor] = None


def _as_float_metric(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


class UncertaintyPredictionHead(nn.Module):
    """Predicts a scalar risk score from action-token hidden states and predicted action statistics."""

    def __init__(self, llm_dim: int, action_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        stats_dim = action_dim * 3
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(llm_dim + stats_dim),
            nn.Linear(llm_dim + stats_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, actions_hidden_states: torch.Tensor, predicted_actions: Optional[torch.Tensor]) -> torch.Tensor:
        pooled_hidden = actions_hidden_states.mean(dim=1)

        if predicted_actions is None:
            action_stats = torch.zeros(
                pooled_hidden.shape[0],
                self.action_dim * 3,
                device=pooled_hidden.device,
                dtype=pooled_hidden.dtype,
            )
        else:
            action_stats = torch.cat(
                [
                    predicted_actions.mean(dim=1),
                    predicted_actions.std(dim=1, unbiased=False),
                    predicted_actions[:, -1] - predicted_actions[:, 0],
                ],
                dim=-1,
            )

        features = torch.cat([pooled_hidden, action_stats.to(pooled_hidden.dtype)], dim=-1)
        return torch.sigmoid(self.net(features.to(_module_dtype(self)))).float().squeeze(-1)


class GroundingHead(nn.Module):
    """Predicts language-conditioned target response maps from projected ViT patch features."""

    def __init__(
        self,
        llm_dim: int,
        patches_per_image: int,
        num_images: int = 1,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or llm_dim
        side = int(patches_per_image**0.5)
        if side * side != patches_per_image:
            raise ValueError(f"patches_per_image must be square, got {patches_per_image}")

        self.patches_per_image = patches_per_image
        self.num_images = num_images
        self.patch_side = side
        self.context_proj = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_dim),
        )
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, projector_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        expected_patches = self.patches_per_image * self.num_images
        patch_features = projector_features[:, :expected_patches]
        context = self.context_proj(context_features.to(_module_dtype(self))).unsqueeze(1)
        patch_features = patch_features.to(_module_dtype(self)) + context
        logits = self.patch_proj(patch_features).squeeze(-1).float()
        return logits.reshape(
            logits.shape[0],
            self.num_images,
            self.patch_side,
            self.patch_side,
        )


def resize_target_masks(target_masks: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Resizes BxHxW, Bx1xHxW, or BxIxHxW masks to the grounding-logit shape."""
    masks = target_masks.float().to(logits.device)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4:
        raise ValueError(f"target masks must have rank 3 or 4, got shape {tuple(masks.shape)}")

    if masks.shape[1] == 1 and logits.shape[1] > 1:
        first = F.interpolate(masks, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        zeros = torch.zeros(
            logits.shape[0],
            logits.shape[1] - 1,
            logits.shape[-2],
            logits.shape[-1],
            device=logits.device,
            dtype=first.dtype,
        )
        masks = torch.cat([first, zeros], dim=1)
    else:
        masks = F.interpolate(masks[:, : logits.shape[1]], size=logits.shape[-2:], mode="bilinear", align_corners=False)

    return masks.clamp(0.0, 1.0)


def dice_loss_from_logits(logits: torch.Tensor, target_masks: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = target_masks.reshape(target_masks.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def masked_patch_pool(projector_features: torch.Tensor, masks: torch.Tensor, patches_per_image: int) -> torch.Tensor:
    batch_size, num_images, height, width = masks.shape
    patch_features = projector_features[:, : patches_per_image * num_images]
    patch_features = patch_features.reshape(batch_size, num_images, patches_per_image, -1)
    weights = masks.reshape(batch_size, num_images, patches_per_image)
    weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    pooled = (patch_features * weights.unsqueeze(-1)).sum(dim=(1, 2))
    return F.normalize(pooled.float(), dim=-1)


class UACGOFTModule(nn.Module):
    """Combines uncertainty calibration, target grounding, contrastive grounding, and action equivariance losses."""

    def __init__(
        self,
        llm_dim: int,
        action_dim: int,
        patches_per_image: int,
        num_images: int = 1,
        hidden_dim: Optional[int] = None,
        uncertainty_loss_weight: float = 1.0,
        grounding_loss_weight: float = 1.0,
        grounding_dice_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        equivariance_loss_weight: float = 1.0,
        uncertainty_error_scale: float = 1.0,
        contrastive_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.uncertainty_head = UncertaintyPredictionHead(llm_dim, action_dim, hidden_dim)
        self.grounding_head = GroundingHead(llm_dim, patches_per_image, num_images, hidden_dim)
        self.patches_per_image = patches_per_image
        self.num_images = num_images
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.grounding_loss_weight = grounding_loss_weight
        self.grounding_dice_weight = grounding_dice_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.equivariance_loss_weight = equivariance_loss_weight
        self.uncertainty_error_scale = max(uncertainty_error_scale, 1e-6)
        self.contrastive_temperature = contrastive_temperature

    def _risk_targets(self, predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> torch.Tensor:
        errors = (predicted_actions.float().detach() - ground_truth_actions.float()).abs().mean(dim=(1, 2))
        return (errors / self.uncertainty_error_scale).clamp(0.0, 1.0)

    def _contrastive_loss(
        self,
        context_features: torch.Tensor,
        projector_features: torch.Tensor,
        target_masks: torch.Tensor,
        negative_target_masks: torch.Tensor,
        grounding_logits: torch.Tensor,
    ) -> torch.Tensor:
        positive_masks = resize_target_masks(target_masks, grounding_logits)
        query = F.normalize(context_features.float(), dim=-1)
        positive = masked_patch_pool(projector_features, positive_masks, self.patches_per_image)

        negative_masks = negative_target_masks.float().to(projector_features.device)
        if negative_masks.ndim == 4:
            negative_masks = negative_masks.unsqueeze(1)
        if negative_masks.ndim != 5:
            raise ValueError(
                "negative_target_masks must be BxKxHxW or BxKxIxHxW, "
                f"got {tuple(negative_masks.shape)}"
            )

        negatives = []
        for idx in range(negative_masks.shape[1]):
            resized = resize_target_masks(negative_masks[:, idx], grounding_logits)
            negatives.append(masked_patch_pool(projector_features, resized, self.patches_per_image))
        negative = torch.stack(negatives, dim=1)

        pos_logits = (query * positive).sum(dim=-1, keepdim=True)
        neg_logits = (query.unsqueeze(1) * negative).sum(dim=-1)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.contrastive_temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        actions_hidden_states: torch.Tensor,
        predicted_actions: Optional[torch.Tensor] = None,
        ground_truth_actions: Optional[torch.Tensor] = None,
        projector_features: Optional[torch.Tensor] = None,
        target_masks: Optional[torch.Tensor] = None,
        negative_target_masks: Optional[torch.Tensor] = None,
        counterfactual_predicted_actions: Optional[torch.Tensor] = None,
        counterfactual_target_actions: Optional[torch.Tensor] = None,
    ) -> UACGOFTOutput:
        device = actions_hidden_states.device
        loss = actions_hidden_states.float().new_zeros(())
        metrics: Dict[str, float] = {}

        uncertainty = self.uncertainty_head(actions_hidden_states, predicted_actions)
        if predicted_actions is not None and ground_truth_actions is not None:
            risk_targets = self._risk_targets(predicted_actions, ground_truth_actions)
            uncertainty_loss = F.mse_loss(uncertainty, risk_targets)
            loss = loss + self.uncertainty_loss_weight * uncertainty_loss
            metrics["ua_uncertainty_loss"] = _as_float_metric(uncertainty_loss)
            metrics["ua_predicted_uncertainty"] = _as_float_metric(uncertainty.mean())
            metrics["ua_target_risk"] = _as_float_metric(risk_targets.mean())

        grounding_logits = None
        context_features = actions_hidden_states.mean(dim=1)
        if projector_features is not None and target_masks is not None:
            grounding_logits = self.grounding_head(projector_features, context_features)
            resized_masks = resize_target_masks(target_masks, grounding_logits)
            bce_loss = F.binary_cross_entropy_with_logits(grounding_logits, resized_masks)
            dice_loss = dice_loss_from_logits(grounding_logits, resized_masks)
            grounding_loss = bce_loss + self.grounding_dice_weight * dice_loss
            loss = loss + self.grounding_loss_weight * grounding_loss
            metrics["cg_grounding_bce_loss"] = _as_float_metric(bce_loss)
            metrics["cg_grounding_dice_loss"] = _as_float_metric(dice_loss)
            metrics["cg_grounding_loss"] = _as_float_metric(grounding_loss)

            if negative_target_masks is not None:
                contrastive_loss = self._contrastive_loss(
                    context_features,
                    projector_features,
                    target_masks,
                    negative_target_masks,
                    grounding_logits,
                )
                loss = loss + self.contrastive_loss_weight * contrastive_loss
                metrics["cg_contrastive_loss"] = _as_float_metric(contrastive_loss)

        if counterfactual_predicted_actions is not None and counterfactual_target_actions is not None:
            equivariance_loss = F.l1_loss(
                counterfactual_predicted_actions.float(),
                counterfactual_target_actions.float().to(device),
            )
            loss = loss + self.equivariance_loss_weight * equivariance_loss
            metrics["cg_equivariance_loss"] = _as_float_metric(equivariance_loss)

        metrics["ua_cg_loss"] = _as_float_metric(loss)
        return UACGOFTOutput(
            loss=loss,
            metrics=metrics,
            uncertainty=uncertainty,
            grounding_logits=grounding_logits,
        )
