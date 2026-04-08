import torch
from torch import Tensor, nn


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.label_smoothing <= 0.0:
            return nn.functional.cross_entropy(logits, targets)

        num_classes = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs,
                fill_value=self.label_smoothing / max(num_classes - 1, 1),
            )
            smooth_targets.scatter_(
                dim=1,
                index=targets.unsqueeze(1),
                value=1.0 - self.label_smoothing,
            )

        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss


class CounterfactualInterventionLoss(nn.Module):
    def __init__(
        self,
        consistency_weight: float = 0.5,
        intervention_weight: float = 0.5,
        margin: float = 0.2,
    ) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight
        self.intervention_weight = intervention_weight
        self.margin = margin

    def jensen_shannon_divergence(self, logits_a: Tensor, logits_b: Tensor) -> Tensor:
        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b, dim=-1)
        mean_probs = 0.5 * (probs_a + probs_b)

        kl_a = torch.sum(
            probs_a * (torch.log(probs_a.clamp_min(1e-8)) - torch.log(mean_probs.clamp_min(1e-8))),
            dim=-1,
        )
        kl_b = torch.sum(
            probs_b * (torch.log(probs_b.clamp_min(1e-8)) - torch.log(mean_probs.clamp_min(1e-8))),
            dim=-1,
        )
        return 0.5 * (kl_a + kl_b)

    def forward(
        self,
        factual_logits: Tensor,
        counterfactual_logits: Tensor,
        factual_labels: Tensor,
        counterfactual_labels: Tensor,
    ) -> Tensor:
        if factual_logits.size(0) == 0:
            return factual_logits.new_zeros(())

        js_divergence = self.jensen_shannon_divergence(factual_logits, counterfactual_logits)
        same_label_mask = factual_labels == counterfactual_labels
        different_label_mask = ~same_label_mask

        if same_label_mask.any():
            consistency_loss = js_divergence[same_label_mask].mean()
        else:
            consistency_loss = js_divergence.new_zeros(())

        if different_label_mask.any():
            intervention_loss = torch.relu(
                self.margin - js_divergence[different_label_mask]
            ).mean()
        else:
            intervention_loss = js_divergence.new_zeros(())

        return (
            self.consistency_weight * consistency_loss
            + self.intervention_weight * intervention_loss
        )
