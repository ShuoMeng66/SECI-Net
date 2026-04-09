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

    def feature_distance(self, features_a: Tensor, features_b: Tensor) -> Tensor:
        features_a = nn.functional.normalize(features_a, dim=-1)
        features_b = nn.functional.normalize(features_b, dim=-1)
        cosine_similarity = (features_a * features_b).sum(dim=-1)
        return 1.0 - cosine_similarity

    def forward(
        self,
        factual_features: Tensor,
        counterfactual_features: Tensor,
        factual_labels: Tensor,
        counterfactual_labels: Tensor,
    ) -> Tensor:
        if factual_features.size(0) == 0:
            return factual_features.new_zeros(())

        feature_distance = self.feature_distance(factual_features, counterfactual_features)
        same_label_mask = factual_labels == counterfactual_labels
        different_label_mask = ~same_label_mask

        if same_label_mask.any():
            consistency_loss = feature_distance[same_label_mask].mean()
        else:
            consistency_loss = feature_distance.new_zeros(())

        if different_label_mask.any():
            intervention_loss = torch.relu(
                self.margin - feature_distance[different_label_mask]
            ).mean()
        else:
            intervention_loss = feature_distance.new_zeros(())

        return (
            self.consistency_weight * consistency_loss
            + self.intervention_weight * intervention_loss
        )


class RecoverabilityLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, recoverability_logits: Tensor, targets: Tensor) -> Tensor:
        if recoverability_logits.numel() == 0:
            return recoverability_logits.new_zeros(())
        return self.criterion(recoverability_logits.view(-1), targets.float().view(-1))


class EvidenceSparsityLoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, router_probabilities: Tensor, block_mask: Tensor) -> Tensor:
        if router_probabilities.numel() == 0:
            return router_probabilities.new_zeros(())

        entropy = -(
            router_probabilities * torch.log(router_probabilities.clamp_min(self.eps))
        ).sum(dim=-1)
        valid_blocks = block_mask.sum(dim=-1).clamp_min(2).float()
        normalized_entropy = entropy / torch.log(valid_blocks)
        return normalized_entropy.mean()
