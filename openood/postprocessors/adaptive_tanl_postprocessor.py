import math

import torch
from torch.nn.functional import normalize

from openood.utils.config import Config

from .activated_neg_postprocessor import ActivatedNegPostprocessor


class AdaptiveTANLPostprocessor(ActivatedNegPostprocessor):
    """TANL with real-image gates, coactivation reranking and visual cache scoring."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.tail = self.args.admission_tail
        self.admission_side = self.args.get("admission_side", "both")
        self.window = self.args.get("admission_window", 1000)
        self.correlation = self.args.diversity_correlation
        self.use_visual_cache = self.args.get("visual_cache", False)
        if self.tail is not None and not (0 < self.tail < 0.5 and self.window > 0):
            raise ValueError(
                "admission_tail must be in (0, 0.5) and the window positive"
            )
        if self.correlation is not None and not 0 < self.correlation <= 1:
            raise ValueError("diversity_correlation must be in (0, 1]")
        if self.admission_side not in ("both", "positive", "negative"):
            raise ValueError("admission_side must be both, positive or negative")
        self.image_score_history: torch.Tensor | None = None
        self.real_positive_count = self.real_negative_count = 0

    def reset_memory(self) -> None:
        super().reset_memory()
        self.image_score_history = None
        self.real_positive_count = self.real_negative_count = 0

    def _admission_masks(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tail is None:
            return super()._admission_masks(scores)
        # Warm-start on the first batch. Later batches use only earlier real images.
        history = self.image_score_history
        reference = scores if history is None else history
        low, high = torch.quantile(
            reference.float(),
            reference.new_tensor([self.tail, 1 - self.tail], dtype=torch.float32),
        ).unbind()
        updated = scores if history is None else torch.cat((history, scores))
        self.image_score_history = updated[-self.window :].detach()
        positive, negative = scores > high, scores < low
        if self.admission_side != "both":
            otsu_positive, otsu_negative = super()._admission_masks(scores)
            if self.admission_side == "positive":
                negative = otsu_negative & ~positive
            else:
                positive = otsu_positive & ~negative
        return positive, negative

    def _rank_batch_negatives(
        self, scores: torch.Tensor, activations: torch.Tensor, count: int
    ) -> torch.Tensor:
        if self.correlation is None or len(activations) < 2 or count == 0:
            return super()._rank_batch_negatives(scores, activations, count)
        shortlist = super()._rank_batch_negatives(scores, activations, 2 * count)
        profiles = activations.index_select(1, shortlist).float()
        profiles = normalize(profiles - profiles.mean(0), dim=0)
        # A label is deferred if a higher-ranked candidate has the same visual response.
        overlap = (profiles.T @ profiles).triu(diagonal=1).amax(0)
        keep = overlap < self.correlation
        return torch.cat((shortlist[keep], shortlist[~keep]))[:count]

    def _score_with_visual_cache(
        self,
        confidence: torch.Tensor,
        images: torch.Tensor,
        scale: torch.Tensor,
        new_positives: int,
        new_negatives: int,
    ) -> torch.Tensor:
        if not self.use_visual_cache:
            return confidence
        # Both outputs are log odds, including the warm-up without real neighbors.
        log_odds = torch.logit(confidence.float(), eps=1e-7)
        if self.real_positive_count and self.real_negative_count:
            # Real images are the tail of each previous FIFO. Never compare to the
            # synthetic initialization or to an image in the batch being scored.
            def log_density(cache: torch.Tensor, count: int) -> torch.Tensor:
                logits = (images.float() @ cache[-count:].float().T) * scale.float()
                return logits.logsumexp(dim=1) - math.log(count)

            log_odds = log_odds + (
                log_density(self.cached_pos_features, self.real_positive_count)
                - log_density(self.cached_neg_features, self.real_negative_count)
            )
        self.real_positive_count = min(
            self.queue_len, self.real_positive_count + new_positives
        )
        self.real_negative_count = min(
            self.queue_len, self.real_negative_count + new_negatives
        )
        return log_odds
