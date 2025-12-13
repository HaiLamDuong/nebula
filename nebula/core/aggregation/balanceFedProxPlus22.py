import gc
import logging
import math

import torch

from nebula.core.aggregation.aggregator import Aggregator


# Old Improve
class BalanceFedProxPlus22(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = 1.5  # filtering constant
        self.K = 1.0  # decay factor
        self.a = 0.5  # weight mix
        self.mu = 0.3  # FedProx regularization
        self.ema_beta = 0.8  # EMA smoothing for prox_center
        self.norm_clip_ratio = 3.0  # reject if ||wj|| > ratio * ||r_i||
        self.soft_k = 2.5  # softness factor for Gaussian decay

        # EMA buffer
        self._ema_prox_center = None

        logging.info(
            f"[{self.__class__.__name__}] Initialized with A={self.A}, K={self.K}, a={self.a}, mu={self.mu}, EMA={self.ema_beta}"
        )

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(
                f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models"
            )
        return models[self._addr]

    # --------------------------------------------------------------
    # 1) Compute neighbor average → apply EMA smoothing
    # --------------------------------------------------------------
    def compute_prox_center(self, models):
        """
        r_i = average of neighbors, smoothed by EMA to reduce malicious spikes.
        """
        local_addr = self._addr
        neighbor_models = [
            params for addr, (params, _) in models.items() if addr != local_addr
        ]

        if not neighbor_models:
            return self.get_local_model(models)[0]

        with torch.no_grad():
            raw_center = {}
            num_neighbors = len(neighbor_models)

            for layer in neighbor_models[0]:
                raw_center[layer] = torch.zeros_like(neighbor_models[0][layer])
                for m in neighbor_models:
                    raw_center[layer] += m[layer] / num_neighbors

            # Apply EMA smoothing
            if self._ema_prox_center is None:
                self._ema_prox_center = {k: v.clone() for k, v in raw_center.items()}
            else:
                for layer in raw_center:
                    self._ema_prox_center[layer] = (
                        self.ema_beta * self._ema_prox_center[layer]
                        + (1 - self.ema_beta) * raw_center[layer]
                    )

            return self._ema_prox_center

    def remove_malicious_models(self, models, prox_center):
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError:
            return models

        local_model, _ = self.get_local_model(models)

        wi_norm = math.sqrt(
            sum(torch.norm(p).item() ** 2 for p in local_model.values())
        )

        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * wi_norm

        filtered = {}

        for node_addr, (model_params, _) in models.items():
            if node_addr == self._addr:
                continue

            distance = math.sqrt(
                sum(
                    torch.norm(model_params[layer] - local_model[layer]).item() ** 2
                    for layer in local_model
                )
            )

            wj_norm = math.sqrt(
                sum(torch.norm(v).item() ** 2 for v in model_params.values())
            )
            norm_ratio = wj_norm / (wi_norm + 1e-9)

            if norm_ratio > self.norm_clip_ratio:
                logging.debug(
                    f"[{self.__class__.__name__}] Model {node_addr} REJECTED (norm ratio {norm_ratio:.2f})"
                )
                continue

            soft_weight = math.exp(
                -((distance / (threshold + 1e-9)) ** 2) * self.soft_k
            )

            logging.debug(
                f"[{self.__class__.__name__}] Node={node_addr} dist={distance:.4f} thr={threshold:.4f} "
                f"norm_ratio={norm_ratio:.2f} soft_w={soft_weight:.4f}"
            )

            filtered[node_addr] = (model_params, soft_weight)

        return filtered

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)
        filtered_models = self.remove_malicious_models(models, prox_center)

        if not filtered_models:
            logging.debug(
                f"[{self.__class__.__name__}] No models passed filtering; returning local model."
            )
            return local_model

        # Soft-weights normalization
        node_items = list(filtered_models.values())
        weights = [w for _, w in node_items]
        W = sum(weights)

        accum = {layer: torch.zeros_like(param) for layer, param in local_model.items()}

        with torch.no_grad():
            # Weighted average
            for params, soft_w in node_items:
                for layer in accum:
                    accum[layer] += params[layer] * (soft_w / W)

            # Combine
            for layer in accum:
                prox_term = self.mu * (local_model[layer] - prox_center[layer])
                accum[layer] = (
                    self.a * local_model[layer]
                    + (1 - self.a) * accum[layer]
                    - prox_term
                )

        del models, filtered_models
        gc.collect()
        logging.info(
            f"[{self.__class__.__name__}] BalanceFedProx (EMA + Soft Filter) aggregation completed."
        )
        self._ema_prox_center = accum
        return accum
