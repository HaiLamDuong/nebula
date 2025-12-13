import gc
import logging
import math

import torch

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxPlus3(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = 1.5  # filtering constant
        self.K = 1.0  # decay factor
        self.a = 0.5  # weight mix
        self.mu = 0.3  # FedProx regularization
        self.ema_beta = 0.6  # EMA smoothing for prox_center
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

    def _compute_norm(self, model_params):
        """Compute L2 norm of model parameters efficiently."""
        # Keep computation in tensor space to avoid CPU-GPU syncs per layer
        return torch.sqrt(sum(torch.sum(p**2) for p in model_params.values()))

    def _compute_dist(self, model1, model2):
        """Compute Euclidean distance between two models efficiently."""
        # Keep computation in tensor space
        return torch.sqrt(sum(torch.sum((model1[l] - model2[l]) ** 2) for l in model1))

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
            num_neighbors = len(neighbor_models)

            # Initialize with the first neighbor's parameters
            raw_center = {k: v.clone() for k, v in neighbor_models[0].items()}

            # Sum remaining neighbors
            for i in range(1, num_neighbors):
                for layer, param in neighbor_models[i].items():
                    raw_center[layer] += param

            # Divide by N once at the end
            for layer in raw_center:
                raw_center[layer] /= num_neighbors

            # Apply EMA smoothing
            if self._ema_prox_center is None:
                self._ema_prox_center = {k: v.clone() for k, v in raw_center.items()}
            else:
                for layer in raw_center:
                    self._ema_prox_center[layer].lerp_(
                        raw_center[layer], 1 - self.ema_beta
                    )
                    # lerp_(end, weight) -> start + weight * (end - start)
                    # We want: beta * old + (1-beta) * new
                    # = old + (1-beta) * (new - old) -> lerp_(new, 1-beta) matches exactly

            return self._ema_prox_center

    def remove_malicious_models(self, models):
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError:
            return models

        local_model, _ = self.get_local_model(models)

        # Compute local model norm once, keep as scalar for threshold calc
        wi_norm_tensor = self._compute_norm(local_model)
        wi_norm = wi_norm_tensor.item()

        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * wi_norm

        filtered = {}

        for node_addr, (model_params, _) in models.items():
            if node_addr == self._addr:
                continue

            # Compute distance and norm efficiently
            distance_tensor = self._compute_dist(model_params, local_model)
            wj_norm_tensor = self._compute_norm(model_params)

            # Sync to CPU only once per model
            distance = distance_tensor.item()
            wj_norm = wj_norm_tensor.item()

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
        filtered_models = self.remove_malicious_models(models)
        prox_center = self.compute_prox_center(
            {self._addr: (local_model, 0), **filtered_models}
        )

        if not filtered_models:
            logging.debug(
                f"[{self.__class__.__name__}] No models passed filtering; returning local model."
            )
            return local_model

        # Soft-weights normalization
        node_items = list(filtered_models.values())
        weights = [w for _, w in node_items]
        W = sum(weights)

        # Initialize accumulator
        accum = {layer: torch.zeros_like(param) for layer, param in local_model.items()}

        with torch.no_grad():
            # Weighted average
            # Optimize: if only one model, skip loop overhead
            if len(node_items) == 1:
                params, soft_w = node_items[0]
                ratio = soft_w / W  # Should be 1.0 but keep for safety
                for layer in accum:
                    accum[layer].copy_(params[layer] * ratio)
            else:
                for params, soft_w in node_items:
                    ratio = soft_w / W
                    for layer in accum:
                        accum[layer].add_(params[layer], alpha=ratio)

            # Combine: accum = a * local + (1-a) * accum - mu * (local - prox)
            # Rearranged: accum = local * (a - mu) + accum * (1-a) + prox * mu
            c1 = self.a - self.mu
            c2 = 1.0 - self.a
            c3 = self.mu

            for layer in accum:
                # accum[layer] = c1 * local + c2 * accum + c3 * prox
                # Use in-place operations for efficiency
                accum[layer].mul_(c2).add_(local_model[layer], alpha=c1).add_(
                    prox_center[layer], alpha=c3
                )

        del models, filtered_models
        gc.collect()
        logging.info(
            f"[{self.__class__.__name__}] BalanceFedProx (EMA + Soft Filter) aggregation completed."
        )
        return accum
