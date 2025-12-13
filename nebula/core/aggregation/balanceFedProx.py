import gc
import torch
import math
import logging

from nebula.core.aggregation.aggregator import Aggregator

class BalanceFedProx(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Hyperparameters
        self.A = 1.5           # balance filtering constant
        self.K = 1.0           # decay factor
        self.a = 0.5           # weight between local and neighbors
        self.mu = 0.1         # FedProx regularization strength
        logging.info(f"[{self.__class__.__name__}] Initialized with A={self.A}, K={self.K}, a={self.a}, mu={self.mu}")

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        return models[self._addr]

    def compute_prox_center(self, models):
        """
        Compute r_i = average of all neighbor models (excluding self)
        """
        local_addr = self._addr
        neighbor_models = [params for addr, (params, _) in models.items() if addr != local_addr]
        if not neighbor_models:
            # fallback to local if no neighbor
            return self.get_local_model(models)[0]

        with torch.no_grad():
            r_i = {}
            num_neighbors = len(neighbor_models)
            for layer in neighbor_models[0]:
                r_i[layer] = torch.zeros_like(neighbor_models[0][layer] if neighbor_models[0][layer].is_floating_point() else neighbor_models[0][layer].float())
                for model in neighbor_models:
                    tensor = model[layer] if model[layer].is_floating_point() else model[layer].float()
                    r_i[layer] += tensor / num_neighbors
        return r_i

    def remove_malicious_models(self, models, prox_center):
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return models

        local_model, _ = self.get_local_model(models)

        local_norm = math.sqrt(sum(torch.norm(p, p=2).item() ** 2 for p in local_model.values()))

        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * local_norm

        filtered_models = {}

        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                continue

            distance = math.sqrt(sum(
                (torch.norm(local_model[layer] - model_params[layer], p=2).item() ** 2)
                for layer in local_model
            ))

            logging.debug(
                f"[{self.__class__.__name__}] Node={node_addr} dist={distance:.4f}, thr={threshold:.4f}"
            )

            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] accepted {node_addr}")
            else:
                logging.debug(f"[{self.__class__.__name__}] rejected {node_addr}")

        return filtered_models
    def run_aggregation(self, models):
        """
        FedProx-enhanced Balance aggregation:
        1. Compute prox center r_i
        2. Filter models using prox-aware condition
        3. Aggregate with FedProx proximal correction
        """
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)
        filtered_models = self.remove_malicious_models(models, prox_center)

        if not filtered_models:
            logging.debug(f"[{self.__class__.__name__}] No models passed filtering; returning local model.")
            return local_model

        filtered_models = list(filtered_models.values())
        S = len(filtered_models)

        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}

        with torch.no_grad():
            for params, _ in filtered_models:
                for layer in accum:
                    accum[layer].add_(params[layer].to(accum[layer].dtype), alpha=1.0 / S)

            # result = a * wi + (1-a)*avg(wj) - mu*(wi - r_i)
            for layer in accum:
                prox_term = self.mu * (local_model[layer] - prox_center[layer])
                accum[layer] = (
                    self.a * local_model[layer] +
                    (1 - self.a) * accum[layer] -
                    prox_term
                )

        del models, filtered_models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] BalanceFedProx aggregation completed.")
        return accum
