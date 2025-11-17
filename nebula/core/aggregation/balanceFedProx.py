import gc
import torch
import math
import logging

from nebula.core.aggregation.aggregator import Aggregator

class BalanceFedProx(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Hyperparameters
        self.A = 2           # balance filtering constant
        self.K = 1.0           # decay factor
        self.a = 0.4           # weight between local and neighbors
        self.mu = 0.3         # FedProx regularization strength
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
                r_i[layer] = torch.zeros_like(neighbor_models[0][layer])
                for model in neighbor_models:
                    r_i[layer] += model[layer] / num_neighbors
        return r_i

    def remove_malicious_models(self, models, prox_center):
        """
        Filter models based on prox-aware distance:
        ||wj - r_i|| <= A * exp(-K * t / T) * ||r_i||
        """
        try:
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return models

        # Compute ||r_i||
        prox_norm = 0.0
        for param in prox_center.values():
            tensor = param if param.is_floating_point() else param.float()
            prox_norm += torch.norm(tensor, p=2).item() ** 2
        prox_norm = math.sqrt(prox_norm)

        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * prox_norm

        filtered_models = {}
        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                continue

            # Compute ||wj - r_i||
            distance = 0.0
            for layer in prox_center:
                diff = prox_center[layer] - model_params[layer]
                # Ensure diff is float
                if not diff.is_floating_point():
                    diff = diff.float()
                distance += torch.norm(diff, p=2).item() ** 2
            distance = math.sqrt(distance)

            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} accepted (dist={distance:.4f} ≤ thr={threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} rejected (dist={distance:.4f} > thr={threshold:.4f})")

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

        accum = {layer: torch.zeros_like(param) for layer, param in local_model.items()}

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
