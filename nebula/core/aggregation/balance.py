import gc
import torch
import math
import logging

from nebula.core.aggregation.aggregator import Aggregator

class Balance(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Constant for Balance algorithm
        self.A = 2
        self.K = 1
        self.a = 0.4
        logging.info(f"[{self.__class__.__name__}] Initializing BALANCE with A={self.A}, K={self.K}, a={self.a}")

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        return models[self._addr]

    def remove_malicious_models(self, models):
        """
        Filter models based on distance to local model using the formula:
        ||wi - wj|| <= A * exp(-K * t / T) * ||wi||

        Args:
            models (dict): Dictionary of model updates, where keys are node addresses
                          and values are tuples of (model_parameters, weight).

        Returns:
            dict: Filtered models satisfying the distance condition.
        """
        # Get round information
        try:
            current_round = self.engine.round + 1
            logging.debug(f"[{self.__class__.__name__}] Current round: {current_round}")
            total_rounds = self.engine.total_rounds
            logging.debug(f"[{self.__class__.__name__}] Total round: {total_rounds}")
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round information: {e}")
            logging.error(f"[{self.__class__.__name__}] Remove malicious models failed: {e}")
            return models

        # Get current local model (wi)
        local_model, _ = self.get_local_model(models)

        # Get norm of current local model ||wi||
        local_norm = 0.0
        for param in local_model.values():
            local_norm += torch.norm(param, p=2).item() ** 2
        local_norm = math.sqrt(local_norm)
        logging.debug(f"[{self.__class__.__name__}] Local model norm ||wi||: {local_norm:.4f}")

        # Threashold = A * exp(-K * t / T) * ||wi||
        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * local_norm

        # Remove malicious models
        filtered_models = {}
        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                # filtered_models[node_addr] = (model_params, weight)  # Luôn giữ mô hình cục bộ
                continue

            # Calculate distance Euclidean ||wi - wj||
            distance = 0.0
            for layer in local_model:
                diff = local_model[layer] - model_params[layer]
                distance += torch.norm(diff, p=2).item() ** 2
            distance = math.sqrt(distance)

            # check the condition
            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Model from {node_addr} accepted (distance: {distance:.4f} <= threshold: {threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] Model from {node_addr} rejected (distance: {distance:.4f} > threshold: {threshold:.4f})")

        return filtered_models


    def run_aggregation(self, models):
        """
        Implements Balance aggregation:
        1. Filter models using remove_malicious_models.
        2. Aggregate using result = a * wi + (1-a) * (1/S) * sum(wj).

        Args:
            models (dict): Dictionary of model updates, where keys are node addresses
                          and values are tuples of (model_parameters, weight).
        Returns:
            dict: Aggregated model parameters.
        """
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)

        filtered_models = self.remove_malicious_models(models)
        if not filtered_models:
            logging.debug("No models left after filtering")
            return local_model

        filtered_models = list(filtered_models.values())
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}

        S = len(filtered_models)

        with torch.no_grad():
            if S == 0:
                logging.debug("No other models to aggregate, returning local model")
                for layer in accum:
                    accum[layer] = local_model[layer]
            else:
                for params, _ in filtered_models:
                    for layer in accum:
                        accum[layer].add_(params[layer], alpha=1.0 / S)

                # result = a * wi + (1-a) * (1/S) * sum(wj)
                for layer in accum:
                    accum[layer].mul_(1 - self.a)  # (1-a) * (1/S) * sum(wj)
                    accum[layer].add_(local_model[layer], alpha=self.a)  # a * wi

        del models, filtered_models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] BALANCE Aggregation completed successfully.")
        return accum
