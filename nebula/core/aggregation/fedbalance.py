import gc
import torch
import math
import logging

from nebula.core.aggregation.aggregator import Aggregator

class FedBalance(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        # Constant for Balance algorithm
        self.a = 0.2
        logging.info(f"[{self.__class__.__name__}] Initializing FED BALANCE with A={self.a} ")

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        # Unpack tuple to get model_params only
        model_params, _ = models[self._addr]
        return model_params

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model = self.get_local_model(models)


        S = len(models) - 1 # skip local mocal
        if S == 0:
            logging.debug("No other models to aggregate, returning local model")
            return local_model

        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}

        with torch.no_grad():
            # result = a * wi + (1-a) * (1/S) * sum(wj)
            for node_addr, (model_parameters, _) in models.items():
                if node_addr == self._addr:
                    continue
                for layer in accum:
                    accum[layer].add_(model_parameters[layer].to(accum[layer].dtype), alpha=1.0 / S)
            # (1-a) * (1/S) * sum(wj)
            for layer in accum:
                accum[layer].mul_(1 - self.a)
                accum[layer].add_(local_model[layer], alpha=self.a)  # a * wi

        gc.collect()
        logging.info(f"[{self.__class__.__name__}] FED BALANCE Aggregation completed successfully.")
        return accum
