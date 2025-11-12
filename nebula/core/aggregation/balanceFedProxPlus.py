import gc
import torch
import math
import logging
import numpy as np
from copy import deepcopy

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxPlus(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        self.A = float(kwargs.get("A", 1.5))       # base factor for filtering
        self.K = float(kwargs.get("K", 1.0))       # decay factor for threshold
        self.a = float(kwargs.get("a", 0.4))      # local weight in final mixing
        self.mu = float(kwargs.get("mu", 0.1))    # FedProx regularization
        self.clip_norm = float(kwargs.get("clip_norm", 10.0))  # clip to avoid instability

        logging.info(f"[{self.__class__.__name__}] Init A={self.A}, K={self.K}, a={self.a}, mu={self.mu}")

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) missing")
        return models[self._addr]

    def compute_prox_center(self, models):
        local_addr = self._addr
        neighbor_models = [params for addr, (params, _) in models.items() if addr != local_addr]
        if not neighbor_models:
            return deepcopy(self.get_local_model(models)[0])

        with torch.no_grad():
            r_i = {k: torch.zeros_like(v) for k, v in neighbor_models[0].items()}
            inv = 1.0 / len(neighbor_models)
            for nb_model in neighbor_models:
                for k in r_i:
                    r_i[k].add_(nb_model[k], alpha=inv)
        return r_i

    def _l2_distance(self, s1, s2):
        total = 0.0
        for k in s1:
            diff = s1[k].float() - s2[k].float()
            total += float(torch.norm(diff, p=2).item() ** 2)
        return math.sqrt(total)

    def remove_malicious_models(self, models, prox_center):
        local_addr = self._addr

        distances = {}
        for addr, (params, _) in models.items():
            if addr == local_addr:
                continue
            distances[addr] = self._l2_distance(params, prox_center)

        if not distances:
            return {}

        dists = np.array(list(distances.values()))
        med = float(np.median(dists))
        mad = float(np.median(np.abs(dists - med))) + 1e-8

        try:
            current_round = getattr(self.engine, "round", 0) + 1
            total_rounds = max(1, getattr(self.engine, "total_rounds", 100))
        except Exception:
            current_round = 1
            total_rounds = 100

        decay = math.exp(-self.K * current_round / total_rounds)
        beta = 2.5 * (1.0 - current_round / total_rounds) + 1.0
        threshold = (med + beta * mad) * (self.A * decay)

        filtered = {
            addr: (params, w)
            for addr, (params, w) in models.items()
            if addr != local_addr and distances[addr] <= threshold
        }

        logging.debug(f"[{self.__class__.__name__}] Filter: kept {len(filtered)} / {len(distances)} models")
        return filtered

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model, _ = self.get_local_model(models)
        prox_center = self.compute_prox_center(models)
        filtered_models = self.remove_malicious_models(models, prox_center)

        if not filtered_models:
            logging.debug(f"[{self.__class__.__name__}] Không có neighbor hợp lệ, trả local model.")
            return deepcopy(local_model)

        filtered_list = list(filtered_models.values())
        S = len(filtered_list)
        accum = {k: torch.zeros_like(v) for k, v in local_model.items()}

        with torch.no_grad():
            for params, _ in filtered_list:
                for k in accum:
                    accum[k].add_(params[k], alpha=1.0 / S)

            # FedProx term + mixing
            result = {}
            for k in accum:
                prox_term = self.mu * (local_model[k] - prox_center[k])
                mixed = self.a * local_model[k] + (1 - self.a) * accum[k] - prox_term

                if self.clip_norm:
                    norm = torch.norm(mixed).item()
                    if norm > self.clip_norm:
                        mixed = mixed * (self.clip_norm / norm)
                result[k] = mixed.clone()

        del models, filtered_models
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] BalanceFedProxPlus aggregation done. Neighbors={S}")
        return result
