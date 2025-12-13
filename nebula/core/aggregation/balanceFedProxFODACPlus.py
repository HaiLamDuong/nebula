import gc
import torch
import math
import logging
import numpy as np
from copy import deepcopy

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxFODACPlus(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = float(kwargs.get("A", 2.5))
        self.K = float(kwargs.get("K", 1.0))
        self.a = float(kwargs.get("a", 0.4))
        self.mu = float(kwargs.get("mu", 0.1))
        self.clip_norm = float(kwargs.get("clip_norm", 10.0))

        # topology
        self.adj = kwargs.get("adjacency_matrix", None)
        self.W = self._build_W(self.adj)

        # FODAC states
        self.x = None
        self.r_prev = None
        self.prev_local = None
        self.device = torch.device(kwargs.get("device", "cpu"))

        logging.info(f"[{self.__class__.__name__}] Init A={self.A}, K={self.K}, a={self.a}, mu={self.mu}")

    def _build_W(self, adj):
        """
        - Nếu adj = None -> dùng Metropolis-Hastings theo số neighbors
        - Nếu không có neighbor -> W = Identity
        """
        if adj is None:
            logging.warning("[FODAC] adjacency_matrix=None -> using dynamic Metropolis weights")
            return None  # sẽ build dynamic trong compute_prox_center()

        A = np.array(adj, dtype=float)
        A = np.maximum(A, A.T)  # enforce symmetry
        np.fill_diagonal(A, 1)

        # Row-normalize
        W = A / A.sum(axis=1, keepdims=True)
        return W

    def _clone_state(self, state):
        return {k: v.detach().clone().to(self.device) for k, v in state.items()}

    def _zeros_like_state(self, state):
        return {k: torch.zeros_like(v) for k, v in state.items()}

    def _l2_distance(self, s1, s2):
        total = 0.0
        for k in s1:
            diff = s1[k].float() - s2[k].float()
            total += float(torch.norm(diff, p=2).item() ** 2)
        return math.sqrt(total)

    # FODAC
    def compute_prox_center(self, models):

        local_addr = self._addr
        local_state, _ = models[local_addr]
        local_state = self._clone_state(local_state)

        if self.x is None:
            self.x = self._clone_state(local_state)
            self.r_prev = self._clone_state(local_state)
            self.prev_local = self._clone_state(local_state)
            return deepcopy(self.x)

        delta_r = {
            k: (local_state[k] - self.r_prev[k]).detach().clone()
            for k in local_state
        }

        addrs = list(models.keys())
        neighbors = [addr for addr in addrs if addr != local_addr]

        if len(neighbors) == 0:
            logging.warning("[FODAC] No neighbors -> x(t+1) = x(t) + Δr")
            x_new = {k: (self.x[k] + delta_r[k]).detach().clone() for k in self.x}
            self.x = deepcopy(x_new)
            self.r_prev = deepcopy(local_state)
            return deepcopy(self.x)

        delta_consensus = self._zeros_like_state(self.x)

        with torch.no_grad():
            for nb in neighbors:
                nb_state, _ = models[nb]
                nb_state = {k: v.to(self.device) for k, v in nb_state.items()}

                if self.W is None:
                    deg_i = len(neighbors)
                    deg_j = len([a for a in models.keys() if a != nb])
                    w_ij = 1.0 / (1 + max(deg_i, deg_j))
                else:
                    idx_i = addrs.index(local_addr)
                    idx_j = addrs.index(nb)
                    w_ij = float(self.W[idx_i, idx_j])

                for k in delta_consensus:
                    delta_consensus[k] += w_ij * (nb_state[k] - self.x[k])

            x_new = {}
            for k in self.x:
                x_new[k] = (self.x[k] + delta_consensus[k] + delta_r[k]).detach().clone()

            self.x = deepcopy(x_new)
            self.r_prev = deepcopy(local_state)
            self.prev_local = deepcopy(local_state)

        return deepcopy(self.x)

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
            prox_norm += torch.norm(param, p=2).item() ** 2
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
                distance += torch.norm(diff, p=2).item() ** 2
            distance = math.sqrt(distance)

            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} accepted (dist={distance:.4f} ≤ thr={threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] Model {node_addr} rejected (dist={distance:.4f} > thr={threshold:.4f})")

        return filtered_models

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model, _ = models[self._addr]
        local_model = self._clone_state(local_model)

        # 1. prox center via FODAC
        prox_center = self.compute_prox_center(models)

        # 2. filtering
        filtered = self.remove_malicious_models(models, prox_center)

        if not filtered:
            logging.debug(f"[{self.__class__.__name__}] No valid neighbor; return local model.")
            return deepcopy(local_model)

        # 3. average neighbor models
        S = len(filtered)
        accum = self._zeros_like_state(local_model)

        with torch.no_grad():
            for params, _ in filtered.values():
                for k in accum:
                    accum[k].add_(params[k], alpha=1.0 / S)

            result = {}
            for k in accum:
                prox_term = self.mu * (local_model[k] - prox_center[k])
                mixed = self.a * local_model[k] + (1 - self.a) * accum[k] - prox_term

                # global model clipping
                if self.clip_norm:
                    norm = torch.norm(mixed).item()
                    if norm > self.clip_norm:
                        mixed = mixed * (self.clip_norm / norm)

                result[k] = mixed.clone()

        del models, filtered
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] Aggregation done. Neighbors={S}")
        return result
