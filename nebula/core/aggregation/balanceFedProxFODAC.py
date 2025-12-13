import gc
import torch
import math
import logging
import numpy as np
from copy import deepcopy

from nebula.core.aggregation.aggregator import Aggregator


class BalanceFedProxFODAC(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Hyperparameters
        self.A = float(kwargs.get("A", 2.0))       # balance filtering constant
        self.K = float(kwargs.get("K", 1.0))       # decay factor
        self.a = float(kwargs.get("a", 0.4))      # weight giữa local và neighbor
        self.mu = float(kwargs.get("mu", 0.1))    # FedProx regularization
        self.clip_norm = float(kwargs.get("clip_norm", 10.0))

        # topology
        self.adj = kwargs.get("adjacency_matrix", None)
        self.W = self._build_W(self.adj)

        # FODAC states
        self.x = None          # dynamic consensus state
        self.r_prev = None     # previous reference (local model)
        self.prev_local = None # previous local model
        self.device = torch.device(kwargs.get("device", "cpu"))

        logging.info(f"[{self.__class__.__name__}] Init A={self.A}, K={self.K}, a={self.a}, mu={self.mu}")

    def _build_W(self, adj):
        if adj is None:
            return None
        A = np.array(adj, dtype=float)
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 1)
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

    def compute_prox_center(self, models):
        local_addr = self._addr
        local_state, _ = models[local_addr]
        local_state = self._clone_state(local_state)

        # init lần đầu
        if self.x is None or self.r_prev is None:
            self.x = self._clone_state(local_state)
            self.r_prev = self._clone_state(local_state)
            self.prev_local = self._clone_state(local_state)
            return deepcopy(self.x)

        # compute Δr_i(t)
        delta_r = {k: (local_state[k] - self.r_prev[k]).detach().clone() for k in local_state}

        # neighbor consensus
        addrs = list(models.keys())
        neighbors = [addr for addr in addrs if addr != local_addr]
        delta_consensus = self._zeros_like_state(self.x)

        with torch.no_grad():
            for nb in neighbors:
                nb_state, _ = models[nb]
                nb_state = {k: v.to(self.device) for k, v in nb_state.items()}
                if self.W is not None:
                    idx_i = addrs.index(local_addr)
                    idx_j = addrs.index(nb)
                    w_ij = float(self.W[idx_i, idx_j])
                else:
                    w_ij = 1.0 / len(neighbors)
                for k in delta_consensus:
                    delta_consensus[k] += w_ij * (nb_state[k] - self.x[k])

            # update x
            x_new = {}
            for k in self.x:
                x_new[k] = (self.x[k] + delta_consensus[k] + delta_r[k]).detach().clone()

            self.x = deepcopy(x_new)
            self.r_prev = deepcopy(local_state)
            self.prev_local = deepcopy(local_state)

        return deepcopy(self.x)

    def remove_malicious_models(self, models, prox_center):
        try:
            current_round = getattr(self.engine, "round", 0) + 1
            total_rounds = max(1, getattr(self.engine, "total_rounds", 100))
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return models

        prox_norm = math.sqrt(sum(torch.norm(p, p=2).item() ** 2 for p in prox_center.values()))
        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * prox_norm

        local_addr = self._addr
        filtered = {}

        for addr, (params, w) in models.items():
            if addr == local_addr:
                continue
            dist = self._l2_distance(params, prox_center)
            if dist <= threshold:
                filtered[addr] = (params, w)
                logging.debug(f"[{self.__class__.__name__}] accept {addr} (dist={dist:.4f} ≤ thr={threshold:.4f})")
            else:
                logging.debug(f"[{self.__class__.__name__}] reject {addr} (dist={dist:.4f} > thr={threshold:.4f})")

        return filtered

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_model, _ = models[self._addr]
        local_model = self._clone_state(local_model)

        # 1. prox center qua FODAC
        prox_center = self.compute_prox_center(models)

        # 2. lọc theo balance gốc
        filtered = self.remove_malicious_models(models, prox_center)

        if not filtered:
            logging.debug(f"[{self.__class__.__name__}] No valid neighbor; return local model.")
            return deepcopy(local_model)

        # 3. tính trung bình neighbor
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

                if self.clip_norm:
                    norm = torch.norm(mixed).item()
                    if norm > self.clip_norm:
                        mixed = mixed * (self.clip_norm / norm)

                result[k] = mixed.clone()

        del models, filtered
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] Aggregation done. Neighbors={S}")
        return result
