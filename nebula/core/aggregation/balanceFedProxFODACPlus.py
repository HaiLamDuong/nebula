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

        # Hyperparameters (có thể expose qua config)
        self.A = float(kwargs.get("A", 1.5))        # filtering base
        self.K = float(kwargs.get("K", 1.0))        # decay factor for threshold
        self.alpha_mix = float(kwargs.get("a", 0.4))# weight between local and neighbors in final aggregation
        self.mu = float(kwargs.get("mu", 0.1))     # FedProx strength (regularizer)
        self.clip_norm = float(kwargs.get("clip_norm", 10.0))  # gradient/model clipping for stability
        self.device = torch.device(kwargs.get("device", "cpu"))

        # topology / mixing matrix
        self.adj = kwargs.get("adjacency_matrix", None)
        self.W = None               # numpy array (N,N) or None -> uniform
        self.addr2idx = {}          # mapping addr -> index for W
        self.idx2addr = {}
        self._build_W_from_topology(self.adj)

        # states for FODAC (per-node stored locally)
        # self.x : current consensus state (state_dict-like), self.r_prev : previous reference input (state_dict)
        # We store for current node only; prev_prox_center holds last r estimate per neighbor if needed
        self.x = None
        self.r_prev = None
        self.prev_local_model = None

        logging.info(f"[{self.__class__.__name__}] Init A={self.A}, K={self.K}, a={self.alpha_mix}, mu={self.mu}")


    def _sinkhorn_knopp(self, A, max_iter=200, tol=1e-6):
        A = np.array(A, dtype=float)
        # ensure symmetry
        A = (A + A.T) / 2.0
        # add small epsilon to avoid div by zero
        A += 1e-8
        for _ in range(max_iter):
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / row_sums
            col_sums = A.sum(axis=0, keepdims=True)
            A = A / col_sums
            if np.allclose(A.sum(axis=1), 1, atol=tol) and np.allclose(A.sum(axis=0), 1, atol=tol):
                break
        return A

    def _build_W_from_topology(self, adj):
        if adj is None:
            logging.warning(f"[{self.__class__.__name__}] No adjacency matrix provided. Mixing will be uniform when needed.")
            self.W = None
            return

        A = np.array(adj, dtype=float)
        # enforce symmetry and self-loop
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 1.0)
        W = self._sinkhorn_knopp(A)
        self.W = W
        # build addr2idx lazily when we see models (addresses may not be contiguous ints)
        logging.info(f"[{self.__class__.__name__}] Built mixing matrix W (shape={W.shape}).")
        return


    @staticmethod
    def _zeros_like_state(state):
        return {k: torch.zeros_like(v) for k, v in state.items()}

    @staticmethod
    def _clone_state(state):
        return {k: v.detach().clone() for k, v in state.items()}

    @staticmethod
    def _add_states(s1, s2, alpha=1.0):
        for k in s1:
            s1[k].add_(s2[k], alpha=alpha)
        return s1

    @staticmethod
    def _scale_state(state, scale):
        return {k: v * scale for k, v in state.items()}

    @staticmethod
    def _l2_distance(state_a, state_b):
        total = 0.0
        for k in state_a:
            diff = state_a[k].float() - state_b[k].float()
            total += float(torch.norm(diff).item() ** 2)
        return math.sqrt(total)

    def compute_prox_center(self, models):
        """
        Returns r_i_new: a state_dict representing the FODAC estimate of the global average for this node.
        Implementation notes:
          - maintain self.x (consensus state) and self.r_prev (previous reference input) and self.prev_local_model
          - refer to Algorithm1: x_i(t+1) = x_i(t) + sum_j w_ij (x_j(t) - x_i(t)) + Delta r_i(t)
            where r_i(t) is current reference input = local_model_i(t)
            and Delta r_i(t) = r_i(t) - r_i(t-1)
        """
        local_addr = self._addr
        # local model provided as (state_dict, weight)
        if local_addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({local_addr}) missing.")
        local_state, _ = models[local_addr]
        # ensure tensors on desired device
        local_state = {k: v.to(self.device) for k, v in local_state.items()}

        # initialize r_prev and x if None
        if self.r_prev is None or self.x is None:
            # initialize using neighbor average (or local if no neighbors)
            neighbor_avg = self._compute_neighbor_average(models)
            if neighbor_avg is None:
                neighbor_avg = self._clone_state(local_state)
            # set both previous reference and x to neighbor average
            self.r_prev = {k: v.detach().clone().to(self.device) for k, v in neighbor_avg.items()}
            self.x = {k: v.detach().clone().to(self.device) for k, v in neighbor_avg.items()}
            self.prev_local_model = self._clone_state(local_state)
            logging.debug(f"[{self.__class__.__name__}] Initialized FODAC states via neighbor average.")
            return self._clone_state(self.x)

        # compute Delta r_i(t) = r_i(t) - r_i(t-1)
        r_current = {k: local_state[k].detach().clone() for k in local_state}
        delta_r = {k: (r_current[k] - self.r_prev[k]).detach().clone() for k in r_current}

        # Build neighbor x_j list and weights
        # Map addresses to indices if needed
        addrs = list(models.keys())
        # build mapping once if W exists but addr2idx empty
        if self.W is not None and not self.addr2idx:
            # assume addrs are stable ordering across nodes; we sort to have reproducible mapping
            sorted_addrs = sorted(addrs, key=lambda x: str(x))
            self.addr2idx = {addr: i for i, addr in enumerate(sorted_addrs)}
            self.idx2addr = {i: addr for addr, i in self.addr2idx.items()}
            # ensure W compatible
            if self.W.shape[0] != len(sorted_addrs):
                logging.warning(f"[{self.__class__.__name__}] W shape mismatch ({self.W.shape}) vs {len(sorted_addrs)} addresses. Ignoring W.")
                self.W = None

        # prepare neighbor x_j states and weights
        neighbor_x_states = []
        weights = []
        for addr in addrs:
            if addr == local_addr:
                continue
            # get x_j from stored prev_prox_center if available, else fallback to neighbor's local model
            # Note: we only maintain self.x for current node; we need neighbor x_j - approximate by prev_prox_center if exists in models
            # Simpler pragmatic approach: use neighbor's last saved state (models[addr][0]) as proxy for x_j
            state_j, _ = models[addr]
            state_j = {k: v.to(self.device) for k, v in state_j.items()}
            neighbor_x_states.append(state_j)
            if self.W is not None:
                idx_i = self.addr2idx.get(local_addr)
                idx_j = self.addr2idx.get(addr)
                w_ij = float(self.W[idx_i, idx_j])
            else:
                w_ij = 1.0 / max(1, len(neighbor_x_states))  # will be recomputed correctly after list built
            weights.append(w_ij)

        # if W is None, recompute uniform weights
        if self.W is None and neighbor_x_states:
            weights = [1.0 / len(neighbor_x_states)] * len(neighbor_x_states)

        # consensus_part = sum_j w_ij * x_j(t) - x_i(t) * sum_j w_ij  (but using formulation x <- x + sum_j w_ij (x_j - x_i))
        # implement x_new = x + sum_j w_ij*(x_j - x)
        with torch.no_grad():
            # compute sum_j w_ij*(x_j - x)
            delta_consensus = self._zeros_like_state(self.x)
            for w_ij, x_j in zip(weights, neighbor_x_states):
                for k in delta_consensus:
                    delta_consensus[k] += (w_ij * (x_j[k].to(self.device) - self.x[k]))

            # update x: x <- x + delta_consensus + delta_r
            x_new = {}
            for k in self.x:
                x_new[k] = (self.x[k] + delta_consensus[k] + delta_r[k]).detach().clone()

            # update internal states
            self.r_prev = {k: v.detach().clone() for k, v in r_current.items()}
            self.x = {k: v.detach().clone() for k, v in x_new.items()}
            self.prev_local_model = self._clone_state(local_state)

        return self._clone_state(self.x)

    def _compute_neighbor_average(self, models):
        local_addr = self._addr
        neighbor_states = []
        for addr, (state, _) in models.items():
            if addr == local_addr:
                continue
            neighbor_states.append({k: v.to(self.device) for k, v in state.items()})

        if not neighbor_states:
            # fallback to local
            local_state, _ = models[local_addr]
            return {k: v.detach().clone().to(self.device) for k, v in local_state.items()}

        # average elementwise
        with torch.no_grad():
            avg = self._zeros_like_state(neighbor_states[0])
            for st in neighbor_states:
                for k in avg:
                    avg[k] += st[k]
            inv = 1.0 / len(neighbor_states)
            avg = {k: (v * inv).detach().clone() for k, v in avg.items()}
        return avg

    def remove_malicious_models(self, models, prox_center):
        """
        Robust filtering:
          - compute distances to prox_center for each neighbor,
          - use median + MAD to set adaptive threshold (more stable in non-iid).
          - Also apply decay factor as secondary multiplier.
        """
        local_addr = self._addr
        distances = {}
        for addr, (state, _) in models.items():
            if addr == local_addr:
                continue
            dist = self._l2_distance({k: v.to(self.device) for k, v in state.items()},
                                     {k: v.to(self.device) for k, v in prox_center.items()})
            distances[addr] = dist

        if not distances:
            return {}

        values = np.array(list(distances.values()))
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med))) + 1e-8
        # threshold scaling factors: be more permissive early, stricter later
        try:
            current_round = getattr(self.engine, "round", 0) + 1
            total_rounds = max(1, getattr(self.engine, "total_rounds", 100))
        except Exception:
            current_round = 1
            total_rounds = 100
        decay = math.exp(-self.K * current_round / total_rounds)
        # multiplier: median + beta * MAD
        beta = max(2.0, 2.5 * (1.0 - (current_round / total_rounds)))
        threshold = med + beta * mad
        threshold *= (self.A * decay)

        filtered = {}
        for addr, (state, weight) in models.items():
            if addr == local_addr:
                continue
            d = distances[addr]
            if d <= threshold:
                filtered[addr] = (state, weight)
                logging.debug(f"[{self.__class__.__name__}] ACCEPT {addr} d={d:.4f} <= thr={threshold:.4f}")
            else:
                logging.debug(f"[{self.__class__.__name__}] REJECT {addr} d={d:.4f} > thr={threshold:.4f}")
        return filtered

    def run_aggregation(self, models):
        super().run_aggregation(models)

        local_state, _ = self.get_local_model(models)
        # compute prox center via FODAC
        prox_center = self.compute_prox_center(models)

        # filter out malicious/outliers
        filtered = self.remove_malicious_models(models, prox_center)

        if not filtered:
            logging.debug(f"[{self.__class__.__name__}] No neighbor models after filtering; returning local state.")
            return self._clone_state(local_state)

        # compute average of filtered neighbors
        neighbor_list = list(filtered.values())
        S = len(neighbor_list)
        accum = self._zeros_like_state(local_state)
        with torch.no_grad():
            for params, _ in neighbor_list:
                for k in accum:
                    accum[k] += params[k].to(self.device)
            for k in accum:
                accum[k].mul_(1.0 / S)

            # FedProx term: proximal regularizer encourages local towards prox_center
            # In FedProx the prox penalty is mu/2 ||w - r||^2, which adds +mu*(w - r) to gradient.
            # Here we approximate a "one-step" proximal correction: subtract mu*(local - prox_center)
            prox_term = {k: (self.mu * (local_state[k].to(self.device) - prox_center[k].to(self.device))).detach().clone() for k in accum}

            # final aggregated model: a*local + (1-a)*neighbor_avg - prox_term
            result = {}
            for k in accum:
                mixed = self.alpha_mix * local_state[k].to(self.device) + (1.0 - self.alpha_mix) * accum[k]
                aggregated = mixed - prox_term[k]
                # optional clipping to avoid jumps
                if self.clip_norm is not None:
                    norm = float(torch.norm(aggregated).item())
                    if norm > self.clip_norm:
                        aggregated = aggregated * (self.clip_norm / norm)
                result[k] = aggregated.detach().clone()

        # cleanup
        del models, filtered, neighbor_list, accum
        gc.collect()
        logging.info(f"[{self.__class__.__name__}] Aggregation finished. neighbors_used={S}")
        return result
