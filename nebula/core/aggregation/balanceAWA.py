import gc
import torch
import math
import logging
import numpy as np
from nebula.core.aggregation.aggregator import Aggregator


# --- Helper: project onto simplex ---
def project_simplex(v: np.ndarray):
    """Project v onto probability simplex {x: x>=0, sum x = 1}."""
    if np.allclose(v, np.zeros_like(v)):
        return np.ones_like(v) / v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    w = w / (w.sum() + 1e-16)
    return w


class BalanceAWA(Aggregator):
    """
    Hybrid aggregator combining Balance (malicious filtering) and FedAWA (adaptive weighting).

    Algorithm:
    1. Phase 1 (Balance): Filter out malicious models using distance threshold
    2. Phase 2 (FedAWA): Compute adaptive weights for remaining models
    3. Aggregate using adaptive weights and combine with local model
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # Balance parameters
        self.A = 2.0          # Threshold scale factor
        self.K = 1.0          # Decay rate
        self.balance_alpha = 0.4  # Local model weight in final aggregation

        # FedAWA parameters
        self.fedawa_alpha = 0.5    # Regularization weight in loss function
        self.lr = 0.01             # Learning rate for gradient descent
        self.max_steps = 30        # Max iterations for optimization
        self.convergence_threshold = 1e-6

        # State
        self.adaptive_weights = None

        logging.info(f"[{self.__class__.__name__}] Initialized Hybrid Balance-FedAWA")
        logging.info(f"[{self.__class__.__name__}] Balance params: A={self.A}, K={self.K}, alpha={self.balance_alpha}")
        logging.info(f"[{self.__class__.__name__}] FedAWA params: alpha={self.fedawa_alpha}, lr={self.lr}, steps={self.max_steps}")


    def get_local_model(self, models):
        """Extract local model from models dict."""
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found")
        model_params, _ = models[self._addr]
        return model_params


    def compute_model_norm(self, model_params):
        """Compute L2 norm of model parameters."""
        norm = 0.0
        for param in model_params.values():
            tensor = param if param.is_floating_point() else param.float()
            norm += torch.norm(tensor, p=2).item() ** 2
        return math.sqrt(norm)


    def compute_distance(self, model1, model2):
        """Compute Euclidean distance between two models."""
        distance = 0.0
        for layer in model1:
            diff = model1[layer] - model2[layer]
            # Ensure diff is float
            if not diff.is_floating_point():
                diff = diff.float()
            distance += torch.norm(diff, p=2).item() ** 2
        return math.sqrt(distance)


    def phase1_balance_filtering(self, models, local_model):
        """
        Phase 1: Filter malicious models using Balance algorithm.

        Returns:
            dict: Filtered models satisfying distance condition
        """
        logging.info(f"[{self.__class__.__name__}] ===== PHASE 1: BALANCE FILTERING =====")

        # Get round information
        try:
            current_round = self.engine.round + 1
            logging.debug(f"[{self.__class__.__name__}] Current round: {current_round}")
            total_rounds = self.engine.total_rounds
            logging.debug(f"[{self.__class__.__name__}] Total round: {total_rounds}")
        except AttributeError as e:
            logging.error(f"[{self.__class__.__name__}] Failed to get round info: {e}")
            return {}

        # Compute local model norm
        local_norm = self.compute_model_norm(local_model)
        logging.info(f"[{self.__class__.__name__}] Local model norm: {local_norm:.4f}")

        # Compute threshold
        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * local_norm
        logging.info(f"[{self.__class__.__name__}] Threshold (round {current_round}/{total_rounds}): {threshold:.4f}")

        # Filter models
        filtered_models = {}
        rejected_count = 0

        for node_addr, (model_params, weight) in models.items():
            if node_addr == self._addr:
                continue  # Skip local model

            # Compute distance
            distance = self.compute_distance(local_model, model_params)

            # Check condition
            if distance <= threshold:
                filtered_models[node_addr] = (model_params, weight)
                logging.debug(f"[{self.__class__.__name__}] Accpet Model {node_addr}: distance={distance:.4f} <= {threshold:.4f}")
            else:
                rejected_count += 1
                logging.debug(f"[{self.__class__.__name__}] Reject Model {node_addr}: distance={distance:.4f} > {threshold:.4f}")

        logging.info(f"[{self.__class__.__name__}] Filtering result: {len(filtered_models)} accepted, {rejected_count} rejected")
        return filtered_models


    def phase2_fedawa_optimization(self, filtered_models, local_model):
        """
        Phase 2: Compute adaptive weights using FedAWA optimization.

        Returns:
            dict: Adaptive weights for each node
        """
        logging.info(f"[{self.__class__.__name__}] ===== PHASE 2: FEDAWA OPTIMIZATION =====")

        if not filtered_models:
            logging.warning(f"[{self.__class__.__name__}] No models to optimize, returning empty weights")
            return {}

        # Build matrices T and Theta
        node_addrs = []
        taus = []      # Client vectors
        thetas = []    # Client models

        with torch.no_grad():
            for node_addr, (model_params, _) in filtered_models.items():
                # Compute tau = theta_k - theta_g
                flat_tau = []
                flat_theta = []

                for layer in local_model:
                    tau_layer = (model_params[layer] - local_model[layer]).to(torch.float32)
                    flat_tau.append(tau_layer.flatten().cpu().numpy())
                    flat_theta.append(model_params[layer].flatten().cpu().numpy())

                # Concatenate to 1D arrays
                tau_flat = np.concatenate(flat_tau, axis=0)
                theta_flat = np.concatenate(flat_theta, axis=0)

                node_addrs.append(node_addr)
                taus.append(tau_flat)
                thetas.append(theta_flat)

        # Stack into matrices
        T = np.stack(taus, axis=1)       # Shape: (D, K)
        Theta = np.stack(thetas, axis=1) # Shape: (P, K)

        # Normalize
        T = T / (np.linalg.norm(T, axis=0, keepdims=True) + 1e-8)
        Theta = Theta / (np.linalg.norm(Theta, axis=0, keepdims=True) + 1e-8)

        D, K = T.shape
        P = Theta.shape[0]

        logging.info(f"[{self.__class__.__name__}] Matrix T shape: {T.shape}, Theta shape: {Theta.shape}")
        logging.info(f"[{self.__class__.__name__}] Optimizing weights for {K} clients")

        # Flatten local model
        flat_theta_g = []
        for layer in local_model:
            flat_theta_g.append(local_model[layer].flatten().cpu().numpy())
        theta_g_vec = np.concatenate(flat_theta_g, axis=0)
        # theta_g_vec = theta_g_vec / (np.linalg.norm(theta_g_vec) + 1e-8)

        # # Initialize lambda
        # if self.adaptive_weights and len(self.adaptive_weights) == K:
        #     lam = np.array([self.adaptive_weights.get(n, 1.0/K) for n in node_addrs], dtype=np.float64)
        #     lam = lam / (lam.sum() + 1e-16)
        #     logging.info(f"[{self.__class__.__name__}] Using previous adaptive weights as initialization")
        # else:
        #     lam = np.ones(K) / K
        #     logging.info(f"[{self.__class__.__name__}] Using uniform initialization")
        lam = np.ones(K) / K
        logging.info(f"[{self.__class__.__name__}] Using uniform initialization")

        # Gradient descent optimization
        logging.info(f"[{self.__class__.__name__}] Starting gradient descent...")

        for it in range(self.max_steps):
            # Compute aggregated vectors
            tau_g = T.dot(lam)           # Shape: (D,)
            theta_agg = Theta.dot(lam)   # Shape: (P,)

            # Gradient for first term: sum_k(lam_k) * ||tau_k - tau_g||^2
            R = T - tau_g.reshape(-1, 1)   # Shape: (D, K)
            r_norm_sq = np.sum(R * R, axis=0)  # Shape: (K,)
            Rlam = R.dot(lam)  # Shape: (D,)
            cross = R.T.dot(Rlam)  # Shape: (K,)
            grad1 = (r_norm_sq - 2.0 * cross) / K

            # Gradient for second term: alpha * ||Theta.lam - theta_g||^2
            diff_theta = theta_agg - theta_g_vec  # Shape: (P,)
            grad2 = 2.0 * self.fedawa_alpha * (Theta.T.dot(diff_theta)) / K

            # Total gradient
            grad = grad1 + grad2

            # Gradient descent step
            lam = lam - self.lr * grad
            lam = project_simplex(lam)

            # Check convergence
            grad_norm = np.linalg.norm(self.lr * grad)
            if it % 10 == 0:
                logging.debug(f"[{self.__class__.__name__}] Step {it}: grad_norm={grad_norm:.6f}, lambda={lam}")

            if grad_norm < self.convergence_threshold:
                logging.info(f"[{self.__class__.__name__}] Converged at step {it}")
                break

        # Build weights dictionary
        weights = {node_addrs[i]: float(lam[i]) for i in range(K)}
        logging.info(f"[{self.__class__.__name__}] Final adaptive weights: {weights}")

        return weights


    def run_aggregation(self, models):
        """
        Main aggregation function combining Balance and FedAWA.

        Algorithm:
        1. Phase 1: Filter malicious models using Balance
        2. Phase 2: Compute adaptive weights using FedAWA
        3. Aggregate models using adaptive weights
        4. Combine with local model
        """
        super().run_aggregation(models)

        logging.info(f"[{self.__class__.__name__}] ========== HYBRID BALANCE-FEDAWA AGGREGATION ==========")
        logging.info(f"[{self.__class__.__name__}] Total models received: {len(models)}")

        # Get local model
        local_model = self.get_local_model(models)

        # Phase 1: Balance filtering
        filtered_models = self.phase1_balance_filtering(models, local_model)

        if not filtered_models:
            logging.warning(f"[{self.__class__.__name__}] No models passed filtering, returning local model")
            return local_model

        # Phase 2: FedAWA optimization
        adaptive_weights = self.phase2_fedawa_optimization(filtered_models, local_model)

        if not adaptive_weights:
            logging.warning(f"[{self.__class__.__name__}] No adaptive weights computed, returning local model")
            return local_model

        # Save adaptive weights for next round
        self.adaptive_weights = adaptive_weights

        # Phase 3: Aggregate using adaptive weights
        logging.info(f"[{self.__class__.__name__}] ===== PHASE 3: WEIGHTED AGGREGATION =====")

        accum = {layer: torch.zeros_like(param, dtype=torch.float32)
                 for layer, param in local_model.items()}

        with torch.no_grad():
            for node_addr, (model_params, _) in filtered_models.items():
                weight = adaptive_weights.get(node_addr, 0.0)
                if weight <= 0.0:
                    continue

                for layer in accum:
                    accum[layer].add_(
                        model_params[layer].to(accum[layer].dtype),
                        alpha=weight
                    )

        # Phase 4: Combine with local model
        logging.info(f"[{self.__class__.__name__}] ===== PHASE 4: LOCAL MODEL COMBINATION =====")
        logging.info(f"[{self.__class__.__name__}] Local weight: {self.balance_alpha}, Aggregated weight: {1-self.balance_alpha}")

        for layer in accum:
            accum[layer].mul_(1 - self.balance_alpha)
            accum[layer].add_(local_model[layer], alpha=self.balance_alpha)

        # Cleanup
        del models, filtered_models
        gc.collect()

        logging.info(f"[{self.__class__.__name__}] ========== AGGREGATION COMPLETED ==========")
        return accum
