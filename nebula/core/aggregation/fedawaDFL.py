import numpy as np
import torch
import math
import gc
import logging
from nebula.core.aggregation.aggregator import Aggregator

# --- helper: project onto simplex ---
def project_simplex(v: np.ndarray):
    """Project v onto probability simplex {x: x>=0, sum x = 1}.
    Gradient descent có thể kéo λ ra ngoài miền hợp lệ
    Ta chiếu nó lại lên simplex để đảm bảo các trọng số vẫn là phân phối xác suất (tổng = 1, không âm).
    """
    # from Duchi et al. (2008)
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
    # numerical fix
    w = w / (w.sum() + 1e-16)
    return w


class FedAWADFL(Aggregator):
    """
    Aggregator: Federated Averaging (FedAvg)
    Authors: McMahan et al.
    Year: 2016
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        logging.debug(f"[{self.__class__.__name__}] Start FedAWA-CFL")
        self._local_model = None
        self.adaptive_weights = None

    # def get_local_model(self):
    #     if self._local_model is None:
    #         raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
    #     return self._local_model

    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        # Unpack tuple to get model_params only
        model_params, _ = models[self._addr]
        return model_params

    def calculate_client_vector_and_global_vector(self, models):
        """
        Compute client vectors tau_k = theta_k - theta_g using sample-size normalized weights
        to compute an initial merged vector if needed.
        Return client_vector (dict of state_dicts) and also arrays/matrices for optimization.
        # Giả sử có 3 clients, model có 2 layers
        # Layer 1: 2 params, Layer 2: 3 params → D = P = 5

        # Global model
        θ_g = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Client 1
        θ_1 = [1.1, 2.1, 3.2, 4.1, 5.0]
        τ_1 = θ_1 - θ_g = [0.1, 0.1, 0.2, 0.1, 0.0]

        # Client 2
        θ_2 = [0.9, 1.9, 2.8, 3.9, 5.1]
        τ_2 = θ_2 - θ_g = [-0.1, -0.1, -0.2, -0.1, 0.1]

        # Client 3 (outlier)
        θ_3 = [5.0, 6.0, 7.0, 8.0, 9.0]
        τ_3 = θ_3 - θ_g = [4.0, 4.0, 4.0, 4.0, 4.0]

        # Matrix T (D=5, K=3)
        T = [[0.1,  -0.1,  4.0],
            [0.1,  -0.1,  4.0],
            [0.2,  -0.2,  4.0],
            [0.1,  -0.1,  4.0],
            [0.0,   0.1,  4.0]]

        # Matrix Theta (P=5, K=3)
        Theta = [[1.1,  0.9,  5.0],
                [2.1,  1.9,  6.0],
                [3.2,  2.8,  7.0],
                [4.1,  3.9,  8.0],
                [5.0,  5.1,  9.0]]
        """
        logging.debug(f"[{self.__class__.__name__}] Starting client vector and global vector calculation")
        logging.debug(f"[{self.__class__.__name__}] Number of models received: {len(models)}")

        # client_vector = {}
        local_model = self.get_local_model(models)

        # Build lists to create matrices
        node_addrs = []
        taus = []      # list of flattened tau vectors
        thetas = []    # list of flattened theta_k vectors

        with torch.no_grad():
            for node_addr, (model_params, weight) in models.items():
                if node_addr == self._addr:
                    logging.debug(f"[{self.__class__.__name__}] Skipping server/local entry: {node_addr}")
                    continue  # server/local entry skip
                # compute tau = theta_k - theta_g
                # τ_k = θ_k - θ_g
                vector = {}
                flat_tau = []
                flat_theta = []
                for layer in local_model:
                    # ensure same dtype
                    tau_layer = (model_params[layer] - local_model[layer]).to(torch.float32)
                    vector[layer] = tau_layer
                    # Flatten each layer from tensor to 1D
                    flat_tau.append(tau_layer.flatten().cpu().numpy())
                    flat_theta.append(model_params[layer].flatten().cpu().numpy())
                # concatenate to 1D arrays
                tau_flat = np.concatenate(flat_tau, axis=0)
                theta_flat = np.concatenate(flat_theta, axis=0)
                # client_vector[node_addr] = vector
                node_addrs.append(node_addr)
                taus.append(tau_flat)
                thetas.append(theta_flat)

        if len(taus) == 0:
            logging.debug(f"[{self.__class__.__name__}] No client models to aggregate!")
            return {}, None, None, None  # no clients

        # Stack into matrices: shape (dim, K) but we'll use (K, dim) for convenience
        T = np.stack(taus, axis=1)    # shape (D, K) -> τ_k
        Theta = np.stack(thetas, axis=1)  # shape (P, K) -> θ_k

        logging.debug(f"[{self.__class__.__name__}] Client vectors computed successfully")
        logging.debug(f"[{self.__class__.__name__}] Matrix T (tau) shape: {T.shape} (D={T.shape[0]}, K={T.shape[1]})")
        logging.debug(f"[{self.__class__.__name__}] Matrix Theta shape: {Theta.shape} (P={Theta.shape[0]}, K={Theta.shape[1]})")
        logging.debug(f"[{self.__class__.__name__}] Node addresses: {node_addrs}")

        return T, Theta, node_addrs


    def calculate_adaptive_weight(self, T, Theta, node_addrs, theta_g_vec, alpha=1.0, lr=0.2, steps=100, init_lambda=None):
        """
        Solve for lambda using projected gradient descent on simplex.
        T: np.ndarray shape (D, K) where columns are tau_k
        Theta: np.ndarray shape (P, K) where columns are theta_k
        theta_g_vec: np.ndarray shape (P,) current global model flattened
        Return dict mapping node_addr -> lambda_k

        T có shape (D, K) — mỗi cột là τk (client vector flattened).
        Theta có shape (P, K) — mỗi cột là θk flattened.
        theta_g_vec θg là vector flattened của global model hiện tại (shape P).
        """

        logging.debug(f"[{self.__class__.__name__}] Start calculate_adaptive_weight with gradient descent")
        # chuan hoa cac vector dau vao
        T = T / (np.linalg.norm(T, axis=0, keepdims=True) + 1e-8)
        Theta = Theta / (np.linalg.norm(Theta, axis=0, keepdims=True) + 1e-8)

        D, K = T.shape
        P = Theta.shape[0]

        # init lambda: use previous adaptive weights if available, else sample-size uniform
        if init_lambda is None:
            lam = np.ones(K) / K
        else:
            lam = np.array(init_lambda, dtype=np.float64)
            # lam = lam / (lam.sum() + 1e-16)

        # Precompute for speed
        # We'll implement objective:
        # L(lam) = sum_k(lam_k) ||tau_k - T.lam||^2 + alpha * ||Theta.lam - theta_g||^2
        for it in range(steps):
            tau_g = T.dot(lam)           # = T.lam = tau_g shape (D,)
            theta_agg = Theta.dot(lam)   # = Theta.lam shape (P,)

            # grad for first term:
            # first_term = sum_k(lam_k) * ||tau_k - tau_g||^2
            # compute vector r_k = tau_k - tau_g (shape D x K)
            R = T - tau_g.reshape(-1, 1)   # D x K
            # compute grad1_j = ||r_j||^2 - 2 * sum_k lam_k (r_k^T r_j)
            r_norm_sq = np.sum(R * R, axis=0)  # shape (K,)
            # compute cross = R.T @ R @ lam = (K,)  (but R.T @ R is KxK expensive; compute R.T @ (R@lam))
            Rlam = R.dot(lam)  # D,
            cross = R.T.dot(Rlam)  # K,
            grad1 = (r_norm_sq - 2.0 * cross) / K

            # grad for second term: alpha * 2 * Theta.T (Theta lam - theta_g)
            diff_theta = theta_agg - theta_g_vec  # P,
            grad2 = 2.0 * alpha * (Theta.T.dot(diff_theta)) / K

            grad = grad1 + grad2

            # gradient descent step (note: minimize)
            lam = lam - lr * grad
            lam = project_simplex(lam)
            if it > 20 and np.linalg.norm(lr * grad) < 1e-6:
                logging.debug(f"[{self.__class__.__name__}] Stop gradient descent at round {self.engine.round} at step {it}")
                break
            if it % 10 == 0:
                logging.debug(f"[{self.__class__.__name__}] Gradient descent at step {it} : {lam}")

            # optional small stopping criterion (can add)
        # build dict mapping
        weights = {node_addrs[i]: float(lam[i]) for i in range(K)}
        logging.debug(f"[{self.__class__.__name__}] Finish calculate_adaptive weight at round {self.engine.round}: {weights}")

        return weights


    def run_aggregation(self, models):
        super().run_aggregation(models)
        logging.info(f"[{self.__class__.__name__}] ========== Starting FedAWA-CFL Aggregation Round ==========")
        logging.info(f"[{self.__class__.__name__}] Number of models to aggregate: {len(models)}")

        # Init stage at round 0, compute global model with FedAvg
        # if self.engine.round == 0 or self._local_model is None:
        #     logging.debug(f"[{self.__class__.__name__}] Init stage at round 0 when local model is None")
        #     models = list(models.values())
        #     total_samples = float(sum(weight for _, weight in models))
        #     if total_samples == 0:
        #         raise ValueError("Total number of samples must be greater than zero.")

        #     last_model_params = models[-1][0]
        #     accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in last_model_params.items()}

        #     with torch.no_grad():
        #         for model_parameters, weight in models:
        #             normalized_weight = weight / total_samples
        #             for layer in accum:
        #                 accum[layer].add_(
        #                     model_parameters[layer].to(accum[layer].dtype),
        #                     alpha=normalized_weight,
        #                 )

        #     del models
        #     gc.collect()
        #     self._local_model = accum
        #     return accum

        # init local model
        local_model = self._local_model = self.get_local_model(models)

        # Step A: compute client vectors and matrices
        logging.info(f"[{self.__class__.__name__}] Step A: Computing client vectors and matrices")
        T, Theta, node_addrs = self.calculate_client_vector_and_global_vector(models)
        if T is None:
            logging.debug(f"[{self.__class__.__name__}] No client vectors computed, returning empty aggregation")
            return {}  # nothing to aggregate

        # flatten current global model theta_g into vector
        logging.info(f"[{self.__class__.__name__}] Step B: Flattening global model theta_g")

        flat_theta_g = []
        for layer in local_model:
            flat_theta_g.append(local_model[layer].flatten().cpu().numpy())
        theta_g_vec = np.concatenate(flat_theta_g, axis=0)

        # Step B: initial lambda: use previous self.adaptive_weights if available aligned with node_addrs
        logging.info(f"[{self.__class__.__name__}] Step C: Preparing initial lambda for optimization")
        init_lambda = None
        # if self.adaptive_weights and self.engine.round <= 2:
        #     init_lambda = np.array(
        #         [self.adaptive_weights[n] for n in node_addrs if n in self.adaptive_weights],
        #         dtype=np.float64
        #     )

        # Step C: solve for adaptive weights via PGD
        logging.info(f"[{self.__class__.__name__}] Step D: Solving for adaptive weights via PGD")
        adaptive_weights_dict = self.calculate_adaptive_weight(
            T, Theta, node_addrs, theta_g_vec,
            alpha=2, lr=0.01, steps=30, init_lambda=init_lambda
        )
        # save for next round
        self.adaptive_weights = adaptive_weights_dict

        # Step D: aggregate model parameters using adaptive_weights
        logging.info(f"[{self.__class__.__name__}] Step E: Aggregating model parameters using adaptive weights")
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}
        with torch.no_grad():
            for node_addr, (model_parameters, _) in models.items():
                if node_addr == self._addr:
                    continue
                w = adaptive_weights_dict.get(node_addr, 0.0)
                if w <= 0.0:
                    continue
                for layer in accum:
                    accum[layer].add_(model_parameters[layer].to(accum[layer].dtype), alpha=w)


        local_model = self.get_local_model(models)
        alpha = 0.2
        for layer in accum:
            accum[layer].mul_(1-alpha)
            accum[layer].add_(local_model[layer], alpha=(alpha))

        logging.info(f"[{self.__class__.__name__}] ========== FedAWA-CFL Aggregation Round Completed ==========")
        gc.collect()
        return accum
