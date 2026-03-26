import torch
import numpy as np
import logging
from collections import deque
from torch.nn.functional import cosine_similarity

class SelfAdaptiveD2BNode:
    """
    Self-Adaptive Trust-Aware Decentralized Dynamic-Bounded DPFL (D2B-DPFL) node.
    Maintains continuous state for evaluating inbound models, dynamically clipping
    local gradients, and dispersing budget-aware noise.
    """
    def __init__(
        self,
        rho_min: float,
        rho_max: float,
        T_target: int,
        k_min: int,
        alpha: float,
        gamma_base: float,
        mu: float,
        w1: float,
        w2: float,
        lambda_max: float,
        tau: float,
        eta: float,
        epsilon_r: float = 1e-6
    ):
        self.rho_min = rho_min
        self.rho_max = rho_max
        # Calculate beta = ((rho_max / rho_min) - 1) / T_target
        self.beta = ((rho_max / rho_min) - 1) / T_target if T_target > 0 else 0
        self.k_min = k_min
        self.alpha = alpha
        self.gamma_base = gamma_base
        self.mu = mu
        self.w1 = w1
        self.w2 = w2
        self.lambda_max = lambda_max
        self.tau = tau
        self.eta = eta
        self.epsilon_r = epsilon_r

        self.W_prev = None  # Local model weights from previous round (1D tensor)
        self.W_prev2 = None # Local model weights from 2 rounds ago, to calc S_i,prev
        self.Neighbor_Models_Prev = {}  # Last received model weights per neighbor
        self.Trust = {}  # Trust score per neighbor (id -> float)
        self.R_prev = {}  # Round-score from previous round per neighbor
        self.H = deque()  # Dynamic-sized deque of L2-norms of local updates
        self.rho_base = rho_min
        self.L_prev = None

        self.Clip_t = None

    def _flatten_model(self, model_dict: dict) -> torch.Tensor:
        """Helper to flatten state dict to a single 1D tensor."""
        tensors = []
        for v in model_dict.values():
            tensors.append(v.view(-1).float())
        if not tensors:
            return torch.tensor([])
        return torch.cat(tensors)

    def _unflatten_model(self, flat_tensor: torch.Tensor, model_dict: dict) -> dict:
        """Helper to restore flattened tensor to state dict shape."""
        res_dict = {}
        offset = 0
        for k, v in model_dict.items():
            numel = v.numel()
            res_dict[k] = flat_tensor[offset:offset+numel].view(v.shape).to(v.dtype)
            offset += numel
        return res_dict

    def inbound_evaluate(self, received_models: dict) -> dict:
        """
        Module 1: Inbound Evaluation & Dynamic Trust
        Args:
            received_models: Dict[neighbor_id, state_dict]
        Returns:
            Aggregated model state_dict.
        """
        if not received_models:
            logging.warning("No models received for inbound evaluation.")
            return {}

        flat_models = {j: self._flatten_model(m) for j, m in received_models.items()}
        reference_dict = list(received_models.values())[0] # To unflatten later

        # 1. Extract Pseudo-Gradients S_j
        S_dict = {}
        Z_dict = {}
        for j, W_j in flat_models.items():
            if j not in self.Neighbor_Models_Prev:
                # If first time, diff against zeros or current?
                # Formula: S_j = W_j - Neighbor_Models_Prev[j].
                # If we don't have it, assume W_prev or zeros.
                S_dict[j] = W_j - (self.W_prev if self.W_prev is not None else torch.zeros_like(W_j))
            else:
                S_dict[j] = W_j - self.Neighbor_Models_Prev[j]

            # 2. Z-Score Calculation
            Z_dict[j] = torch.mean(torch.abs(S_dict[j])).item()

            # Update history
            self.Neighbor_Models_Prev[j] = W_j.clone()

        # 3. Dynamic Thresholding
        Z_values = list(Z_dict.values())
        Z_tensor = torch.tensor(Z_values)
        sigma_Z = torch.std(Z_tensor, unbiased=False).item() if len(Z_values) > 1 else 0.0

        gamma_t = self.gamma_base * np.exp(-self.mu * sigma_Z)

        Z_median = torch.median(Z_tensor).item()
        Z_mad = torch.median(torch.abs(Z_tensor - Z_median)).item()

        Z_thresh = Z_median + gamma_t * Z_mad

        # Local model difference for P_sim
        if self.W_prev is not None and self.W_prev2 is not None:
            S_i_prev = self.W_prev - self.W_prev2
        else:
            S_i_prev = None

        # 4. Trust Update
        total_trust = 0.0
        updated_trusts = {}
        for j, S_j in S_dict.items():
            Z_j = Z_dict[j]

            # P_safe
            if Z_thresh <= 0:
                P_safe = 1.0 if Z_j == 0 else 0.0
            else:
                P_safe = max(0.0, 1.0 - Z_j / Z_thresh)

            # P_sim
            if S_i_prev is not None:
                # Add epsilon to prevent NaN in cosine sim
                c_sim = cosine_similarity(S_i_prev.unsqueeze(0), S_j.unsqueeze(0), eps=1e-8).item()
                if np.isnan(c_sim):
                    c_sim = 0.0
                P_sim = (c_sim + 1) / 2
            else:
                P_sim = 1.0 # Default if no local history

            R_ij = self.w1 * P_safe + self.w2 * P_sim

            R_prev_j = self.R_prev.get(j, 1.0)

            # λ_ij
            diff = abs(R_ij - R_prev_j)
            lambda_ij = self.lambda_max * max(0.0, 1.0 - (diff / 0.5)**2)

            prev_trust = self.Trust.get(j, 1.0)
            new_trust = lambda_ij * prev_trust + (1 - lambda_ij) * R_ij

            self.Trust[j] = new_trust
            self.R_prev[j] = R_ij

            updated_trusts[j] = new_trust
            total_trust += new_trust

        # 5. Aggregation
        W_agg = torch.zeros_like(list(flat_models.values())[0])
        if total_trust > 0:
            for j, W_j in flat_models.items():
                norm_trust = updated_trusts[j] / total_trust
                W_agg += W_j * norm_trust
        else:
            # Fallback if trust is 0
            for j, W_j in flat_models.items():
                W_agg += W_j * (1.0 / len(flat_models))

        # Unflatten
        return self._unflatten_model(W_agg, reference_dict)


    def local_process(self, W_train_dict: dict, W_agg_dict: dict, current_loss: float) -> tuple[dict, float]:
        """
        Module 2: Local Processing & Adaptive Clipping
        Args:
            W_train_dict: Local trained model
            W_agg_dict: Aggregated model from inbound_evaluate
            current_loss: Local training loss L^(t)
        Returns:
            Tuple (W_safe_dict, L^(t))
        """
        W_train = self._flatten_model(W_train_dict)
        W_agg = self._flatten_model(W_agg_dict)

        # Calculate Deviation
        delta_W = W_train - W_agg
        norm_delta_W = torch.norm(delta_W, p=2).item()

        # Dynamic Window
        H_var = np.var(self.H) if len(self.H) > 1 else 0.0
        k_t = max(self.k_min, int(np.floor(self.alpha * np.log(1 + H_var))))

        self.H.append(norm_delta_W)
        while len(self.H) > k_t:
            self.H.popleft()

        # Adaptive Clipping
        self.Clip_t = np.mean(self.H) if self.H else epsilon_r
        # Avoid zero division
        if self.Clip_t <= 0:
            self.Clip_t = self.epsilon_r

        clip_ratio = max(1.0, norm_delta_W / self.Clip_t)
        delta_W_prime = delta_W / clip_ratio

        # Safe Model
        W_safe = W_agg + delta_W_prime

        # Update local history for next round
        self.W_prev2 = self.W_prev.clone() if self.W_prev is not None else None
        self.W_prev = W_safe.clone()

        return self._unflatten_model(W_safe, W_train_dict), current_loss

    def outbound_dispatch(self, W_safe_dict: dict, current_loss: float, neighbor_j: str) -> dict:
        """
        Module 3: Loss-Aware Budget & Bounded Noise for a specific neighbor.
        Args:
            W_safe_dict: The safe clipped model
            current_loss: current training loss
            neighbor_j: The id of the target neighbor
        Returns:
            Noisy model customized for neighbor_j
        """
        W_safe = self._flatten_model(W_safe_dict)
        D_i_size = len(W_safe) # Number of parameters |D_i|

        # Customized Noise Injection for j
        trust_j = self.Trust.get(neighbor_j, 1.0)
        rho_ij = max(self.rho_base * trust_j, self.epsilon_r)

        clip_t_sq = float(self.Clip_t**2) if self.Clip_t else float(self.epsilon_r**2)

        # Variance calculation
        # To avoid overflow, ensure D_i_size^2 is float
        D_i_sq = float(D_i_size) ** 2
        sigma_ij_sq = (2 * clip_t_sq) / (D_i_sq * rho_ij)

        # Convert zCDP to DP
        delta = 1e-5
        epsilon_ij = np.sqrt(4 * rho_ij * np.log(1/delta))

        # Bound
        exp_eps = np.exp(epsilon_ij)
        b = (exp_eps - self.eta) / (exp_eps + self.eta)

        # Generate noise
        noise = torch.normal(mean=0.0, std=np.sqrt(sigma_ij_sq), size=W_safe.shape, device=W_safe.device)

        # Apply bounds
        noise_bound = torch.clamp(noise, min=-b, max=b)

        # Output
        W_out = W_safe + noise_bound

        return self._unflatten_model(W_out, W_safe_dict)

    def post_round_update(self, current_loss: float):
        """Called at the end of the round to update budget and L_prev."""
        if self.L_prev is not None:
            if abs(current_loss - self.L_prev) <= self.tau:
                self.rho_base = min(self.rho_base * (1 + self.beta), self.rho_max)
        self.L_prev = current_loss
