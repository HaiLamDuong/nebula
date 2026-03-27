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
        # Base zero-concentrated DP budget (lower and upper bounds)
        self.rho_min = rho_min
        self.rho_max = rho_max

        # Calculate the budget growth factor `beta`
        # based on reaching rho_max in exactly T_target rounds.
        self.beta = ((rho_max / rho_min) - 1) / T_target if T_target > 0 else 0

        # Hyperparameters for local processing (dynamic window size and clipping)
        self.k_min = k_min         # Minimum size of moving window history H
        self.alpha = alpha         # Scaling factor for the dynamic window calculation

        # Hyperparameters for calculating dynamic bounds in inbound_evaluate
        self.gamma_base = gamma_base # Base multiplier for threshold calculation
        self.mu = mu                 # Decay factor penalizing variations in the network

        # Weighting factors to balance Safety Score (P_safe) vs Similarity Score (P_sim)
        self.w1 = w1
        self.w2 = w2

        # Memory penalty factor representing trust decay and inertia
        self.lambda_max = lambda_max

        # Threshold for loss variation. If local loss hasn't dropped by at least `tau`, budget is increased.
        self.tau = tau

        # Truncation boundary for the dynamically bounded normal noise
        self.eta = eta

        # A tiny floating point value added to prevent division-by-zero errors in mathematically unstable formulas
        self.epsilon_r = epsilon_r

        # --- STATE VARIABLES PERSISTED ACROSS ROUNDS ---

        # 'W_prev' holds the local agent's model weights from the IMMEDIATELY PREVIOUS round t-1.
        self.W_prev = None

        # 'W_prev2' holds the local agent's model weights from two rounds ago (t-2).
        # We need this to calculate S_i_prev = W_prev - W_prev2.
        self.W_prev2 = None

        # Dictionary storing the most recent weights received from EACH neighbor j.
        self.Neighbor_Models_Prev = {}

        # Dictionary storing our trust evaluations [0, 1] for each neighbor j.
        self.Trust = {}

        # Dictionary storing what the Trust evaluation vector (R) was in the previous round.
        # Used to measure abrupt behavioral shifts in neighboring nodes.
        self.R_prev = {}

        # 'H' is a moving history ring-buffer storing the L2-norms of our local model updates across rounds.
        self.H = deque()

        # The dynamic Privacy Budget that grows over rounds when loss plateaus.
        self.rho_base = rho_min

        # Loss from the previous round (used purely to decide if we increase privacy budget).
        self.L_prev = None

        # The dynamically calculated clipping threshold calculated from H in the local processing step.
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
        This function handles calculating trust based on incoming models and aggregating them safely.
        """
        if not received_models:
            logging.warning("No models received for inbound evaluation.")
            return {}

        logging.info(f"[D2B] inbound_evaluate: Received {(len(received_models))} models for evaluation.")

        # Flatten all inbound state_dicts to simple 1D tensors to heavily speed up vector math.
        flat_models = {j: self._flatten_model(m) for j, m in received_models.items()}
        reference_dict = list(received_models.values())[0] # To unflatten later

        # 1. Extract Pseudo-Gradients S_j
        # Pseudo-gradient represents "how much neighbor j's model changed since last round".
        S_dict = {}
        Z_dict = {}
        for j, W_j in flat_models.items():
            if j not in self.Neighbor_Models_Prev:
                # If this is the node's very first time seeing neighbor j, we compute pseudo-gradient
                # against our own W_prev (or completely zero vectors if there is no W_prev).
                S_dict[j] = W_j - (self.W_prev if self.W_prev is not None else torch.zeros_like(W_j))
            else:
                # If we've seen them before, it's just their current weights minus their past weights
                S_dict[j] = W_j - self.Neighbor_Models_Prev[j]

            # 2. Z-Score Calculation
            # The Z-score basically tells us the absolute average rate of update for this specific neighbor.
            Z_dict[j] = torch.mean(torch.abs(S_dict[j])).item()

            # Save the current state of neighbor j so we can do this math again next round.
            self.Neighbor_Models_Prev[j] = W_j.clone()

        # 3. Dynamic Thresholding
        # We need to find the "average" behavior of the entire network to find anomalies.
        Z_values = list(Z_dict.values())
        Z_tensor = torch.tensor(Z_values)

        # Standard deviation of updates tells us how much nodes disagree with each other!
        sigma_Z = torch.std(Z_tensor, unbiased=False).item() if len(Z_values) > 1 else 0.0

        # Create a dynamic envelope/multiplier `gamma`. As nodes diverge (high sigma_Z), we constrain the envelope heavily (gamma drops).
        gamma_t = self.gamma_base * np.exp(-self.mu * sigma_Z)

        # Baseline "normal" network update scale. We use median & Median Absolute Deviation (MAD) for robust statistics instead of mean.
        Z_median = torch.median(Z_tensor).item()
        Z_mad = torch.median(torch.abs(Z_tensor - Z_median)).item()

        # Anything jumping wildly above this Z_thresh is classified as malicious / erratic.
        Z_thresh = Z_median + gamma_t * Z_mad

        # Local model difference for P_sim (comparing their updates to our own historical updates)
        if self.W_prev is not None and self.W_prev2 is not None:
            S_i_prev = self.W_prev - self.W_prev2
        else:
            S_i_prev = None

        # 4. Trust Update
        total_trust = 0.0
        updated_trusts = {}
        rejected_neighbors = set()
        for j, S_j in S_dict.items():
            Z_j = Z_dict[j]

            # REJECTION RULE:
            # If the neighbor made bizarrely huge parameter modifications exceeding the dynamic threshold
            # we completely reject their model for this round.
            if Z_j > Z_thresh:
                logging.warning(f"[D2B] Rejecting neighbor {j}: Z-Score ({Z_j:.6f}) exceeds Z_thresh ({Z_thresh:.6f})")
                rejected_neighbors.add(j)
                self.Trust[j] = 0.0
                self.R_prev[j] = 0.0
                continue

            # P_safe: Evaluates if neighbor j made bizarrely huge parameter modifications.
            if Z_thresh <= 0:
                # Failsafe if the network has zero variance.
                P_safe = 1.0 if Z_j == 0 else 0.0
            else:
                # The closer you get to Z_thresh ceiling, the lower your safety score. Capped at 0.
                P_safe = max(0.0, 1.0 - Z_j / Z_thresh)

            # P_sim: Evaluates the direction neighbor j is moving in the optimization landscape vs us.
            if S_i_prev is not None:
                # Use Cosine Similarity (-1 to 1) comparing their gradient vector vs our gradient vector
                # Shifting it into a [0, 1] probability plane.
                c_sim = cosine_similarity(S_i_prev.unsqueeze(0), S_j.unsqueeze(0), eps=1e-8).item()
                if np.isnan(c_sim):
                    c_sim = 0.0
                P_sim = (c_sim + 1) / 2
            else:
                P_sim = 1.0 # Default fallback if we haven't trained enough locally yet.

            # We blend Size anomaly (P_safe) and Direction anomaly (P_sim) into R_ij
            R_ij = self.w1 * P_safe + self.w2 * P_sim

            R_prev_j = self.R_prev.get(j, 1.0)

            # λ_ij: Dynamic Memory Factor.
            # If neighbor abruptly changes behavior, we apply a quadrative penalty causing us to rapidly forget their past high trust.
            diff = abs(R_ij - R_prev_j)
            lambda_ij = self.lambda_max * max(0.0, 1.0 - (diff / 0.5)**2)

            # EMA calculation of Trust tying together their historical reputation vs. their current action score.
            prev_trust = self.Trust.get(j, 1.0)
            new_trust = lambda_ij * prev_trust + (1 - lambda_ij) * R_ij

            # Lock in history state.
            self.Trust[j] = new_trust
            self.R_prev[j] = R_ij

            updated_trusts[j] = new_trust
            total_trust += new_trust
            logging.debug(f"[D2B] Neighbor {j}: Z-Score={Z_j:.6f}, Trust Update: {prev_trust:.4f} -> {new_trust:.4f}")

        logging.info(f"[D2B] inbound_evaluate: Dynamic Threshold Z_thresh={Z_thresh:.6f}, Total Trust={total_trust:.4f}")

        # 5. Aggregation
        # Sums all models weighted against their verified `Trust`!
        W_agg = torch.zeros_like(list(flat_models.values())[0])
        accepted_models = {j: W_j for j, W_j in flat_models.items() if j not in rejected_neighbors}

        if total_trust > 0:
            for j, W_j in accepted_models.items():
                norm_trust = updated_trusts[j] / total_trust
                W_agg += W_j * norm_trust
        else:
            # Absolute fallback if all trust somehow completely bombs to absolute zero.
            if len(accepted_models) > 0:
                for j, W_j in accepted_models.items():
                    W_agg += W_j * (1.0 / len(accepted_models))
            else:
                # If ALL models are rejected, we retain our own previous model (if it exists) to prevent corruption.
                logging.error("[D2B] All neighbor models rejected! Reverting to local W_prev.")
                if self.W_prev is not None:
                    W_agg = self.W_prev.clone()
                else:
                    # In round 0 if everything is absurd, just average them anyway as a last resort.
                    for j, W_j in flat_models.items():
                        W_agg += W_j * (1.0 / len(flat_models))

        # Unflatten the calculated global model back to a normal state parameter dictionary
        return self._unflatten_model(W_agg, reference_dict)


    def local_process(self, W_train_dict: dict, W_agg_dict: dict, current_loss: float) -> tuple[dict, float]:
        """
        Module 2: Local Processing & Adaptive Clipping
        This module takes our fresh locally-trained model, calculates how much we updated it,
        and aggressively dynamically clips its L2-norm size locally to guarantee Differential Privacy limits.
        """
        W_train = self._flatten_model(W_train_dict)
        W_agg = self._flatten_model(W_agg_dict)

        # Calculate Deviation (Delta W) - How much did our training step actually shift the global weights?
        delta_W = W_train - W_agg
        norm_delta_W = torch.norm(delta_W, p=2).item()

        # Dynamic Window sizing algorithm
        # We stretch the memory parameter `k_t` based on the logarithmic variance of our historical updates.
        H_var = np.var(self.H) if len(self.H) > 1 else 0.0
        k_t = max(self.k_min, int(np.floor(self.alpha * np.log(1 + H_var))))

        self.H.append(norm_delta_W)
        # Cap window sequence. If it stretches beyond dynamically sized 'k_t', pop oldest records.
        while len(self.H) > k_t:
            self.H.popleft()

        # Adaptive Clipping Factor (Clip_t) - computed straight from historical local delta sizes
        self.Clip_t = np.mean(self.H) if self.H else epsilon_r
        # Avoid zero division
        if self.Clip_t <= 0:
            self.Clip_t = self.epsilon_r

        # Perform the DP Clipping
        # We only clip if norm_delta_W excedes our historical clip limit boundaries!
        clip_ratio = max(1.0, norm_delta_W / self.Clip_t)
        delta_W_prime = delta_W / clip_ratio

        # Synthesize final "Safe Model"
        # Since bounded noise will be applied LATER (outbound dispatch), we rebuild the params locally with the clipped delta!
        W_safe = W_agg + delta_W_prime

        # Update local history for next round
        self.W_prev2 = self.W_prev.clone() if self.W_prev is not None else None
        self.W_prev = W_safe.clone()

        logging.info(f"[D2B] local_process: Loss={current_loss:.6f}, Delta W norm={norm_delta_W:.6f}, Clip_t={self.Clip_t:.6f}, Clip Ratio={clip_ratio:.4f}. Window size |H|={len(self.H)}")

        return self._unflatten_model(W_safe, W_train_dict), current_loss

    def outbound_dispatch(self, W_safe_dict: dict, current_loss: float, neighbor_j: str) -> dict:
        """
        Module 3: Loss-Aware Budget & Bounded Noise
        Takes the DP-Clipped local safe model, custom tailors DP noise values exclusively based
        on the Trust index of `neighbor_j`, constraints the noise mathematically via bounds,
        and dispenses.
        """
        W_safe = self._flatten_model(W_safe_dict)
        D_i_size = len(W_safe) # Total Number of parameters in model (i.e. |D_i|)

        # Fetch Neighbor J's specific tailored trust rating!
        trust_j = self.Trust.get(neighbor_j, 1.0)

        # Calculate exactly how much zCDP DP Budget (rho_ij) we are willing to expend on Neighbor J.
        # Direct consequence: Lower trust implies we give them LESS true data and MORE noise.
        rho_ij = max(self.rho_base * trust_j, self.epsilon_r)

        clip_t_sq = float(self.Clip_t**2) if self.Clip_t else float(self.epsilon_r**2)

        # Variance calculation for noise drawing process based strictly on Differential Privacy budget theorem.
        # To avoid overflow over billions of params, ensure D_i_size^2 is float
        D_i_sq = float(D_i_size) ** 2
        sigma_ij_sq = (2 * clip_t_sq) / (D_i_sq * rho_ij)

        # Convert zCDP budget into raw mathematically pure standard differential privacy (Epsilon EPS, Delta).
        delta = 1e-5
        epsilon_ij = np.sqrt(4 * rho_ij * np.log(1/delta))

        # Bound calculations. Calculates dynamic mathematically derived parameter ceiling bounds
        # dictating exactly how tight we slice off extreme probability distribution instances of noise.
        exp_eps = np.exp(epsilon_ij)
        b = (exp_eps - self.eta) / (exp_eps + self.eta)

        # Generate Gaussian DP Noise matching the exact standard deviation allocated.
        noise = torch.normal(mean=0.0, std=np.sqrt(sigma_ij_sq), size=W_safe.shape, device=W_safe.device)

        # Apply bounds - the most critical DP privacy defense step. Trims the ends of the noise distribution.
        noise_bound = torch.clamp(noise, min=-b, max=b)

        # Output mathematically anonymized model specifically built for neighbor J.
        W_out = W_safe + noise_bound

        logging.info(f"[D2B] outbound_dispatch to {neighbor_j}: Trust={trust_j:.4f}, rho_ij={rho_ij:.6f}, eps={epsilon_ij:.6f}, std_noise={np.sqrt(sigma_ij_sq):.6f}, bound={b:.6f}")

        return self._unflatten_model(W_out, W_safe_dict)

    def post_round_update(self, current_loss: float):
        """
        Loss-Aware Budget Engine. Evaluates the local trainer's output loss at the very end of the cycle.
        If the model stops learning rapidly (e.g. plateau limits bounded by tau), it allocates greater DP noise budgets!
        """
        if self.L_prev is not None:
            # If our loss changed very little compared to yesterday (<= tau) -> we are maturing / converging.
            # So, increase our fundamental baseline limit for sending truthful data out!
            if abs(current_loss - self.L_prev) <= self.tau:
                old_rhobase = self.rho_base
                self.rho_base = min(self.rho_base * (1 + self.beta), self.rho_max)
                logging.info(f"[D2B] post_round_update: Budget updated from {old_rhobase:.6f} to {self.rho_base:.6f} (Loss Diff <= tau)")
            else:
                logging.info(f"[D2B] post_round_update: Loss diff > tau, budget unchanged {self.rho_base:.6f}")

        # Commit metric state for next round's references.
        self.L_prev = current_loss
