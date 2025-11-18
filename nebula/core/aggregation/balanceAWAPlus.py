import torch
import torch.nn.functional as F
import logging
import math
import gc
from nebula.core.aggregation.aggregator import Aggregator

class BalanceAWAPlus(Aggregator):
    """
    Hybrid Aggregator: Balance + FedAWA (Pure PyTorch Version)

    Process:
    1. Flatten Models (Pure PyTorch)
    2. Phase 1: Balance Filtering (Filter based on dynamic distance threshold)
    3. Phase 2: FedAWA Optimization (Calculate adaptive weights for *accepted* models)
    4. Aggregation & Momentum
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

        # --- Balance Parameters ---
        self.A = 2.0             # Threshold scale factor
        self.K = 1.0             # Decay rate
        self.balance_alpha = 0.4 # Weight of Local Model in final combination

        # --- FedAWA Parameters ---
        self.fedawa_alpha = 0.2  # Regularization weight
        self.lr = 0.02       # Learning rate
        self.steps = 50      # Optimization steps

        self.adaptive_weights = None

        # Auto-detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"[{self.__class__.__name__}] Initialized on {self.device}")

    def get_local_model(self, models):
        if self._addr not in models:
            if self._local_model is not None:
                return self._local_model
            raise ValueError(f"[{self.__class__.__name__}] Local model not found.")
        model_params, _ = models[self._addr]
        return model_params

    def _flatten_model(self, model_state_dict):
        """
        Flatten model params into a single 1D Tensor on self.device.
        """
        return torch.cat([param.flatten().to(self.device) for param in model_state_dict.values()])

    def phase1_balance_filtering_and_prepare_matrices(self, models, local_model):
        """
        Kết hợp Phase 1 (Lọc) và chuẩn bị Ma trận cho Phase 2.
        Thay vì làm 2 bước tách biệt tốn kém, ta làm 1 lần loop.
        """
        # 1. Chuẩn bị thông tin Round & Threshold
        try:
            # Giả sử self.engine có thông tin round
            current_round = self.engine.round + 1
            total_rounds = self.engine.total_rounds
        except AttributeError:
            current_round = 1
            total_rounds = 100 # Fallback

        # Flatten Global Model
        theta_g_vec = self._flatten_model(local_model)

        # Tính Threshold Balance
        local_norm = torch.norm(theta_g_vec, p=2).item()
        threshold = self.A * math.exp(-self.K * current_round / total_rounds) * local_norm

        logging.info(f"Balance Threshold (R{current_round}/{total_rounds}): {threshold:.4f} | Norm: {local_norm:.4f}")

        # 2. Lọc và xây dựng List
        node_addrs = []
        theta_list = [] # List các vector theta (đã được chấp nhận)
        tau_list = []   # List các vector tau (đã được chấp nhận)

        with torch.no_grad():
            for node_addr, (model_params, _) in models.items():
                if node_addr == self._addr:
                    continue

                # Flatten Client Model
                theta_k_vec = self._flatten_model(model_params)

                # --- BALANCE CHECK ---
                # Tính khoảng cách Euclidean
                dist = torch.norm(theta_k_vec - theta_g_vec, p=2).item()

                if dist > threshold:
                    logging.debug(f"Reject {node_addr}: dist={dist:.4f} > {threshold:.4f}")
                    continue # Bỏ qua model này, không thêm vào list

                logging.debug(f"Accept {node_addr}: dist={dist:.4f}")

                # Nếu được chấp nhận, chuẩn bị dữ liệu cho FedAWA
                # Tính Tau = Theta_k - Theta_g
                tau_k_vec = theta_k_vec - theta_g_vec

                node_addrs.append(node_addr)
                theta_list.append(theta_k_vec)
                tau_list.append(tau_k_vec)

        if not theta_list:
            return None, None, None, None

        # 3. Stack thành ma trận (Pure PyTorch)
        # dim=1 -> Vector cột (D, K)
        T = torch.stack(tau_list, dim=1)
        Theta = torch.stack(theta_list, dim=1)

        return T, Theta, theta_g_vec, node_addrs

    def calculate_adaptive_weight(self, T, Theta, theta_g_vec,
                                  alpha=1.0, lr=0.02, steps=50):
        """
        Logic tối ưu hóa y hệt FedAWADFLPlus bạn cung cấp.
        """
        # T, Theta, theta_g_vec ĐÃ ở trên self.device

        # Chuẩn hóa (Normalization)
        eps = 1e-8
        T = T / (torch.norm(T, dim=0, keepdim=True) + eps)
        Theta = Theta / (torch.norm(Theta, dim=0, keepdim=True) + eps)

        K = T.shape[1]

        # Init Logits
        logits = torch.zeros(K, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([logits], lr=lr)

        # Loop tối ưu
        for i in range(steps):
            optimizer.zero_grad()
            lam = F.softmax(logits, dim=0)

            # Matrix Multiplication
            tau_agg = torch.mv(T, lam)
            theta_agg = torch.mv(Theta, lam)

            # Loss Calculation
            diff_tau = T - tau_agg.unsqueeze(1)
            # dist_sq = ||tau_k - tau_agg||^2
            dist_sq = torch.sum(diff_tau**2, dim=0)

            loss_variance = torch.sum(lam * dist_sq)
            loss_reg = alpha * torch.sum((theta_agg - theta_g_vec)**2)

            loss = loss_variance + loss_reg

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_lam = F.softmax(logits, dim=0).cpu().numpy()

        return final_lam

    def run_aggregation(self, models):
        super().run_aggregation(models)
        logging.info(f"[{self.__class__.__name__}] ========== Balance-AWA Aggregation ==========")

        local_model = self.get_local_model(models)

        # --- Phase 1: Filter & Prepare Matrices ---
        # Hàm này vừa lọc Balance, vừa tạo ma trận T, Theta cho những node được nhận
        T, Theta, theta_g_vec, node_addrs = self.phase1_balance_filtering_and_prepare_matrices(models, local_model)

        if T is None:
            logging.warning("No models passed Balance filtering. Returning local model.")
            return local_model

        # --- Phase 2: Calculate Weights (FedAWA Optimization) ---
        logging.info(f"[{self.__class__.__name__}] Optimizing weights for {len(node_addrs)} accepted models...")

        lam_values = self.calculate_adaptive_weight(
            T, Theta, theta_g_vec,
            alpha=self.fedawa_alpha, lr=self.lr, steps=self.steps
        )

        adaptive_weights_dict = {node_addrs[i]: float(lam_values[i]) for i in range(len(node_addrs))}
        self.adaptive_weights = adaptive_weights_dict
        logging.info(f"Adaptive Weights: {adaptive_weights_dict}")

        # --- Phase 3: Weighted Aggregation ---
        accum = {k: torch.zeros_like(v) for k, v in local_model.items()}
        total_w = 0.0

        with torch.no_grad():
            for node_addr, (model_params, _) in models.items():
                # Chỉ cộng những node có trong danh sách đã lọc
                w = adaptive_weights_dict.get(node_addr, 0.0)

                if w < 1e-6: continue

                total_w += w
                for name, param in model_params.items():
                    accum[name].add_(param.to(accum[name].device), alpha=w)

        # --- Phase 4: Global Momentum (Balance Logic) ---
        # Kết hợp model tổng hợp với model cũ của chính mình
        # New = (1 - balance_alpha) * Aggregated + balance_alpha * Local

        if total_w < 1e-3:
            logging.warning("Total weight too small after optimization. Reverting to local model.")
            accum = local_model
        else:
            # Normalize aggregated model (nếu cần thiết, thường softmax đã đảm bảo tổng xấp xỉ 1)
            # Nhưng vì ta có lọc model, tổng weight = 1.
            logging.info(f"Combining: {1-self.balance_alpha} Agg + {self.balance_alpha} Local")
            for name in accum:
                accum[name].mul_(1 - self.balance_alpha)
                accum[name].add_(local_model[name].to(accum[name].device), alpha=self.balance_alpha)

        self._local_model = accum

        # Cleanup
        del T, Theta, theta_g_vec
        torch.cuda.empty_cache()
        gc.collect()

        return accum
