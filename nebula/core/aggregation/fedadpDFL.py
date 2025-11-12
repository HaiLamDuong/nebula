import gc
import torch
import math
import logging
from nebula.core.aggregation.aggregator import Aggregator

class FedAdpDFL(Aggregator):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self._local_model = None
        self.round = self.engine.round
        self._prev_smoothed = None
        logging.info(f"[{self.__class__.__name__}] Init FedAdp")


    def get_local_model(self, models):
        if self._addr not in models:
            raise ValueError(f"[{self.__class__.__name__}] Local model ({self._addr}) not found in models")
        # Unpack tuple to get model_params only
        model_params, _ = models[self._addr]
        return model_params


    def calculate_cosine_similarity(self, models, local_model):
        """
        Calculate cosine similarity between the local model and each received model.

        Args:
            models (dict): { node_addr: (model_state_dict, weight) }
            local_model (dict): Local model parameters (state_dict)

        Returns:
            dict: { node_addr: cosine_similarity_value }
        """
        similarities = {}

        # Flatten local model into a single vector
        local_vec = []
        for layer in sorted(local_model.keys()):
            local_vec.append(local_model[layer].detach().flatten())
        local_vec = torch.cat(local_vec)
        local_norm = torch.norm(local_vec, p=2).item()

        if local_norm == 0:
            logging.debug("[calculate_cosine_similarity] Local model norm is zero. Returning zeros.")
            return {addr: 0.0 for addr in models.keys()}

        for node_addr, (model_params, _) in models.items():
            if node_addr == self._addr:
                # skip self comparison
                continue

            # Flatten remote model
            remote_vec = []
            for layer in sorted(model_params.keys()):
                remote_vec.append(model_params[layer].detach().flatten())
            remote_vec = torch.cat(remote_vec)

            # Compute cosine similarity
            dot = torch.dot(local_vec, remote_vec).item()
            client_norm = torch.norm(remote_vec, p=2).item()

            if client_norm == 0:
                cosine_sim = 0.0
            else:
                cosine_sim = dot / (local_norm * client_norm)
                cosine_sim = max(-1.0, min(1.0, cosine_sim))  # numerical safety

            similarities[node_addr] = cosine_sim
            logging.debug(f"[{self.__class__.__name__}] Cosine similarity({self._addr}, {node_addr}) = {cosine_sim:.4f}")
        logging.debug(f"[{self.__class__.__name__}] finish cosine similarities  {similarities}")
        return similarities


    def smooth_cosine_similarity(self, similarities):
        """
        Smooth the cosine similarity values according to FedAdp:
            θ̃_i(t) = θ_i(t)                    if t == 1
                    = ((t-1)/t)*θ̃_i(t-1) + (1/t)*θ_i(t)   if t > 1
        Args:
            similarities (dict): { node_addr: θ_i(t) }  — cosine similarities at current round
        Returns:
            dict: { node_addr: θ̃_i(t) } — smoothed cosine similarities
        """
        t = self.engine.round # current training round
        logging.debug(f"[smooth_cosine_similarity] t = {t}")
        smoothed = {}

        # Dictionary lưu trạng thái θ̃_i(t-1)
        if self._prev_smoothed is None:
            self._prev_smoothed = {}
            logging.debug("[smooth_cosine_similarity] previous smoothed is None at round {self.engine.round}")

        for node_addr, theta_t in similarities.items():

            if t == 0:
                # Lần đầu: không có giá trị trước → lấy luôn giá trị hiện tại
                smoothed_theta = theta_t
            else:
                prev_smooth = self._prev_smoothed.get(node_addr, 0.0)
                # Làm mượt theo công thức FedAdp
                smoothed_theta = ((t - 1) / t) * prev_smooth + (1 / t) * theta_t

            smoothed[node_addr] = smoothed_theta
            # Lưu lại để dùng cho vòng tiếp theo
            self._prev_smoothed[node_addr] = smoothed_theta
        logging.debug(f"[{self.__class__.__name__}] finish smoothing cosine similarities  {smoothed}")
        return smoothed


    def get_non_linear_mapping(self, smoothed):
        """
        Apply the nonlinear mapping function from FedAdp:
            f(θ̃_i(t)) = α * (1 - exp(-exp(-α * (θ̃_i(t) - 1))))

        Args:
            smoothed (dict): { node_addr: θ̃_i(t) } — smoothed cosine similarities

        Returns:
            dict: { node_addr: f(θ̃_i(t)) } — mapped values after nonlinear transformation
        """
        alpha = 2.0
        mapped = {}

        for node_addr, theta_smooth in smoothed.items():
            # f(θ̃_i(t)) = α * (1 - e^{-e^{-α(θ̃_i(t)-1)}})
            inner_exp = -alpha * (theta_smooth - 1)
            value = alpha * (1 - math.exp(-math.exp(inner_exp)))

            mapped[node_addr] = value
            logging.debug(f"[{self.__class__.__name__}] f(θ̃_{node_addr}) = {value:.6f}")

        logging.debug(f"[{self.__class__.__name__}] Finish nonlinear mapping  {mapped}")
        return mapped


    def softmax(self, mapped, weights):
        """
        Compute FedAdp-style softmax weighting:
            ψ_i(t) = (D_i * exp(f(θ̃_i(t)))) / Σ_j [D_j * exp(f(θ̃_j(t)))]
        Args:
            mapped (dict): { node_addr: f(θ̃_i(t)) } — output from get_non_linear_mapping()
            weights (dict): { node_addr: D_i } — dataset sizes or client weights
        Returns:
            dict: { node_addr: ψ_i(t) } — normalized softmax weights for aggregation
        """
        weighted_exps = {}

        # Step 1: D_i * exp(f(θ̃_i(t)))
        for node_addr, f_theta in mapped.items():
            Di = weights.get(node_addr, 1.0)
            value = Di * math.exp(f_theta)
            weighted_exps[node_addr] = value
            logging.debug(f"[{self.__class__.__name__}] Weighted exp for {node_addr}: D_i*e^(fθ) = {value:.6f}")

        # Step 2: Tổng mẫu số
        denom = sum(weighted_exps.values()) + 1e-12  # tránh chia 0

        # Step 3: Chuẩn hóa thành softmax
        softmax_weights = {}
        for node_addr, value in weighted_exps.items():
            psi = value / denom
            softmax_weights[node_addr] = psi
            logging.debug(f"[{self.__class__.__name__}] ψ_i({node_addr}) = {psi:.6f}")

        logging.debug(f"[{self.__class__.__name__}] Finish softmax  {softmax_weights}")
        return softmax_weights


    def run_aggregation(self, models):
        """
        Implements Balance aggregation:
        1. Filter models using remove_malicious_models.
        2. Aggregate using result = a * wi + (1-a) * (1/S) * sum(wj).

        Args:
            models (dict): Dictionary of model updates, where keys are node addresses
                          and values are tuples of (model_parameters, weight).

        Returns:
            dict: Aggregated model parameters.
        """
        super().run_aggregation(models)

        # init weight for each client
        weights = {}
        for addr, (_, weight) in models.items():
            weights[addr] = weight

        local_model = self.get_local_model(models)

        similarities = self.calculate_cosine_similarity(models, local_model)
        smoothed = self.smooth_cosine_similarity(similarities)
        mapped = self.get_non_linear_mapping(smoothed)

        adpaptive_weights = self.softmax(mapped, weights)

        # Step D: aggregate model parameters using adaptive_weights
        logging.info(f"[{self.__class__.__name__}] Step E: Aggregating model parameters using adaptive weights")
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in local_model.items()}
        with torch.no_grad():
            for node_addr, (model_parameters, _) in models.items():
                if node_addr == self._addr:
                    continue
                w = adpaptive_weights.get(node_addr, 0.0)
                if w <= 0.0:
                    continue
                for layer in accum:
                    accum[layer].add_(model_parameters[layer].to(accum[layer].dtype), alpha=w)

        alpha = 0.2
        for layer in accum:
            accum[layer].mul_(1-alpha)
            accum[layer].add_(local_model[layer], alpha=(alpha))

        logging.info(f"[{self.__class__.__name__}] ========== FedAdp-DFL Aggregation Round Completed ==========")
        gc.collect()
        return accum
