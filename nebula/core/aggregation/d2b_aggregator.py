import logging
from nebula.core.aggregation.aggregator import Aggregator
from nebula.core.dpfl.d2b import SelfAdaptiveD2BNode

class D2BAggregator(Aggregator):
    def __init__(self, config=None, engine=None):
        super().__init__(config, engine)

        # Read parameters from config
        dpfl_args = config.participant.get("dpfl_args", {})
        rho_min = dpfl_args.get("rho_min", 0.1)
        rho_max = dpfl_args.get("rho_max", 5.0)
        T_target = dpfl_args.get("T_target", 50)
        k_min = dpfl_args.get("k_min", 5)
        alpha = dpfl_args.get("alpha", 1.0)
        gamma_base = dpfl_args.get("gamma_base", 1.0)
        mu = dpfl_args.get("mu", 0.1)
        w1 = dpfl_args.get("w1", 0.7)
        w2 = dpfl_args.get("w2", 0.3)
        lambda_max = dpfl_args.get("lambda_max", 0.8)
        tau = dpfl_args.get("tau", 0.01)
        eta = dpfl_args.get("eta", 0.5)
        epsilon_r = dpfl_args.get("epsilon_r", 1e-6)

        self.d2b_node = SelfAdaptiveD2BNode(
            rho_min=rho_min,
            rho_max=rho_max,
            T_target=T_target,
            k_min=k_min,
            alpha=alpha,
            gamma_base=gamma_base,
            mu=mu,
            w1=w1,
            w2=w2,
            lambda_max=lambda_max,
            tau=tau,
            eta=eta,
            epsilon_r=epsilon_r
        )
        logging.info("[D2BAggregator] Initialized D2B-DPFL node with adaptive mechanisms.")

    def run_aggregation(self, models):
        super().run_aggregation(models)

        if models is None or len(models) == 0:
            return None

        # models is a dict: {neighbor_addr: (model_params, weight)}
        # We need to extract just the model_params for inbound_evaluate
        received_models = {k: v[0] for k, v in models.items()}

        aggregated_model = self.d2b_node.inbound_evaluate(received_models)

        return aggregated_model
