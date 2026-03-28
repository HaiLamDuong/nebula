# Integration of D2B-DPFL into Nebula

This document summarizes the complete step-by-step implementation of the **Self-Adaptive Trust-Aware Decentralized Dynamic-Bounded DPFL (D2B-DPFL)** algorithm into the Nebula Differential Federated Learning framework.

## 1. Core Algorithm Implementation
**File:** `nebula/core/dpfl/d2b.py`

We created a stateful, strictly-vectorized module `SelfAdaptiveD2BNode` representing the algorithmic core of D2B-DPFL. Operations on tensors were heavily flattened to handle math across hundreds of thousands of parameters rapidly. The logic is separated into 4 distinct lifecycle mechanisms:

- **Module 1: Inbound Trust & Evaluation (`inbound_evaluate`)**
  - Extracts incoming model deviations (pseudo-gradients) against historical matrices.
  - Computes exact `Z-Scores` natively using Absolute Average Rates.
  - Dynamically calculates a volatile anomaly network envelope threshold ($Z_{thresh}$) using Median Absolute Deviation (MAD).
  - Calculates Trust via Size Penalties ($P_{safe}$) and Directional Trajectory matching ($P_{sim}$).
  - Uses an exponential moving average powered by a penalty scalar ($\lambda_{ij}$) to remember and heavily punish malicious betrayal.
  - **Strict Rejection:** Analyzes if any update crosses the $Z_{thresh}$ boundary. Over-the-limit neighbor models are violently rejected, scoring $0.0$, skipped in the aggregation averaging, and completely nullified. If all models are bad, the node safely falls back to its previous local state to avoid database corruption.

- **Module 2: Local Processing & Adaptive Clipping (`local_process`)**
  - Measures precisely how large the local training shift delta was vs the global model ($\Delta W$).
  - Implements a continuously dynamic log-based window sequence $H_t$ that scales based on local historical update variance.
  - Generates the $Clip_t$ boundary limit and automatically shrinks L2-norm scales of the model parameters to guarantee mathematically rigorous bounds prior to DP processing.

- **Module 3: Outbound Distribution & Noise Generation (`outbound_dispatch`)**
  - Runs uniquely **per neighbor connection**.
  - Consults the latest $Trust_j$ score to identify exactly how much privacy budget $\rho_{ij}$ should be allocated for that explicit neighbor.
  - Converts zCDP budgets to Standard $\epsilon, \delta$ Differential Privacy representations.
  - Implements heavy parameter clamps (`b` bounds) and adds Gaussian noise strictly to limit extremities before routing.

- **Module 4: Budget Adjustment (`post_round_update`)**
  - Tracks convergence. If learning loss improvement completely stalls out (drops by $\le \tau$), it rewards the agent by mathematically multiplying and expanding the DP base budget constraint ($\rho_{base}$) scaling up to a ceiling limit.

## 2. Aggregator Wrapper Bridge
**Files:**
- `nebula/core/aggregation/d2b_aggregator.py`
- `nebula/core/aggregation/aggregator.py`

- Built `D2BAggregator` leveraging the `Aggregator` base class interface.
- Imports configuration constraints dynamically from the `.yaml` files utilizing the `dpfl_args` sub-dictionary.
- Overrides `run_aggregation` to pass parsed neighbor payload parameters directly to `inbound_evaluate`.
- Registered natively in the `ALGORITHM_MAP` meaning that passing `algorithm: D2BDpfl` correctly routes the framework logic to bypass classic `FedAvg` in exchange for our Trust logic.

## 3. Training Interception (Local Process Injection)
**File:** `nebula/core/noderole.py`

To accurately calculate $\Delta W$ training shifts, we intercepted the fundamental local node training architectures:
- Overrided `extended_learning_cycle` routines inside of `TrainerRoleBehavior` and `TrainerAggregatorRoleBehavior`.
- After waiting for `trainer.train()` to finish cleanly executing, we snapshot `trainer.get_model_parameters()`.
- Send the local weights and the aggregated weights tightly to `local_process`.
- Overwrote the trainer's local weights immediately via `trainer.set_model_parameters(safe_model)` substituting the naked gradients with strictly L2-Clipped DP gradients.
- Triggered `post_round_update` with the round's resulting loss trajectory.

## 4. Network Transport Tailoring
**File:** `nebula/core/network/propagator.py`

Original implementations dispatched the same global exact model to all neighboring devices uniformly. D2B-DPFL relies mathematically on allocating varied levels of noise specific to the historical reputation of the exact recipient.
- Intercepted `StableModelPropagation`.
- Refactored `prepare_model_payload` to explicitly accept the `node` (neighbor's address hash).
- When looping through neighbors preparing message streams, if `d2b_node` exists, we run `outbound_dispatch` in-time substituting the payload parameters with wildly different distributions of tailored Trust-weighted Gaussian noise.
- Sent exactly bounded representations over standard socket transport pipelines successfully natively inside Nebula core.

## 5. Extensive Logging & Diagnostics
Added precise internal console tagging (`logging.info` and `logging.debug`) tagged tightly under `[D2B]` flags so runtime observations correctly yield anomaly Z-Scores, Trust shifting metrics, real-time threshold ceilings, dynamically computed epsilons ($\epsilon$), dynamic bounds, array sizes, and exact DP budget adjustments.
