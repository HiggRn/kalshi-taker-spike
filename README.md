# ORIE 5257 Capstone Proposal

Group 1: Wiliam Chang (cc2983), Zijie Gu (zg353)

## Market and Protocol

- Venue and Mechanism: Kalshi centralized limit order book (CLOB) exchange for binary event contracts, continuous 24/7 double auction with strict price-time (FIFO) priority.
- Protocol Constraints:
    - Contract prices are bounded on a discrete grid $P_t\in[0.01,0.99]$ with a fixed tick size of \$$0.01$ or \$$0.001$
    - Convex taker fee schedules that peak at $P_t = 0.50$ and decay toward the boundaries ($0$ and $1$).
$$\text{Taker fee} = 0.07 * \text{contract size} * P_t * (1-P_t)$$
    - Dual-book equivalence, where a resting bid for a YES contract at price $p$ is matched directly against an incoming market/limit order for a NO contract at price $1 - p$.
- Observability: The trading protocol exposes real-time and historical L2 aggregated queue depths across price levels alongside individual trade prints (timestamp, execution price, contract volume, and taker side). Individual trader identities, hidden reserve quantities, and exact FIFO queue positions of specific resting orders remain unobserved.

## Research Question

- Primary Question: In an event-time framework, can nonlinear machine learning and deep learning architectures leveraging L2 order book state and trade flow outperform linear microstructure and self-exciting point-process baselines in predicting directional taker flow spikes on Kalshi?
- Transfer Question: To what extent does a model trained on a highly liquid contract transfer (zero-shot) to
    - (i) another highly liquid contract;
    - (ii) an illiquid contract characterized by sparse L2 queues?

## Available Data

Synchronized L2 1-minute order book snapshots and tick-by-tick trade records (`timestamp`, `price`, `size`, `taker_side`) across three representative Kalshi binary contracts:
1. **`CONTROLS`, 2026 Mid-Term Senate Control**: Primary Training & In-Domain Test, High Liquidity, Prices range \$0.30 - \$0.70, tick size \$0.01.
2. **`SENATETX`, 2026 Mid-Term Texas Senator Seat**: Homogeneous Transfer Test, High Liquidity, \$0.30 - \$0.70, \$0.01.
3. **`SENATEGA`, 2026 Mid-Term Georgia Senator Seat**: Heterogeneous Transfer Test, Low Liquidity, edge prices \$0.10- and \$0.90+, \$0.001.

## Proposed Method

### Cross Book Events
- Unify direction: Republican yes = Democrat no, therefore we assess prices of Republican yes and Democrat no (1 - Democrat yes).
- Apply algorithm and training processes separately to different books, but predict all four sides of taker events (Republican/ Democrat $\times$ Yes/NO)

### Event-Time Clock & Scale-Invariant Target Construction

- Advance the system clock on market events by each party $p \in \{\text{Republican}, \text{Democrat}\}$ and sides $s \in \{\text{Yes}, \text{No}\}$:  $$k_{p, s} \in \{1_{p, s}, 2_{p, s}, \dots, N_{p, s}\}$$ (trades or L2 updates) rather than uniform wall-clock intervals to handle irregular arrivals across liquid and illiquid regimes.
- At event $k_{p,s}$, compute rolling baseline statistics (e.g., rolling mean $\mu_{k_{p,s}}$ and standard deviation $\sigma_{k_{p,s}}$, or empirical $(1-\alpha)$-quantile $Q^{1-\alpha}_{k_{p,s}}$) of directional taker volume over a purely backward-looking trailing window of $W$ events $[k-W, k]$ to prevent look-ahead leakage.
- Define the binary target $Y_k \in \{0, 1\}$ over a forward event horizon $H$ as a directional taker flow spike:

$$
Y_{k_{p,s}} = \mathbb{I}\left( \sum_{j=k_{p,s}+1}^{k_{p,s}+H} V_j^{\text{taker}} > \mu_{k_{p,s}} + c \cdot \sigma_{k_{p,s}} \right) \quad \text{or} \quad \mathbb{I}\left( \sum_{j=k_{p,s}+1}^{k_{p,s}+H} V_j^{\text{taker}} > Q^{1-\alpha}_{k_{p,s}}(V_{{k_{p,s}}-W:k_{p,s}}^{\text{taker}}) \right)
$$

### Feature Engineering & Model Hierarchy

- Potential Feature Set: Multi-level Order Book Imbalance (OBI), Multi-level Order Flow Imbalance (OFI), Trade Flow Imbalance (TFI) across multiple event-time horizons, relative bid-ask spread, book slope, microprice deviation, log-odds transformed price $\log(P_{k_{p,s}} / (1 - P_{k_{p,s}}))$, and log inter-event physical duration $\ln(\Delta t_{k_{p,s}})$.
- Models:
    - Linear Family: $L_1/L_2$-penalized Logistic Regression (GLM) trained on normalized event-time microstructure features.
    - Tabular ML Family: Gradient Boosted Decision Trees (LightGBM) to capture nonlinear threshold interactions between queue depletion, spread widening, and inter-trade duration.
    - Deep Learning Family: LSTM, GRU or transformer
- Meaningful Benchmarks:
    - Trade-Only Self-Exciting Baseline: A univariate Hawkes-inspired exponentially weighted moving average (EWMA) of past taker volumes and inter-trade arrival times (testing whether L2 order book depth adds predictive value beyond simple trade clustering).
    - Microstructure Linear Baseline: A single-feature L1 Queue Imbalance / TFI logistic model.

## Evaluation Design

- Leakage Controls & Splits: Strict chronological walk-forward split (Train / Validation / Test) on all contracts, enforcing a mandatory timestamp latency buffer between L2 snapshot observation and trade label evaluation. Test contracts are held out entirely for locked out-of-sample transfer testing and we'll only use their test period data to avoid leakage of the same macro events.
- Statistical Metrics: Precision-Recall AUC (PR-AUC) and F1-score at top-$K\%$ predicted probabilities (accounting for class imbalance of spike events), alongside Brier Score and Expected Calibration Error (ECE).
- Market Metric: Simulated Adverse Selection Avoidance for a passive L1 liquidity provider—measuring the realized spread preservation and reduction in post-trade markout losses (at $H$ events post-fill) achieved by canceling or widening L1 quotes when a taker spike is predicted, net of missed-fill opportunity costs and Kalshi fee schedules.

## Result Counting Against Hypothesis

- In-Domain Falsification: If LightGBM and the Continuous-Time DL architecture fail to achieve a statistically significant improvement in out-of-sample PR-AUC or net markout avoidance over the regularized linear GLM or trade-only EWMA baseline on the main contract, we reject the hypothesis that complex nonlinear L2 interactions drive short-horizon taker spikes.
- Transfer Falsification: If nonlinear and deep models suffer severe calibration collapse on illiquid contract (performing worse than the simple linear baseline despite event-time normalization and relative rolling thresholds), this establishes an empirical boundary showing that high-capacity microstructure models overfit to high-liquidity queue dynamics.

## Main Feasibility Risks

- Timestamp Alignment & Asynchronous Updates: Misalignment between L2 order book snapshot polling timestamps and individual trade execution timestamps could introduce subtle look-ahead leakage if trade-induced book updates are recorded prior to the trade print itself.
- Lookback Window Warm-Up & Sparsity in Illiquid Markets: On the highly illiquid contract, an event-time lookback window of $W$ events may span several hours or days of physical wall-clock time, potentially mixing distinct macroeconomic information regimes and reducing the total count of positive spike instances available for tight confidence intervals.
