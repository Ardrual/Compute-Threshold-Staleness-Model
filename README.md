# Compute Threshold Staleness Model (CTSM)

A model for analyzing how compute thresholds in AI policy become "stale" as algorithmic efficiency improves. The core insight: a fixed compute threshold catches fewer risky runs over time because less compute is needed to achieve the same capabilities.

---

## Core Concept

As algorithmic efficiency improves, the same capability can be achieved with less compute. A policy threshold set today (e.g., "runs above 10²⁵ FLOPs require oversight") becomes increasingly permissive over time.

| Quantity | Formula | Interpretation |
|----------|---------|----------------|
| Efficiency multiplier | `A(t) = 2^(t/τ)` | Same compute yields `A(t)×` more capability at time `t` |
| Ideal threshold | `T*(t) = E_risk / A(t)` | Threshold halves every `τ` months |
| Staleness | `S(t) = 2^((t - tₙ)/τ)` | How much too permissive the current threshold is |
| Max staleness | `S_max = 2^(U/τ)` | Staleness just before a policy update |

---

## Quick Start

**Requirements:** Python 3.11+

```bash
# Install dependencies
uv sync

# Launch the interactive dashboard
uv run streamlit run src/ctsm/app.py
```

---

## Interactive Dashboard

### 📐 Closed-Form Analysis
Visualizes staleness accumulation and threshold decay over time with uncertainty bands for τ. Shows how different update intervals affect max staleness.

### 📊 Historical Backtest
Validates the τ parameter against real AI model data. Compares predicted vs. actual time gaps for capability-matching model pairs (e.g., when did LLaMA-13B match GPT-3?). Includes:
- 14 historical models from GPT-2 (2019) to Qwen 2.5 72B (2024)
- 6 capability-matching pairs for backtesting
- τ sensitivity analysis with best-fit optimization

### 🧮 Calculators
Direct calculations: required update interval for target staleness, catch-up timelines, threshold decay at arbitrary times.

---

## τ (Tau) Presets

The efficiency doubling time τ controls how fast the threshold becomes stale:

| Preset | τ (months) | Source |
|--------|------------|--------|
| Ho et al. 2024 | 8.0 | ~3× efficiency gain per year (frontier) |
| Backtest Best Fit | 9.5 | Minimizes RMSE on historical pairs |
| Scher 2025 (catch-up) | 2.0–2.9 | 16×–60× per year; smaller models catching up to frontier |

> [!NOTE]
> Ho et al. measure **frontier** algorithmic progress (how much better the best labs get each year). Scher measures **catch-up** progress (how quickly efficient smaller models replicate frontier capabilities). Catch-up rates appear higher because they include both innovation diffusion and post-training improvements, but they're bounded by the frontier. For policy staleness analysis, frontier progress is typically more relevant.

---

## Python API

```python
from ctsm import DriftModel

model = DriftModel(erisk=1e25, tau=8.0, update_interval=12.0, mu0=57.0, r=0.12, sigma=2.0)

# Core calculations (no simulation needed)
threshold = model.threshold(t=8.0)
staleness = model.staleness(t=8.0)

# Monte Carlo simulation for FNR/coverage
metrics = model.simulate_metrics(t=8.0, n=10000)

# With strategic gaming scenario
metrics_strategic = model.simulate_metrics(
    t=8.0,
    n=10000,
    scenario="threshold-sensitive",
    strategic_share=0.2,
    strategic_beta=0.9,
    strategic_sigma=1.0,
)
```

---

## CLI

```bash
# Core computations
uv run ctsm efficiency --t 6 --tau 8
uv run ctsm threshold --t 6 --erisk 1e25 --tau 8 --update-interval 12
uv run ctsm staleness --t 8 --erisk 1e25 --tau 8 --update-interval 12

# Monte Carlo simulation
uv run ctsm simulate --t 6 --n 10000 --erisk 1e25 --tau 8 --update-interval 12 \
  --mu0 57 --r 0.12 --sigma 2.0 --seed 42 --json

# With threshold-gaming scenario
uv run ctsm simulate --t 6 --n 10000 --erisk 1e25 --tau 8 --update-interval 12 \
  --mu0 57 --r 0.12 --sigma 2.0 --scenario threshold-sensitive \
  --strategic-share 0.2 --strategic-beta 0.9 --strategic-sigma 1.0 --seed 42 --json
```

---

## Historical Backtest Data

The backtest uses real AI model releases to validate τ estimates. Key capability-matching pairs:

| Reference Model | Matching Model | Compute Ratio | Time Gap |
|-----------------|----------------|---------------|----------|
| GPT-3 175B | LLaMA-13B | 0.35× | 32 mo |
| GPT-3 175B | Mistral 7B | 0.27× | 40 mo |
| GPT-4 | LLaMA 3 70B | 0.43× | 13 mo |
| GPT-4 | Qwen 2.5 72B | 0.36× | 18 mo |

Models that used *more* compute than their reference are excluded (they test data efficiency, not algorithmic efficiency).

---

## Tests

```bash
uv run pytest
```

---

## References

- **Ho et al. (2024)**: "Algorithmic progress in language models" — empirical estimate of ~8-month efficiency doubling time for frontier models
- **Scher (2025)**: Analysis of catch-up efficiency gains in smaller models replicating frontier capabilities
