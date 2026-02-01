"""Historical AI model data for backtesting the CTSM efficiency model.

This module contains training compute estimates and release dates for notable
AI models, along with capability-matching pairs used to validate the efficiency
doubling time (tau) parameter.

Data sources:
- Epoch AI (epochai.org)
- SemiAnalysis
- Model papers and technical reports
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple
import math


@dataclass(frozen=True)
class HistoricalModel:
    """A historical AI model with known training compute and release date."""
    
    name: str
    release_date: date
    training_flops: float  # Total training FLOPs
    parameters: Optional[float] = None  # Model parameters (billions)
    notes: str = ""
    
    @property
    def log_flops(self) -> float:
        """Natural log of training FLOPs."""
        return math.log(self.training_flops)


@dataclass(frozen=True)
class CapabilityMatch:
    """A pair of models where the later/smaller one matches the earlier/larger one's capability."""
    
    reference: HistoricalModel  # Earlier, typically larger model
    matching: HistoricalModel   # Later model that achieved similar capability
    benchmark: str              # Benchmark or capability dimension
    notes: str = ""
    
    @property
    def time_gap_months(self) -> float:
        """Actual time gap between models in months."""
        delta = self.matching.release_date - self.reference.release_date
        return delta.days / 30.44  # Average days per month
    
    @property
    def compute_ratio(self) -> float:
        """Ratio of matching model's compute to reference model's compute."""
        return self.matching.training_flops / self.reference.training_flops
    
    def predicted_time_gap(self, tau: float) -> float:
        """
        Predict the time gap needed for capability parity given efficiency doubling time tau.
        
        If E₁ = E₂ (same effective compute), then:
            A(t₁) × C₁ = A(t₂) × C₂
            2^(t₁/τ) × C₁ = 2^(t₂/τ) × C₂
            2^((t₂-t₁)/τ) = C₁/C₂
            Δt = τ × log₂(C₁/C₂)
        
        Returns predicted time gap in months.
        """
        if self.compute_ratio <= 0:
            return float('nan')
        # log₂(C₁/C₂) = log₂(1 / compute_ratio) = -log₂(compute_ratio)
        return -tau * math.log2(self.compute_ratio)
    
    def prediction_error(self, tau: float) -> float:
        """
        Error between predicted and actual time gap (in months).
        Positive = model over-predicts efficiency gains (predicted gap < actual).
        """
        return self.time_gap_months - self.predicted_time_gap(tau)


# =============================================================================
# Historical Model Database
# =============================================================================

# Reference date: GPT-2 release (Feb 2019) = month 0
BASELINE_DATE = date(2019, 2, 1)

def months_from_baseline(d: date) -> float:
    """Convert a date to months from baseline (Feb 2019)."""
    delta = d - BASELINE_DATE
    return delta.days / 30.44


# Notable models with training compute estimates
MODELS = {
    # --- 2019 Era ---
    "gpt2": HistoricalModel(
        name="GPT-2",
        release_date=date(2019, 2, 14),
        training_flops=2.5e21,
        parameters=1.5,
        notes="OpenAI, 1.5B params, ~40GB training data",
    ),
    # --- 2020 Era ---
    "gpt3": HistoricalModel(
        name="GPT-3 175B",
        release_date=date(2020, 6, 11),
        training_flops=3.14e23,
        parameters=175,
        notes="OpenAI, 175B params, 300B tokens",
    ),
    # --- 2022 Era ---
    "chinchilla": HistoricalModel(
        name="Chinchilla 70B",
        release_date=date(2022, 3, 29),
        training_flops=1e24,
        parameters=70,
        notes="DeepMind, compute-optimal scaling, 1.4T tokens",
    ),
    "gopher": HistoricalModel(
        name="Gopher 280B",
        release_date=date(2021, 12, 8),
        training_flops=6.31e23,  # Epoch AI
        parameters=280,
        notes="DeepMind, 280B params, ~60% MMLU",
    ),
    # --- 2023 Era ---
    "llama1_13b": HistoricalModel(
        name="LLaMA-13B",
        release_date=date(2023, 2, 24),
        training_flops=1.09e23,  # 6 * 13e9 * 1.4e12
        parameters=13,
        notes="Meta, 13B params, 1.4T tokens",
    ),
    "llama1_33b": HistoricalModel(
        name="LLaMA-33B",
        release_date=date(2023, 2, 24),
        training_flops=2.77e23,  # 6 * 33e9 * 1.4e12
        parameters=33,
        notes="Meta, 33B params, 1.4T tokens, HellaSwag ~83%",
    ),
    "llama1_65b": HistoricalModel(
        name="LLaMA-65B",
        release_date=date(2023, 2, 24),
        training_flops=5.46e23,  # 6 * 65e9 * 1.4e12
        parameters=65,
        notes="Meta, 65B params, 1.4T tokens",
    ),
    "gpt4": HistoricalModel(
        name="GPT-4",
        release_date=date(2023, 3, 14),
        training_flops=2.15e25,
        parameters=1760,  # Estimated total across experts
        notes="OpenAI, MoE architecture, 16x111B experts",
    ),
    "falcon_40b": HistoricalModel(
        name="Falcon 40B",
        release_date=date(2023, 5, 25),
        training_flops=2.4e23,  # 6 * 40e9 * 1e12 tokens
        parameters=40,
        notes="TII, 40B params, 1T tokens, HellaSwag ~85%",
    ),
    "llama2_70b": HistoricalModel(
        name="LLaMA 2 70B",
        release_date=date(2023, 7, 18),
        training_flops=8.4e23,  # 6 * 70e9 * 2e12 tokens
        parameters=70,
        notes="Meta, 70B params, 2T tokens",
    ),
    "llama2_13b": HistoricalModel(
        name="LLaMA 2 13B",
        release_date=date(2023, 7, 18),
        training_flops=1.56e23,  # 6 * 13e9 * 2e12 tokens
        parameters=13,
        notes="Meta, 13B params, 2T tokens, MMLU ~55%",
    ),
    "llama2_7b": HistoricalModel(
        name="LLaMA 2 7B",
        release_date=date(2023, 7, 18),
        training_flops=8.4e22,  # 6 * 7e9 * 2e12 tokens
        parameters=7,
        notes="Meta, 7B params, 2T tokens, MMLU ~45%",
    ),
    "mistral_7b": HistoricalModel(
        name="Mistral 7B",
        release_date=date(2023, 9, 27),
        training_flops=8.4e22,  # Estimated: 6 * 7e9 * 2e12 tokens
        parameters=7,
        notes="Mistral AI, 7B params, compute not disclosed",
    ),
    "phi2": HistoricalModel(
        name="Phi-2",
        release_date=date(2023, 12, 12),
        training_flops=1.0e22,  # Estimated ~2.7B params, ~1.4T tokens
        parameters=2.7,
        notes="Microsoft, 2.7B params, textbook-quality data",
    ),
    # --- 2024 Era ---
    "claude3_opus": HistoricalModel(
        name="Claude 3 Opus",
        release_date=date(2024, 3, 4),
        training_flops=1.6e25,  # Epoch AI estimate
        parameters=None,  # Not disclosed
        notes="Anthropic, GPT-4 class, Epoch AI imputed compute",
    ),
    "llama3_70b": HistoricalModel(
        name="LLaMA 3 70B",
        release_date=date(2024, 4, 18),
        training_flops=9.3e24,  # ~6.4M GPU hours estimate
        parameters=70,
        notes="Meta, 70B params, 15T tokens",
    ),
    "qwen2_72b": HistoricalModel(
        name="Qwen 2.5 72B",
        release_date=date(2024, 9, 19),
        training_flops=7.8e24,  # Epoch AI estimate
        parameters=72,
        notes="Alibaba, 72B params",
    ),
    "llama3_8b": HistoricalModel(
        name="LLaMA 3 8B",
        release_date=date(2024, 4, 18),
        training_flops=7.2e23,  # 6 * 8e9 * 15e12 tokens
        parameters=8,
        notes="Meta, 8B params, 15T tokens",
    ),
    "mistral_nemo": HistoricalModel(
        name="Mistral Nemo 12B",
        release_date=date(2024, 7, 18),
        training_flops=1.4e23,  # Estimated
        parameters=12,
        notes="Mistral AI / Nvidia, 12B params",
    ),
    # --- Early LLM → Modern SLM Comparison Models ---
    "gemma_2b": HistoricalModel(
        name="Gemma 2B",
        release_date=date(2024, 2, 21),
        training_flops=2.4e22,  # 6 * 2e9 * 2e12 tokens
        parameters=2,
        notes="Google, 2B params, 2T tokens, HellaSwag ~70%",
    ),
    "gemma_7b": HistoricalModel(
        name="Gemma 7B",
        release_date=date(2024, 2, 21),
        training_flops=2.5e23,  # 6 * 7e9 * 6e12 tokens
        parameters=7,
        notes="Google, 7B params, 6T tokens",
    ),
    "phi3_mini": HistoricalModel(
        name="Phi-3 Mini",
        release_date=date(2024, 4, 23),
        training_flops=1.1e23,  # 6 * 3.8e9 * 4.9e12 tokens
        parameters=3.8,
        notes="Microsoft, 3.8B params, 4.9T tokens, MMLU 69%",
    ),
    "gemma2_9b": HistoricalModel(
        name="Gemma 2 9B",
        release_date=date(2024, 6, 27),
        training_flops=4.3e23,  # 6 * 9e9 * 8e12 tokens
        parameters=9,
        notes="Google, 9B params, 8T tokens, HellaSwag ~82%",
    ),
}


# =============================================================================
# Capability-Matching Pairs
# =============================================================================

CAPABILITY_MATCHES: List[CapabilityMatch] = [
    # --- GPT-3 class being matched by smaller models ---
    # NOTE: Chinchilla pair EXCLUDED - it used MORE compute than GPT-3 (3x), 
    # testing data efficiency not compute efficiency
    CapabilityMatch(
        reference=MODELS["gpt3"],
        matching=MODELS["llama1_13b"],
        benchmark="HellaSwag",
        notes="LLaMA-13B matched/exceeded GPT-3 on HellaSwag with ~3x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["gpt3"],
        matching=MODELS["mistral_7b"],
        benchmark="Summarization tasks",
        notes="Mistral 7B matched GPT-3.5 Turbo on news summarization (GPT-3 class)",
    ),
    CapabilityMatch(
        reference=MODELS["gpt3"],
        matching=MODELS["phi2"],
        benchmark="Reasoning benchmarks",
        notes="Phi-2 2.7B matched GPT-3.5 on some reasoning benchmarks",
    ),
    # --- Within-family improvements ---
    # NOTE: LLaMA 1 65B → LLaMA 2 70B EXCLUDED - LLaMA 2 used MORE compute (1.5x)
    # NOTE: LLaMA 2 70B → LLaMA 3 8B EXCLUDED - similar compute (~0.86x, not 12x less)
    CapabilityMatch(
        reference=MODELS["llama2_70b"],
        matching=MODELS["mistral_nemo"],
        benchmark="General benchmarks",
        notes="Mistral Nemo 12B competitive with LLaMA 2 70B (~6x less compute)",
    ),
    # --- GPT-4 class comparisons ---
    # NOTE: Claude 3 Opus uses similar compute to GPT-4, not a good efficiency test
    CapabilityMatch(
        reference=MODELS["gpt4"],
        matching=MODELS["llama3_70b"],
        benchmark="Coding/MMLU",
        notes="LLaMA 3 70B approaches GPT-4 on many benchmarks with ~2x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["gpt4"],
        matching=MODELS["qwen2_72b"],
        benchmark="MMLU/Math",
        notes="Qwen 2.5 72B competitive with GPT-4 on many benchmarks (~3x less compute)",
    ),
    # --- Cross-era validated capability matches (wide time gaps) ---
    # NOTE: Gopher→Chinchilla EXCLUDED - Chinchilla used 1.6x MORE compute, tests data efficiency not compute efficiency
    CapabilityMatch(
        reference=MODELS["llama1_33b"],
        matching=MODELS["mistral_7b"],
        benchmark="HellaSwag",
        notes="Mistral 7B (~83%) matches LLaMA 33B (~83%) with ~3x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["llama1_65b"],
        matching=MODELS["falcon_40b"],
        benchmark="HellaSwag",
        notes="Falcon 40B (~85%) matches LLaMA 65B (~84%) with ~2x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["llama1_65b"],
        matching=MODELS["llama2_13b"],
        benchmark="MMLU",
        notes="LLaMA 2 13B (~55% MMLU) approaches LLaMA 1 65B (~64% MMLU) with ~3.5x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["gpt3"],
        matching=MODELS["llama2_13b"],
        benchmark="MMLU",
        notes="LLaMA 2 13B (~55% MMLU) exceeds GPT-3 (~44% MMLU) with ~2x less compute",
    ),
    CapabilityMatch(
        reference=MODELS["llama1_13b"],
        matching=MODELS["llama2_7b"],
        benchmark="MMLU",
        notes="LLaMA 2 7B (~45% MMLU) matches LLaMA 1 13B (~47% MMLU) with ~1.3x less compute",
    ),
]


def get_valid_efficiency_matches() -> List[CapabilityMatch]:
    """Return only matches where the matching model uses LESS compute (valid for efficiency testing)."""
    return [m for m in CAPABILITY_MATCHES if m.compute_ratio < 1.0]


def get_all_models() -> List[HistoricalModel]:
    """Return all historical models sorted by release date."""
    return sorted(MODELS.values(), key=lambda m: m.release_date)


def get_all_matches() -> List[CapabilityMatch]:
    """Return all capability-matching pairs."""
    return CAPABILITY_MATCHES


def find_best_tau(matches: Optional[List[CapabilityMatch]] = None) -> Tuple[float, float]:
    """
    Find the tau value that minimizes total squared prediction error.
    
    Returns (best_tau, mean_squared_error).
    """
    if matches is None:
        matches = CAPABILITY_MATCHES
    
    if not matches:
        return (8.0, float('nan'))
    
    best_tau = 8.0
    best_mse = float('inf')
    
    # Search over tau values from 1 to 36 months
    for tau_candidate in [t / 10.0 for t in range(10, 361)]:
        mse = sum(m.prediction_error(tau_candidate) ** 2 for m in matches) / len(matches)
        if mse < best_mse:
            best_mse = mse
            best_tau = tau_candidate
    
    return (best_tau, best_mse)
