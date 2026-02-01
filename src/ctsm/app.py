import sys
from pathlib import Path

# Add the project root/src to sys.path for robust imports
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ctsm.model import DriftModel
from ctsm.historical_data import get_all_models, get_all_matches, months_from_baseline, BASELINE_DATE
from ctsm.backtest import run_backtest, tau_sensitivity_analysis
import math

st.set_page_config(
    page_title="CTSM Model Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Compute Threshold Staleness Model (CTSM)")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📐 Closed-Form Analysis", "📊 Historical Backtest", "🧮 Calculators"])

# --- Sidebar: Core Parameters ---
st.sidebar.header("Core Parameters")

# Tau presets based on literature
# Note: Scher 2025 measures catch-up progress (smaller models replicating frontier capabilities),
# not frontier progress. We use his 16×–60× range, not his wider 80% CI of [2×–200×].
TAU_PRESETS = {
    "Ho et al. 2024 (3×/yr, frontier)": 8.0,
    "Backtest Best Fit (~2.5×/yr)": 9.5,
    "Scher 2025 (16–60×/yr, catch-up)": 2.5,  # midpoint of τ = 2.0–2.9 months
    "Custom": None,
}

tau_preset = st.sidebar.selectbox(
    "τ Preset (Efficiency Doubling)",
    options=list(TAU_PRESETS.keys()),
    index=0,
    help="Select a preset based on literature estimates, or choose Custom for manual input.",
)

if tau_preset == "Custom":
    tau = st.sidebar.slider(
        "Custom τ [months]", 
        min_value=1.0, 
        max_value=36.0, 
        value=8.0,
        help="Time for algorithmic efficiency to double.",
    )
else:
    tau = TAU_PRESETS[tau_preset]
    st.sidebar.caption(f"τ = {tau:.1f} months")

# Tau uncertainty range for sensitivity
st.sidebar.divider()
st.sidebar.header("Uncertainty Range")
# Default bounds: tau/2 to tau*2, clamped to reasonable range
default_tau_low = max(1.0, tau / 2)
default_tau_high = min(48.0, tau * 2)
tau_low = st.sidebar.number_input("τ lower bound [months]", value=default_tau_low, min_value=1.0, max_value=tau)
tau_high = st.sidebar.number_input("τ upper bound [months]", value=default_tau_high, min_value=tau, max_value=48.0)

st.sidebar.divider()
st.sidebar.header("Policy Settings")
update_interval = st.sidebar.number_input(
    "Update Interval U [months]", 
    min_value=1.0, 
    value=12.0,
    help="How often the policy threshold is updated."
)

max_time = st.sidebar.slider("Time Horizon [months]", min_value=12, max_value=72, value=36)


# --- Helper Functions ---
def staleness_at_time(t: float, tau: float, update_interval: float) -> float:
    """Staleness within an update cycle: S(t) = 2^((t - t_n) / τ)"""
    t_n = math.floor(t / update_interval) * update_interval
    return 2 ** ((t - t_n) / tau)

def max_staleness(tau: float, update_interval: float) -> float:
    """Maximum staleness before update: S_max = 2^(U/τ)"""
    return 2 ** (update_interval / tau)

def required_update_interval(tau: float, max_acceptable_staleness: float) -> float:
    """Update interval needed to stay under target staleness: U = τ × log₂(S_max)"""
    return tau * math.log2(max_acceptable_staleness)

def catchup_time(tau: float, compute_reduction_factor: float) -> float:
    """Time for capability to become achievable at 1/k compute: t = τ × log₂(k)"""
    return tau * math.log2(compute_reduction_factor)

def ideal_threshold_ratio(t: float, tau: float) -> float:
    """How much the ideal threshold has decayed: T*(t)/T*(0) = 1/A(t) = 2^(-t/τ)"""
    return 2 ** (-t / tau)


# --- Tab 1: Closed-Form Analysis ---
with tab1:
    st.markdown("""
    Core closed-form relationships showing how thresholds decay and staleness accumulates.
    These relationships hold regardless of compute distribution assumptions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Staleness Over Time")
        st.markdown(f"**Formula:** `S(t) = 2^((t - tₙ) / τ)` within each update cycle")
        
        times = np.linspace(0, max_time, 500)
        
        # Main staleness curve
        staleness_main = [staleness_at_time(t, tau, update_interval) for t in times]
        
        # Uncertainty band
        staleness_low = [staleness_at_time(t, tau_high, update_interval) for t in times]  # high tau = lower staleness
        staleness_high = [staleness_at_time(t, tau_low, update_interval) for t in times]  # low tau = higher staleness
        
        fig_staleness = go.Figure()
        
        # Uncertainty band
        fig_staleness.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([staleness_high, staleness_low[::-1]]),
            fill='toself',
            fillcolor='rgba(99, 110, 250, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'τ ∈ [{tau_low:.0f}, {tau_high:.0f}] months',
            hoverinfo='skip',
        ))
        
        # Main line
        fig_staleness.add_trace(go.Scatter(
            x=times, y=staleness_main, 
            mode='lines', 
            name=f'τ = {tau:.1f} mo', 
            line=dict(color='#636EFA', width=3)
        ))
        
        # Reference line at staleness = 2
        fig_staleness.add_hline(y=2, line_dash="dash", line_color="red", 
                                annotation_text="2× stale", annotation_position="right")
        
        fig_staleness.update_layout(
            xaxis_title="Time (months)",
            yaxis_title="Staleness (T/T*)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
        )
        st.plotly_chart(fig_staleness, width='stretch')
        
        # Key metrics
        s_max = max_staleness(tau, update_interval)
        st.metric("Max Staleness (end of cycle)", f"{s_max:.2f}×")

    with col2:
        st.subheader("Ideal Threshold Decay")
        st.markdown(f"**Formula:** `T*(t) = E_risk / A(t)` — threshold halves every τ months")
        
        # Threshold decay (normalized to T*(0) = 1)
        threshold_main = [ideal_threshold_ratio(t, tau) for t in times]
        threshold_low = [ideal_threshold_ratio(t, tau_low) for t in times]  # low tau = faster decay
        threshold_high = [ideal_threshold_ratio(t, tau_high) for t in times]  # high tau = slower decay
        
        fig_threshold = go.Figure()
        
        # Uncertainty band
        fig_threshold.add_trace(go.Scatter(
            x=np.concatenate([times, times[::-1]]),
            y=np.concatenate([threshold_high, threshold_low[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 204, 150, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'τ ∈ [{tau_low:.0f}, {tau_high:.0f}] months',
            hoverinfo='skip',
        ))
        
        fig_threshold.add_trace(go.Scatter(
            x=times, y=threshold_main, 
            mode='lines', 
            name=f'τ = {tau:.1f} mo', 
            line=dict(color='#00CC96', width=3)
        ))
        
        fig_threshold.add_hline(y=0.5, line_dash="dash", line_color="orange")
        fig_threshold.add_annotation(
            x=1, xref="paper", y=math.log10(0.5), yref="y",
            text="50% (1 doubling)", showarrow=False,
            xanchor="left", yanchor="middle", font=dict(color="orange")
        )
        fig_threshold.add_hline(y=0.1, line_dash="dash", line_color="red")
        fig_threshold.add_annotation(
            x=1, xref="paper", y=math.log10(0.1), yref="y",
            text="10% (>3 doublings)", showarrow=False,
            xanchor="left", yanchor="middle", font=dict(color="red")
        )
        
        fig_threshold.update_layout(
            xaxis_title="Time (months)",
            yaxis_title="T*(t) / T*(0)",
            yaxis_type="log",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            height=400,
        )
        st.plotly_chart(fig_threshold, width='stretch')
        
        # Key metric
        threshold_at_end = ideal_threshold_ratio(max_time, tau)
        st.metric(f"Threshold at t={max_time} mo", f"{threshold_at_end:.1%} of original")

    st.divider()
    
    # Update Interval Comparison
    st.subheader("Update Interval Comparison")
    st.markdown("How does max staleness change with different update intervals?")
    
    intervals = [3, 6, 12, 18, 24, 36]
    comparison_data = []
    for u in intervals:
        s_main = max_staleness(tau, u)
        s_low = max_staleness(tau_high, u)
        s_high = max_staleness(tau_low, u)
        comparison_data.append({
            "Update Interval": f"{u} months",
            f"Max Staleness (τ={tau:.0f})": f"{s_main:.2f}×",
            f"Range (τ∈[{tau_low:.0f},{tau_high:.0f}])": f"{s_low:.2f}× – {s_high:.2f}×",
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, width='stretch', hide_index=True)
    
    st.info("""
    **Key Insight:** Staleness is exponential in U/τ. Doubling the update interval doesn't double staleness—it *squares* the staleness factor.
    For τ = 8 months: U=12 gives 2.8× stale, U=24 gives 8× stale.
    """)


# --- Tab 2: Historical Backtest ---
with tab2:
    st.warning("""
    ⚠️ **Sanity Check Only** — This backtest is a rough plausibility check, not a replacement for 
    rigorous empirical research. The capability-matching pairs are approximate, benchmarks measure 
    different aspects of capability, and training compute estimates have significant uncertainty.
    """)
    
    st.markdown("""
    Tests whether the model correctly predicts **when smaller models would reach capability parity** 
    with earlier larger models. Provides a rough sanity check on τ estimates.
    """)
    
    # Run backtest with current tau
    backtest_result = run_backtest(tau)
    
    # Summary metrics
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Current τ", f"{tau:.1f} mo")
    with col_b:
        st.metric("Best-fit τ", f"{backtest_result.best_fit_tau:.1f} mo")
    with col_c:
        st.metric("RMSE", f"{backtest_result.rmse:.1f} mo")
    
    st.divider()
    
    # Model pairs analysis
    st.subheader("Capability-Matching Model Pairs")
    
    models = get_all_models()
    matches = get_all_matches()
    
    # Create a detailed table of the analysis
    rows = []
    for a in backtest_result.analyses:
        rows.append({
            "Reference Model": a.match.reference.name,
            "Matching Model": a.match.matching.name,
            "Benchmark": a.match.benchmark,
            "Compute Ratio": f"{a.match.compute_ratio:.2f}×",
            "Actual Gap": f"{a.actual_gap_months:.1f} mo",
            "Predicted Gap": f"{a.predicted_gap_months:.1f} mo",
            "Error": f"{a.error_months:+.1f} mo",
        })
    
    df_analysis = pd.DataFrame(rows)
    st.dataframe(df_analysis, width='stretch', hide_index=True)
    
    st.divider()
    
    # Timeline visualization
    st.subheader("Model Timeline")
    
    fig_timeline = go.Figure()
    
    # Plot all models on a timeline
    model_months = [months_from_baseline(m.release_date) for m in models]
    model_flops = [m.training_flops for m in models]
    model_names = [m.name for m in models]
    
    fig_timeline.add_trace(go.Scatter(
        x=model_months,
        y=model_flops,
        mode='markers+text',
        text=model_names,
        textposition='top center',
        marker=dict(size=12, color='#636EFA'),
        name='Models',
    ))
    
    # Draw lines connecting matched pairs
    for match in matches:
        ref_m = months_from_baseline(match.reference.release_date)
        mat_m = months_from_baseline(match.matching.release_date)
        fig_timeline.add_trace(go.Scatter(
            x=[ref_m, mat_m],
            y=[match.reference.training_flops, match.matching.training_flops],
            mode='lines',
            line=dict(dash='dot', color='rgba(99, 110, 250, 0.4)', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    fig_timeline.update_layout(
        xaxis_title=f"Months from baseline ({BASELINE_DATE.strftime('%b %Y')})",
        yaxis_title="Training FLOPs",
        yaxis_type="log",
        height=450,
        hovermode="closest",
    )
    st.plotly_chart(fig_timeline, width='stretch')
    
    st.divider()
    
    # Tau sensitivity analysis
    st.subheader("τ Sensitivity Analysis")
    st.markdown("How does prediction error (RMSE) vary with different τ values?")
    
    sensitivity = tau_sensitivity_analysis(tau_range=(2.0, 24.0), step=0.5)
    tau_vals = [s[0] for s in sensitivity]
    rmse_vals = [s[1] for s in sensitivity]
    
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=tau_vals, y=rmse_vals, mode='lines', name='RMSE', line=dict(color='#EF553B', width=2)))
    fig_sens.add_vline(x=tau, line_dash="dash", line_color="#636EFA", annotation_text=f"Current τ={tau}")
    fig_sens.add_vline(x=backtest_result.best_fit_tau, line_dash="dot", line_color="#00CC96", annotation_text=f"Best τ={backtest_result.best_fit_tau:.1f}")
    
    # Show uncertainty region
    fig_sens.add_vrect(x0=tau_low, x1=tau_high, fillcolor="rgba(99, 110, 250, 0.1)", 
                       layer="below", line_width=0, annotation_text="Uncertainty range")
    
    fig_sens.update_layout(
        xaxis_title="τ (efficiency doubling time, months)",
        yaxis_title="RMSE (months)",
        height=350,
    )
    st.plotly_chart(fig_sens, width='stretch')
    
    st.warning("""
    **⚠️ Significant Limitations:**
    - This is a **sanity check**, not a rigorous empirical analysis
    - Capability "matching" is approximate — benchmarks measure different aspects of ability
    - Training compute estimates have high uncertainty (often 2-5× error)
    - Different capability dimensions may have different efficiency curves
    - τ likely varies over time rather than being constant
    - For policy decisions, rely on peer-reviewed empirical research (e.g., Ho et al., Epoch AI)
    """)


# --- Tab 3: Calculators --- 
with tab3:
    st.markdown("""
    Direct calculations from the closed-form relationships. No simulation or distribution assumptions.
    """)
    
    col_calc1, col_calc2 = st.columns(2)
    
    with col_calc1:
        st.subheader("🎯 Required Update Interval")
        st.markdown("*Given a target max staleness, what update interval do you need?*")
        
        target_staleness = st.number_input(
            "Target Max Staleness", 
            min_value=1.1, 
            max_value=10.0, 
            value=2.0,
            help="Maximum acceptable staleness before policy update"
        )
        
        u_required = required_update_interval(tau, target_staleness)
        u_required_low = required_update_interval(tau_low, target_staleness)
        u_required_high = required_update_interval(tau_high, target_staleness)
        
        st.markdown(f"""
        **Result:** To keep staleness ≤ {target_staleness}×:
        - With τ = {tau:.1f} mo: Update every **{u_required:.1f} months**
        - Range (τ ∈ [{tau_low:.0f}, {tau_high:.0f}]): **{u_required_low:.1f} – {u_required_high:.1f} months**
        """)
        
        st.divider()
        
        st.subheader("⏱️ Catch-Up Timeline")
        st.markdown("*When does a reference capability become achievable at reduced compute?*")
        
        compute_reduction = st.selectbox(
            "Compute Reduction Factor",
            options=[2, 5, 10, 50, 100, 1000],
            index=2,
            format_func=lambda x: f"{x}× less compute"
        )
        
        t_catchup = catchup_time(tau, compute_reduction)
        t_catchup_low = catchup_time(tau_low, compute_reduction)
        t_catchup_high = catchup_time(tau_high, compute_reduction)
        
        st.markdown(f"""
        **Result:** Capability reachable at 1/{compute_reduction}th compute:
        - With τ = {tau:.1f} mo: In **{t_catchup:.1f} months** ({t_catchup/12:.1f} years)
        - Range (τ ∈ [{tau_low:.0f}, {tau_high:.0f}]): **{t_catchup_low:.1f} – {t_catchup_high:.1f} months**
        """)
    
    with col_calc2:
        st.subheader("📉 Threshold Decay")
        st.markdown("*How much has the ideal threshold decayed at a given time?*")
        
        time_input = st.number_input(
            "Time [months]",
            min_value=0.0,
            max_value=120.0,
            value=24.0,
        )
        
        decay_main = ideal_threshold_ratio(time_input, tau)
        decay_low = ideal_threshold_ratio(time_input, tau_low)  # faster decay
        decay_high = ideal_threshold_ratio(time_input, tau_high)  # slower decay
        
        st.markdown(f"""
        **Result:** At t = {time_input:.0f} months, ideal threshold is:
        - With τ = {tau:.1f} mo: **{decay_main:.1%}** of original ({1/decay_main:.1f}× efficiency improvement)
        - Range: **{decay_low:.1%} – {decay_high:.1%}** of original
        """)
        
        st.divider()
        
        st.subheader("📊 Staleness at Time")
        st.markdown("*Current staleness given update interval and time since last update?*")
        
        time_since_update = st.slider(
            "Time Since Last Update [months]",
            min_value=0.0,
            max_value=float(update_interval),
            value=float(update_interval) / 2,
        )
        
        current_staleness = 2 ** (time_since_update / tau)
        current_staleness_low = 2 ** (time_since_update / tau_high)
        current_staleness_high = 2 ** (time_since_update / tau_low)
        
        st.markdown(f"""
        **Result:** {time_since_update:.1f} months since last update:
        - With τ = {tau:.1f} mo: **{current_staleness:.2f}× stale**
        - Range: **{current_staleness_low:.2f}× – {current_staleness_high:.2f}×**
        """)

    st.divider()
    
    st.subheader("📋 Formula Reference")
    st.markdown("""
    | Quantity | Formula | Interpretation |
    |----------|---------|----------------|
    | Efficiency multiplier | `A(t) = 2^(t/τ)` | Same compute yields A(t)× more capability at time t |
    | Ideal threshold | `T*(t) = E_risk / A(t)` | Threshold needed to catch runs with effective compute ≥ E_risk |
    | Staleness | `S(t) = 2^((t - tₙ)/τ)` | How much too permissive the threshold is |
    | Max staleness | `S_max = 2^(U/τ)` | Staleness just before update |
    | Required interval | `U = τ × log₂(S_target)` | Update frequency to stay under target staleness |
    | Catch-up time | `t = τ × log₂(k)` | Time for capability to become k× cheaper |
    """)
