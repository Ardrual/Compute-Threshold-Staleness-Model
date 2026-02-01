"""Generate staleness over time figure for τ=8 months."""

import numpy as np
import matplotlib.pyplot as plt
import math


def staleness_at_time(t: float, tau: float, update_interval: float) -> float:
    """Staleness within an update cycle: S(t) = 2^((t - t_n) / τ)"""
    t_n = math.floor(t / update_interval) * update_interval
    return 2 ** ((t - t_n) / tau)


def main():
    # Parameters
    tau = 8.0  # efficiency doubling time (months)
    update_interval = 12.0  # policy update interval (months)
    max_time = 36  # time horizon (months)
    
    # Generate data
    times = np.linspace(0, max_time, 500)
    staleness = [staleness_at_time(t, tau, update_interval) for t in times]
    
    # Calculate max staleness
    s_max = 2 ** (update_interval / tau)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Main staleness curve
    ax.plot(times, staleness, 'b-', linewidth=2, label=f'τ = {tau:.0f} mo, U = {update_interval:.0f} mo')
    
    # Reference line at 2× stale
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='2× stale')
    
    # Mark update points
    for update_time in range(0, max_time + 1, int(update_interval)):
        ax.axvline(x=update_time, color='gray', linestyle=':', alpha=0.5)
    
    # Annotate max staleness
    ax.annotate(f'$S_{{max}} = 2^{{U/τ}} ≈ {s_max:.2f}×$', 
                xy=(11.5, s_max), xytext=(15, s_max + 0.3),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    ax.set_xlabel('Time (months)', fontsize=12)
    ax.set_ylabel('Staleness (T / T*)', fontsize=12)
    ax.set_title('Policy Threshold Staleness Over Time\n(τ = 8 months, annual updates)', fontsize=12)
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_time)
    ax.set_ylim(0.9, 3.2)
    ax.grid(True, alpha=0.3)
    
    # Save figure
    output_path = 'figures/staleness_tau8.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    
    # Also save PDF for LaTeX
    plt.savefig('figures/staleness_tau8.pdf', bbox_inches='tight')
    print("Saved PDF version for LaTeX")


if __name__ == "__main__":
    main()
