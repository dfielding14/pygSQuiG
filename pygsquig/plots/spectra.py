"""Spectrum analysis plotting functions for pygSQuiG simulations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict

from ..core.grid import Grid
from ..utils.diagnostics import compute_energy_spectrum
from .style import PlotStyle


def plot_energy_spectrum_with_analysis(
    theta_hat: np.ndarray,
    grid: Grid,
    alpha: float,
    reference_slopes: Optional[Dict[str, float]] = None,
    inertial_range: Optional[Tuple[float, float]] = None,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[plt.Figure], float]:
    """Plot energy spectrum with reference slopes and inertial range analysis.

    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        reference_slopes: Dict of label->slope for reference lines
        inertial_range: Tuple of (k_min, k_max) for inertial range
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Tuple of (Figure or None, measured slope in inertial range)
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    if reference_slopes is None:
        reference_slopes = {r"$k^{-5/3}$": -5 / 3, r"$k^{-3}$": -3}

    # Compute spectrum
    k, E_k = compute_energy_spectrum(theta_hat, grid, alpha)

    fig, ax = plt.subplots(figsize=figsize)

    # Main spectrum
    ax.loglog(k[1:], E_k[1:], PlotStyle.SPECTRUM_COLOR, linewidth=2.5, label="E(k)")

    # Reference slopes
    # Make sure we have enough points for reference slopes
    if len(k) > 20:
        k_ref_start = max(10, int(len(k) * 0.2))
        k_ref_end = min(len(k) - 10, int(len(k) * 0.6))
        k_ref = k[k_ref_start:k_ref_end]

        if len(k_ref) > 10:
            for i, (label, slope) in enumerate(reference_slopes.items()):
                # Use a point in the middle of the reference range
                ref_idx = min(10, len(k_ref) // 2)
                E_ref = E_k[k_ref_start + ref_idx] * (k_ref / k_ref[ref_idx]) ** slope
                ax.loglog(k_ref, E_ref, "--", linewidth=1.5, label=label, alpha=0.7)

    # Fit and show inertial range if specified
    measured_slope = None
    if inertial_range:
        k_min, k_max = inertial_range
        mask = (k >= k_min) & (k <= k_max)
        k_inertial = k[mask]
        E_inertial = E_k[mask]

        if len(k_inertial) > 2:
            # Fit power law
            log_k = np.log(k_inertial)
            log_E = np.log(E_inertial + 1e-20)
            measured_slope, intercept = np.polyfit(log_k, log_E, 1)

            # Plot fit
            E_fit = np.exp(intercept) * k_inertial**measured_slope
            ax.loglog(
                k_inertial, E_fit, "k:", linewidth=2, label=f"Fit: $k^{{{measured_slope:.2f}}}$"
            )

            # Shade inertial range
            ax.axvspan(k_min, k_max, alpha=0.1, color="gray", label="Inertial range")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy Spectrum E(k)")
    ax.set_title(f"Energy Spectrum (α={alpha})")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None, measured_slope

    return fig, measured_slope


def plot_spectrum_evolution(
    spectra_data: Dict[float, Tuple[np.ndarray, np.ndarray]],
    alpha: float,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    """Plot evolution of energy spectra at different times.

    Args:
        spectra_data: Dict of {time: (k, E_k)} pairs
        alpha: Fractional exponent
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    fig, ax = plt.subplots(figsize=figsize)

    # Sort times
    times = sorted(spectra_data.keys())

    # Create colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(times)))

    for i, t in enumerate(times):
        k, E_k = spectra_data[t]
        ax.loglog(k[1:], E_k[1:], color=colors[i], linewidth=2, label=f"t={t:.1f}")

    # Add reference slope
    k_ref = k[20:60]
    E_ref = E_k[30] * (k_ref / k_ref[10]) ** (-5 / 3)
    ax.loglog(k_ref, E_ref, "k--", alpha=0.5, linewidth=1.5, label=r"$k^{-5/3}$")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy Spectrum E(k)")
    ax.set_title(f"Spectrum Evolution (α={alpha})")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def plot_compensated_spectrum(
    theta_hat: np.ndarray,
    grid: Grid,
    alpha: float,
    compensation_exponent: float = 5 / 3,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    """Plot compensated energy spectrum E(k) * k^(compensation_exponent).

    This helps visualize deviations from a power law.

    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        compensation_exponent: Exponent to compensate by (default 5/3)
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    # Compute spectrum
    k, E_k = compute_energy_spectrum(theta_hat, grid, alpha)

    # Compensate
    E_comp = E_k * k**compensation_exponent

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(k[1:], E_comp[1:], PlotStyle.SPECTRUM_COLOR, linewidth=2.5)

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel(f"$E(k) \\cdot k^{{{compensation_exponent:.2f}}}$")
    ax.set_title(f"Compensated Spectrum (α={alpha})")
    ax.grid(True, alpha=0.3, which="both")

    # Add horizontal line at expected plateau level
    ax.axhline(
        np.median(E_comp[20:60]), color="r", linestyle="--", alpha=0.5, label="Expected plateau"
    )
    ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig
