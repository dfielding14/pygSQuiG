"""Field visualization functions for pygSQuiG simulations."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..core.grid import Grid, ifft2
from ..core.operators import compute_velocity_from_theta, fractional_laplacian
from .style import PlotStyle


def plot_field_slice(
    field: np.ndarray,
    grid: Grid,
    title: str = "Field",
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[tuple[float, float]] = None,
    output_path: Optional[Path] = None,
    show_colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plot a 2D field slice with customizable options.

    Args:
        field: 2D array of field values
        grid: Grid object
        title: Plot title
        cmap: Colormap name (default: RdBu_r)
        vmin, vmax: Color scale limits (default: symmetric around 0)
        figsize: Figure size
        output_path: Path to save figure
        show_colorbar: Whether to show colorbar
        ax: Existing axes to plot on (creates new figure if None)

    Returns:
        Figure object if ax is None, otherwise None
    """
    if cmap is None:
        cmap = PlotStyle.FIELD_CMAP

    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return_fig = True
    else:
        fig = ax.get_figure()
        return_fig = False

    # Determine color scale
    if vmin is None and vmax is None:
        vmax = np.max(np.abs(field))
        vmin = -vmax

    # Plot field
    # Use 1D arrays if grid has 2D arrays (for compatibility)
    if grid.x.ndim == 2:
        x = grid.x[0, :]
        y = grid.y[:, 0]
    else:
        x = grid.x
        y = grid.y

    im = ax.pcolormesh(x, y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(title, rotation=270, labelpad=20)

    if output_path and return_fig:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig if return_fig else None


def plot_vorticity(
    theta_hat: np.ndarray,
    grid: Grid,
    alpha: float,
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Optional[plt.Figure]:
    """Plot the generalized vorticity field q = (-Δ)^(α/2)θ.

    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        title: Plot title (default: "Vorticity q = (-Δ)^(α/2)θ")
        output_path: Path to save figure
        **kwargs: Additional arguments passed to plot_field_slice

    Returns:
        Figure object or None if saved
    """
    # Compute vorticity
    q_hat = fractional_laplacian(theta_hat, grid, alpha)
    q = ifft2(q_hat)

    if title is None:
        title = f"Vorticity q = (-Δ)^({alpha/2:.1f})θ"

    # Use vorticity colormap by default
    if "cmap" not in kwargs:
        kwargs["cmap"] = PlotStyle.VORTICITY_CMAP

    return plot_field_slice(q, grid, title, output_path=output_path, **kwargs)


def plot_velocity_fields(
    theta_hat: np.ndarray,
    grid: Grid,
    alpha: float,
    output_path: Optional[Path] = None,
    figsize: Optional[tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    """Plot velocity components u and v side by side.

    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        output_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_DOUBLE

    # Compute velocity
    u, v = compute_velocity_from_theta(theta_hat, grid, alpha)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot u
    plot_field_slice(u, grid, "u velocity", ax=ax1)

    # Plot v
    plot_field_slice(v, grid, "v velocity", ax=ax2)

    fig.suptitle(f"Velocity Field (α={alpha})")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig


def plot_diagnostic_summary(
    theta_hat: np.ndarray,
    grid: Grid,
    alpha: float,
    time: float,
    forcing_info: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Create a comprehensive diagnostic summary plot.

    Shows theta field, vorticity, spectrum, and key quantities.

    Args:
        theta_hat: Fourier coefficients of theta
        grid: Grid object
        alpha: Fractional exponent
        time: Current simulation time
        forcing_info: Optional dict with 'kf' and 'epsilon' keys
        output_path: Path to save figure

    Returns:
        Figure object or None if saved
    """
    # Import needed functions locally to avoid circular imports
    from ..utils.diagnostics import (
        compute_energy_spectrum,
        compute_enstrophy,
        compute_palinstrophy,
        compute_total_energy,
    )

    fig = plt.figure(figsize=PlotStyle.FIGSIZE_GRID)

    # Create grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Theta field
    ax1 = fig.add_subplot(gs[0, 0])
    theta = ifft2(theta_hat)
    plot_field_slice(theta, grid, "θ", ax=ax1, show_colorbar=False)

    # Vorticity
    ax2 = fig.add_subplot(gs[0, 1])
    q_hat = fractional_laplacian(theta_hat, grid, alpha)
    q = ifft2(q_hat)
    plot_field_slice(q, grid, "q", ax=ax2, show_colorbar=False, cmap=PlotStyle.VORTICITY_CMAP)

    # Velocity magnitude
    ax3 = fig.add_subplot(gs[0, 2])
    u, v = compute_velocity_from_theta(theta_hat, grid, alpha)
    speed = np.sqrt(u**2 + v**2)
    plot_field_slice(
        speed, grid, "|u|", ax=ax3, show_colorbar=False, cmap="viridis", vmin=0, vmax=None
    )

    # Energy spectrum
    ax4 = fig.add_subplot(gs[1, :])
    k, E_k = compute_energy_spectrum(theta_hat, grid, alpha)
    ax4.loglog(k[1:], E_k[1:], PlotStyle.SPECTRUM_COLOR, linewidth=2)

    # Add reference slopes
    if len(k) > 60:
        k_ref = k[20:60]
        E_ref = E_k[30] * (k_ref / k_ref[0]) ** (-5 / 3)
        ax4.loglog(k_ref, E_ref, "r--", alpha=0.5, label=r"$k^{-5/3}$")

    if forcing_info and "kf" in forcing_info:
        ax4.axvline(forcing_info["kf"], color="green", linestyle=":", alpha=0.7, label="Forcing")

    ax4.set_xlabel("k")
    ax4.set_ylabel("E(k)")
    ax4.set_title("Energy Spectrum")
    ax4.legend()
    ax4.grid(True, alpha=0.3, which="both")

    # Diagnostic quantities text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    # Compute diagnostics
    energy = compute_total_energy(theta_hat, grid, alpha)
    enstrophy = compute_enstrophy(theta_hat, grid, alpha)
    palinstrophy = compute_palinstrophy(theta_hat, grid, alpha)

    # Create text
    text_lines = [
        f"Time: t = {time:.2f}",
        f"Energy: E = {float(energy):.4e}",
        f"Enstrophy: Ω = {float(enstrophy):.4e}",
        f"Palinstrophy: P = {float(palinstrophy):.4e}",
    ]

    if forcing_info:
        text_lines.extend(
            [
                f"Forcing: kf = {forcing_info.get('kf', 'N/A')}, "
                f"ε = {forcing_info.get('epsilon', 'N/A')}"
            ]
        )

    text = "\n".join(text_lines)
    ax5.text(
        0.5,
        0.5,
        text,
        transform=ax5.transAxes,
        fontsize=14,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(f"pygSQuiG Diagnostic Summary (α={alpha})", fontsize=16)

    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig
