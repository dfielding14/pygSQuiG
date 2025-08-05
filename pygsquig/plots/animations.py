"""Animation functions for pygSQuiG simulations."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation

from .style import PlotStyle


def create_field_animation(
    field_files: list[Path],
    field_name: str = "theta",
    output_path: Optional[Path] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: Optional[tuple[float, float]] = None,
) -> FuncAnimation:
    """Create an animation from a series of field files.

    Args:
        field_files: List of paths to field files
        field_name: Name of field to animate
        output_path: Path to save animation (mp4)
        fps: Frames per second
        dpi: DPI for animation
        figsize: Figure size

    Returns:
        FuncAnimation object
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    # Load first file to get grid info
    ds0 = xr.open_dataset(field_files[0])
    N = ds0.attrs["N"]
    ds0.attrs["L"]
    x = ds0.x.values
    y = ds0.y.values

    # Determine color scale from all files
    vmax = 0
    for f in field_files[:10]:  # Sample first 10 files
        ds = xr.open_dataset(f)
        field = ds[field_name].values
        vmax = max(vmax, np.max(np.abs(field)))
        ds.close()

    ds0.close()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initial plot
    im = ax.pcolormesh(
        x, y, np.zeros((N, N)), cmap=PlotStyle.FIELD_CMAP, vmin=-vmax, vmax=vmax, shading="gouraud"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    title = ax.set_title("")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(field_name, rotation=270, labelpad=20)

    def update(frame):
        """Update function for animation."""
        ds = xr.open_dataset(field_files[frame])
        field = ds[field_name].values
        time = float(ds.time.values)
        ds.close()

        im.set_array(field.ravel())
        title.set_text(f"{field_name} (t={time:.2f})")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(field_files), interval=1000 / fps, blit=True)

    if output_path:
        anim.save(output_path, fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264"])
        plt.close(fig)

    return anim


def create_vorticity_animation(
    field_files: list[Path],
    alpha: float,
    output_path: Optional[Path] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: Optional[tuple[float, float]] = None,
) -> FuncAnimation:
    """Create an animation of vorticity evolution.

    Args:
        field_files: List of paths to field files containing theta_hat
        alpha: Fractional exponent for vorticity
        output_path: Path to save animation (mp4)
        fps: Frames per second
        dpi: DPI for animation
        figsize: Figure size

    Returns:
        FuncAnimation object
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    # Import needed functions
    from ..core.grid import ifft2, make_grid
    from ..core.operators import fractional_laplacian

    # Load first file to get grid info
    ds0 = xr.open_dataset(field_files[0])
    N = ds0.attrs["N"]
    L = ds0.attrs["L"]
    grid = make_grid(N, L)

    # Determine color scale
    vmax = 0
    for f in field_files[:10]:
        ds = xr.open_dataset(f)
        theta_hat = ds["theta_hat"].values
        q_hat = fractional_laplacian(theta_hat, grid, alpha)
        q = ifft2(q_hat)
        vmax = max(vmax, np.max(np.abs(q)))
        ds.close()

    ds0.close()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initial plot
    # Use 1D arrays if grid has 2D arrays (for compatibility)
    if grid.x.ndim == 2:
        x_plot = grid.x[0, :]
        y_plot = grid.y[:, 0]
    else:
        x_plot = grid.x
        y_plot = grid.y

    im = ax.pcolormesh(
        x_plot,
        y_plot,
        np.zeros((N, N)),
        cmap=PlotStyle.VORTICITY_CMAP,
        vmin=-vmax,
        vmax=vmax,
        shading="gouraud",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    title = ax.set_title("")

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Vorticity q", rotation=270, labelpad=20)

    def update(frame):
        """Update function for animation."""
        ds = xr.open_dataset(field_files[frame])
        theta_hat = ds["theta_hat"].values
        time = float(ds.time.values)
        ds.close()

        # Compute vorticity
        q_hat = fractional_laplacian(theta_hat, grid, alpha)
        q = ifft2(q_hat)

        im.set_array(q.ravel())
        title.set_text(f"Vorticity (t={time:.2f}, α={alpha})")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(field_files), interval=1000 / fps, blit=True)

    if output_path:
        anim.save(output_path, fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264"])
        plt.close(fig)

    return anim


def create_spectrum_animation(
    field_files: list[Path],
    alpha: float,
    output_path: Optional[Path] = None,
    fps: int = 5,
    dpi: int = 100,
    figsize: Optional[tuple[float, float]] = None,
) -> FuncAnimation:
    """Create an animation of spectrum evolution.

    Args:
        field_files: List of paths to field files
        alpha: Fractional exponent
        output_path: Path to save animation (mp4)
        fps: Frames per second
        dpi: DPI for animation
        figsize: Figure size

    Returns:
        FuncAnimation object
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE

    # Import needed functions
    from ..core.grid import make_grid
    from ..utils.diagnostics import compute_energy_spectrum

    # Load first file to get grid info
    ds0 = xr.open_dataset(field_files[0])
    N = ds0.attrs["N"]
    L = ds0.attrs["L"]
    grid = make_grid(N, L)
    ds0.close()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Initial plot
    (line,) = ax.loglog([], [], PlotStyle.SPECTRUM_COLOR, linewidth=2.5)

    # Reference slope
    k_ref = np.logspace(1, 2, 50)
    E_ref = 1e-3 * k_ref ** (-5 / 3)
    ax.loglog(k_ref, E_ref, "r--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy Spectrum E(k)")
    ax.set_xlim(1, N / 2)
    ax.set_ylim(1e-10, 1e0)
    ax.grid(True, alpha=0.3, which="both")
    title = ax.set_title("")

    def update(frame):
        """Update function for animation."""
        ds = xr.open_dataset(field_files[frame])
        theta_hat = ds["theta_hat"].values
        time = float(ds.time.values)
        ds.close()

        # Compute spectrum
        k, E_k = compute_energy_spectrum(theta_hat, grid, alpha)

        line.set_data(k[1:], E_k[1:])
        title.set_text(f"Energy Spectrum (t={time:.2f}, α={alpha})")
        return [line, title]

    anim = FuncAnimation(fig, update, frames=len(field_files), interval=1000 / fps, blit=True)

    if output_path:
        anim.save(output_path, fps=fps, dpi=dpi, extra_args=["-vcodec", "libx264"])
        plt.close(fig)

    return anim
