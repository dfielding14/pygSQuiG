"""Analysis script for pygSQuiG simulation outputs.

This script provides tools for post-processing and visualizing simulation results.
"""

import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import h5py
from typing import Optional, List, Tuple

from pygsquig.io import load_diagnostics, load_output
from pygsquig.utils import get_logger


logger = get_logger("pygsquig.analyse")


def load_time_series(diagnostics_file: Path) -> dict:
    """Load diagnostic time series from HDF5 file."""
    return load_diagnostics(diagnostics_file)


def plot_time_series(
    diagnostics: dict,
    quantities: List[str],
    output_file: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """Plot time series of diagnostic quantities.

    Args:
        diagnostics: Dictionary with time series data
        quantities: List of quantities to plot
        output_file: Optional path to save figure
        figsize: Figure size
    """
    time = diagnostics["time"]
    n_quantities = len(quantities)

    fig, axes = plt.subplots(n_quantities, 1, figsize=figsize, sharex=True)
    if n_quantities == 1:
        axes = [axes]

    for ax, quantity in zip(axes, quantities):
        if quantity in diagnostics:
            ax.plot(time, diagnostics[quantity])
            ax.set_ylabel(quantity)
            ax.grid(True, alpha=0.3)
        else:
            logger.warning(f"Quantity '{quantity}' not found in diagnostics")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Diagnostic Time Series")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved time series plot to {output_file}")
    else:
        # Save to default location instead of showing
        plt.savefig("timeseries.png", dpi=150, bbox_inches="tight")
        logger.info("Saved time series plot to timeseries.png")
    plt.close(fig)


def plot_energy_spectrum(
    field_file: Path,
    output_file: Optional[Path] = None,
    reference_slope: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """Plot energy spectrum from a field snapshot.

    Args:
        field_file: Path to field output file
        output_file: Optional path to save figure
        reference_slope: Optional reference slope (e.g., -5/3)
        figsize: Figure size
    """
    # Load field data
    ds = load_output(field_file)

    if "theta" not in ds:
        logger.error("No theta field found in output file")
        return

    # Get field and grid info
    theta = ds["theta"].values
    N = ds.attrs["N"]
    L = ds.attrs["L"]

    # Compute 2D FFT
    theta_hat = np.fft.fft2(theta)

    # Compute power spectrum
    power = np.abs(theta_hat) ** 2

    # Create wavenumber arrays
    kx = np.fft.fftfreq(N, d=L / (2 * np.pi * N)) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L / (2 * np.pi * N)) * 2 * np.pi
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    # Bin the spectrum
    dk = 2 * np.pi / L
    k_edges = np.arange(0, N / 2 * dk, dk)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    E_k = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k >= k_edges[i]) & (k < k_edges[i + 1])
        E_k[i] = np.sum(power[mask])

    # Normalize
    E_k /= dk

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Main spectrum
    ax.loglog(k_centers, E_k, "b-", linewidth=2, label="E(k)")

    # Reference slope if provided
    if reference_slope:
        k_ref = k_centers[10:-10]  # Avoid edges
        E_ref = E_k[20] * (k_ref / k_ref[10]) ** reference_slope
        ax.loglog(k_ref, E_ref, "k--", alpha=0.5, label=f"k^{{{reference_slope}}}")

    ax.set_xlabel("k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Energy Spectrum (t={ds.time.values:.2f})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved spectrum plot to {output_file}")
    else:
        # Save to default location instead of showing
        plt.savefig("spectrum.png", dpi=150, bbox_inches="tight")
        logger.info("Saved spectrum plot to spectrum.png")
    plt.close(fig)


def plot_field_snapshot(
    field_file: Path,
    field_name: str = "theta",
    output_file: Optional[Path] = None,
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8, 6),
) -> None:
    """Plot a field snapshot.

    Args:
        field_file: Path to field output file
        field_name: Name of field to plot
        output_file: Optional path to save figure
        cmap: Colormap name
        figsize: Figure size
    """
    # Load field data
    ds = load_output(field_file)

    if field_name not in ds:
        logger.error(f"Field '{field_name}' not found in output file")
        logger.info(f"Available fields: {list(ds.data_vars)}")
        return

    # Get field
    field = ds[field_name].values
    x = ds.x.values
    y = ds.y.values

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use symmetric color scale around zero
    vmax = np.max(np.abs(field))
    im = ax.pcolormesh(x, y, field, cmap=cmap, vmin=-vmax, vmax=vmax, shading="gouraud")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{field_name} (t={ds.time.values:.2f})")
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(field_name)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Saved field plot to {output_file}")
    else:
        # Save to default location instead of showing
        plt.savefig("field_snapshot.png", dpi=150, bbox_inches="tight")
        logger.info("Saved field plot to field_snapshot.png")
    plt.close(fig)


def compute_time_averaged_spectrum(
    field_dir: Path, start_time: Optional[float] = None, end_time: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-averaged energy spectrum from multiple snapshots.

    Args:
        field_dir: Directory containing field files
        start_time: Start time for averaging (None = use all)
        end_time: End time for averaging (None = use all)

    Returns:
        k_centers: Wavenumber centers
        E_k_avg: Time-averaged spectrum
    """
    # Find all field files
    field_files = sorted(field_dir.glob("fields_*.nc"))

    if not field_files:
        logger.error(f"No field files found in {field_dir}")
        return None, None

    # Filter by time if requested
    selected_files = []
    for file in field_files:
        ds = xr.open_dataset(file)
        t = float(ds.time.values)
        ds.close()

        if start_time is not None and t < start_time:
            continue
        if end_time is not None and t > end_time:
            continue

        selected_files.append(file)

    logger.info(f"Averaging over {len(selected_files)} snapshots")

    if not selected_files:
        logger.error("No files in specified time range")
        return None, None

    # Initialize spectrum accumulator
    E_k_sum = None
    k_centers = None

    for i, file in enumerate(selected_files):
        ds = load_output(file)

        if "theta" not in ds:
            continue

        # Get field and grid info
        theta = ds["theta"].values
        N = ds.attrs["N"]
        L = ds.attrs["L"]

        # Compute spectrum (same as in plot_energy_spectrum)
        theta_hat = np.fft.fft2(theta)
        power = np.abs(theta_hat) ** 2

        kx = np.fft.fftfreq(N, d=L / (2 * np.pi * N)) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=L / (2 * np.pi * N)) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky)
        k = np.sqrt(kx**2 + ky**2)

        dk = 2 * np.pi / L
        k_edges = np.arange(0, N / 2 * dk, dk)
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

        E_k = np.zeros(len(k_centers))
        for j in range(len(k_centers)):
            mask = (k >= k_edges[j]) & (k < k_edges[j + 1])
            E_k[j] = np.sum(power[mask])

        E_k /= dk

        # Accumulate
        if E_k_sum is None:
            E_k_sum = E_k
        else:
            E_k_sum += E_k

    # Average
    E_k_avg = E_k_sum / len(selected_files)

    return k_centers, E_k_avg


@click.group()
def cli():
    """pygSQuiG analysis tools."""
    pass


@cli.command()
@click.argument("diagnostics_file", type=click.Path(exists=True))
@click.option(
    "--quantities", "-q", multiple=True, default=["energy", "enstrophy"], help="Quantities to plot"
)
@click.option("--output", "-o", type=click.Path(), help="Output file for plot")
def timeseries(diagnostics_file, quantities, output):
    """Plot diagnostic time series."""
    logger.info(f"Loading diagnostics from {diagnostics_file}")
    diags = load_time_series(Path(diagnostics_file))

    logger.info(f"Available quantities: {list(diags.keys())}")
    plot_time_series(diags, list(quantities), Path(output) if output else None)


@cli.command()
@click.argument("field_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for plot")
@click.option("--slope", "-s", type=float, help="Reference slope to plot")
def spectrum(field_file, output, slope):
    """Plot energy spectrum from field snapshot."""
    logger.info(f"Computing spectrum from {field_file}")
    plot_energy_spectrum(Path(field_file), Path(output) if output else None, slope)


@cli.command()
@click.argument("field_file", type=click.Path(exists=True))
@click.option("--field", "-f", default="theta", help="Field to plot")
@click.option("--output", "-o", type=click.Path(), help="Output file for plot")
@click.option("--cmap", "-c", default="RdBu_r", help="Colormap name")
def snapshot(field_file, field, output, cmap):
    """Plot field snapshot."""
    logger.info(f"Plotting {field} from {field_file}")
    plot_field_snapshot(Path(field_file), field, Path(output) if output else None, cmap)


@cli.command()
@click.argument("field_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--start-time", "-s", type=float, help="Start time for averaging")
@click.option("--end-time", "-e", type=float, help="End time for averaging")
@click.option("--output", "-o", type=click.Path(), help="Output file for plot")
@click.option("--slope", type=float, help="Reference slope to plot")
def averaged_spectrum(field_dir, start_time, end_time, output, slope):
    """Compute and plot time-averaged spectrum."""
    logger.info(f"Computing time-averaged spectrum from {field_dir}")

    k, E_k = compute_time_averaged_spectrum(Path(field_dir), start_time, end_time)

    if k is None:
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k, E_k, "b-", linewidth=2, label="⟨E(k)⟩")

    if slope:
        k_ref = k[10:-10]
        E_ref = E_k[20] * (k_ref / k_ref[10]) ** slope
        ax.loglog(k_ref, E_ref, "k--", alpha=0.5, label=f"k^{{{slope}}}")

    ax.set_xlabel("k")
    ax.set_ylabel("⟨E(k)⟩")
    ax.set_title("Time-Averaged Energy Spectrum")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        logger.info(f"Saved averaged spectrum to {output}")
    else:
        # Save to default location instead of showing
        plt.savefig("averaged_spectrum.png", dpi=150, bbox_inches="tight")
        logger.info("Saved averaged spectrum to averaged_spectrum.png")
    plt.close(fig)


def main():
    """Main entry point for analysis script."""
    cli()


if __name__ == "__main__":
    main()
