"""Comprehensive analysis script for pygSQuiG simulations.

This script provides a full suite of analysis tools including:
- Time series analysis of conserved quantities
- Power spectrum analysis with inertial range identification
- Field snapshots and visualizations
- Summary statistics and reports
"""

import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
import pandas as pd
from typing import Optional, List, Dict, Tuple

from pygsquig.io import load_diagnostics, load_output
from pygsquig.core.grid import make_grid
from pygsquig.utils.diagnostics import (
    compute_energy_spectrum,
    compute_total_energy,
    compute_enstrophy,
    compute_palinstrophy,
    compute_scalar_flux,
)
from pygsquig.utils.plotting import (
    plot_field_slice,
    plot_vorticity,
    plot_velocity_fields,
    plot_energy_spectrum_with_analysis,
    plot_diagnostic_summary,
    plot_time_series_multiplot,
    create_field_animation,
    PlotStyle,
)
from pygsquig.utils import get_logger


logger = get_logger("pygsquig.full_analysis")


def analyze_time_series(
    diagnostics_file: Path,
    output_dir: Path,
    plot_quantities: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyze and plot time series data from diagnostics file.
    
    Args:
        diagnostics_file: Path to diagnostics HDF5 file
        output_dir: Directory to save plots
        plot_quantities: List of quantities to plot (None = all)
        
    Returns:
        DataFrame with statistics for each quantity
    """
    logger.info(f"Analyzing time series from {diagnostics_file}")
    
    # Load diagnostics
    diags = load_diagnostics(diagnostics_file)
    time = diags['time']
    
    # Determine quantities to analyze
    if plot_quantities is None:
        plot_quantities = [k for k in diags.keys() if k != 'time']
    
    # Compute statistics
    stats_data = []
    for quantity in plot_quantities:
        if quantity not in diags:
            logger.warning(f"Quantity '{quantity}' not found in diagnostics")
            continue
            
        data = diags[quantity]
        stats = {
            'quantity': quantity,
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'initial': data[0],
            'final': data[-1],
            'relative_change': (data[-1] - data[0]) / (data[0] + 1e-10)
        }
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics
    stats_file = output_dir / "time_series_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"Saved statistics to {stats_file}")
    
    # Plot time series
    if len(plot_quantities) > 0:
        # Energy quantities
        energy_quantities = ['energy', 'enstrophy', 'palinstrophy']
        energy_data = {q: diags[q] for q in energy_quantities if q in diags}
        if energy_data:
            fig = plot_time_series_multiplot(
                time, energy_data,
                labels={'energy': 'Energy E', 
                        'enstrophy': 'Enstrophy Ω',
                        'palinstrophy': 'Palinstrophy P'},
                output_path=output_dir / "energy_timeseries.png"
            )
            logger.info("Saved energy time series plot")
        
        # Dissipation and flux quantities
        flux_quantities = ['dissipation_rate', 'energy_flux', 'scalar_flux']
        flux_data = {q: diags[q] for q in flux_quantities if q in diags}
        if flux_data:
            fig = plot_time_series_multiplot(
                time, flux_data,
                labels={'dissipation_rate': 'Dissipation ε_ν',
                        'energy_flux': 'Energy Flux ⟨θF⟩',
                        'scalar_flux': 'Scalar Flux ⟨u·∇θ²⟩'},
                output_path=output_dir / "flux_timeseries.png"
            )
            logger.info("Saved flux time series plot")
    
    return stats_df


def analyze_spectrum_evolution(
    field_dir: Path,
    output_dir: Path,
    n_snapshots: int = 5,
    reference_slopes: Optional[Dict[str, float]] = None
) -> Dict[float, float]:
    """Analyze how the energy spectrum evolves over time.
    
    Args:
        field_dir: Directory containing field snapshot files
        output_dir: Directory to save plots
        n_snapshots: Number of snapshots to analyze
        reference_slopes: Reference slopes to plot
        
    Returns:
        Dictionary mapping time to measured spectral slope
    """
    logger.info(f"Analyzing spectrum evolution from {field_dir}")
    
    # Find all field files
    field_files = sorted(field_dir.glob("fields_*.nc"))
    if not field_files:
        logger.error(f"No field files found in {field_dir}")
        return {}
    
    # Select snapshots evenly spaced in time
    if len(field_files) <= n_snapshots:
        selected_files = field_files
    else:
        indices = np.linspace(0, len(field_files)-1, n_snapshots, dtype=int)
        selected_files = [field_files[i] for i in indices]
    
    # Analyze each snapshot
    slopes = {}
    for i, field_file in enumerate(selected_files):
        # Load field
        ds = load_output(field_file)
        time = float(ds.time.values)
        theta_hat = ds['theta_hat'].values if 'theta_hat' in ds else None
        
        if theta_hat is None:
            # Compute FFT if only physical field available
            if 'theta' in ds:
                theta = ds['theta'].values
                theta_hat = np.fft.fft2(theta)
            else:
                logger.warning(f"No theta field in {field_file}")
                continue
        
        # Get grid info
        N = ds.attrs['N']
        L = ds.attrs['L']
        alpha = ds.attrs.get('alpha', 1.0)
        grid = make_grid(N, L)
        
        # Determine inertial range (rough estimate)
        kf = ds.attrs.get('forcing_kf', 30)
        k_min = kf * 1.5
        k_max = N / 3 * 0.7
        
        # Plot spectrum with analysis
        fig, slope = plot_energy_spectrum_with_analysis(
            theta_hat, grid, alpha,
            reference_slopes=reference_slopes,
            inertial_range=(k_min, k_max),
            output_path=output_dir / f"spectrum_t{time:.1f}.png"
        )
        
        slopes[time] = slope
        logger.info(f"t={time:.1f}: Spectral slope = {slope:.3f}")
        
        ds.close()
    
    # Plot slope evolution
    if slopes:
        fig, ax = plt.subplots(figsize=(8, 6))
        times = sorted(slopes.keys())
        slope_values = [slopes[t] for t in times]
        
        ax.plot(times, slope_values, 'o-', linewidth=2, markersize=8)
        ax.axhline(-5/3, color='red', linestyle='--', alpha=0.7, 
                   label='SQG: -5/3')
        ax.set_xlabel('Time')
        ax.set_ylabel('Spectral Slope')
        ax.set_title('Evolution of Spectral Slope')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        fig.savefig(output_dir / "slope_evolution.png", 
                    dpi=PlotStyle.DPI, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved spectral slope evolution plot")
    
    return slopes


def create_field_snapshots(
    field_files: List[Path],
    output_dir: Path,
    field_names: Optional[List[str]] = None,
    n_snapshots: int = 4
) -> None:
    """Create snapshot plots of various fields.
    
    Args:
        field_files: List of field file paths
        output_dir: Directory to save plots
        field_names: Fields to plot (default: theta, vorticity, speed)
        n_snapshots: Number of snapshots to create
    """
    logger.info("Creating field snapshots")
    
    if field_names is None:
        field_names = ['theta', 'vorticity', 'speed']
    
    # Select snapshots
    if len(field_files) <= n_snapshots:
        selected_files = field_files
    else:
        indices = np.linspace(0, len(field_files)-1, n_snapshots, dtype=int)
        selected_files = [field_files[i] for i in indices]
    
    for field_file in selected_files:
        # Load data
        ds = load_output(field_file)
        time = float(ds.time.values)
        N = ds.attrs['N']
        L = ds.attrs['L']
        alpha = ds.attrs.get('alpha', 1.0)
        grid = make_grid(N, L)
        
        # Get theta field
        if 'theta_hat' in ds:
            theta_hat = ds['theta_hat'].values
        elif 'theta' in ds:
            theta = ds['theta'].values
            theta_hat = np.fft.fft2(theta)
        else:
            logger.warning(f"No theta field in {field_file}")
            ds.close()
            continue
        
        # Create comprehensive diagnostic plot
        forcing_info = {}
        if 'forcing_kf' in ds.attrs:
            forcing_info['kf'] = ds.attrs['forcing_kf']
        if 'forcing_epsilon' in ds.attrs:
            forcing_info['epsilon'] = ds.attrs['forcing_epsilon']
        
        plot_diagnostic_summary(
            theta_hat, grid, alpha, time,
            forcing_info=forcing_info,
            output_path=output_dir / f"diagnostic_summary_t{time:.1f}.png"
        )
        
        # Individual field plots if requested
        if 'velocity' in field_names:
            plot_velocity_fields(
                theta_hat, grid, alpha,
                output_path=output_dir / f"velocity_t{time:.1f}.png"
            )
        
        ds.close()
    
    logger.info(f"Created {len(selected_files)} field snapshots")


def generate_analysis_report(
    output_dir: Path,
    stats_df: pd.DataFrame,
    spectral_slopes: Dict[float, float],
    config_info: Dict
) -> None:
    """Generate a text report summarizing the analysis.
    
    Args:
        output_dir: Directory to save report
        stats_df: DataFrame with time series statistics
        spectral_slopes: Dictionary of time -> spectral slope
        config_info: Simulation configuration info
    """
    report_lines = [
        "=" * 60,
        "pygSQuiG Simulation Analysis Report",
        "=" * 60,
        "",
        "Simulation Parameters:",
        f"  Grid: {config_info.get('N', 'N/A')}x{config_info.get('N', 'N/A')}",
        f"  Domain size: L = {config_info.get('L', 'N/A')}",
        f"  Alpha: α = {config_info.get('alpha', 'N/A')}",
        f"  Hyperviscosity: ν_p = {config_info.get('nu_p', 'N/A')}, p = {config_info.get('p', 'N/A')}",
        "",
        "Time Series Statistics:",
        "-" * 40,
    ]
    
    # Add statistics table
    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            report_lines.extend([
                f"\n{row['quantity']}:",
                f"  Mean: {row['mean']:.4e}",
                f"  Std Dev: {row['std']:.4e}",
                f"  Range: [{row['min']:.4e}, {row['max']:.4e}]",
                f"  Relative change: {row['relative_change']:.2%}"
            ])
    
    # Add spectral analysis
    if spectral_slopes:
        report_lines.extend([
            "",
            "Spectral Analysis:",
            "-" * 40,
        ])
        
        times = sorted(spectral_slopes.keys())
        slopes = [spectral_slopes[t] for t in times]
        
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        
        report_lines.extend([
            f"Mean spectral slope: {mean_slope:.3f} ± {std_slope:.3f}",
            f"Expected SQG slope: -5/3 ≈ {-5/3:.3f}",
            f"Deviation from theory: {abs(mean_slope - (-5/3))/(5/3)*100:.1f}%",
            "",
            "Slope evolution:"
        ])
        
        for t, slope in zip(times, slopes):
            report_lines.append(f"  t = {t:.1f}: slope = {slope:.3f}")
    
    report_lines.extend([
        "",
        "=" * 60,
        "Analysis completed successfully",
        "=" * 60
    ])
    
    # Save report
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved analysis report to {report_file}")


@click.command()
@click.argument('simulation_dir', type=click.Path(exists=True, file_okay=False))
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for analysis results')
@click.option('--n-snapshots', '-n', default=5,
              help='Number of field snapshots to analyze')
@click.option('--create-animation', '-a', is_flag=True,
              help='Create animation of field evolution')
@click.option('--animation-field', default='theta',
              help='Field to animate')
@click.option('--reference-slopes', '-r', multiple=True, nargs=2,
              help='Reference slopes for spectrum plots (e.g., -r k^-5/3 -1.667)')
def main(simulation_dir, output_dir, n_snapshots, create_animation, 
         animation_field, reference_slopes):
    """Perform comprehensive analysis of pygSQuiG simulation results.
    
    SIMULATION_DIR: Directory containing simulation output files
    """
    simulation_dir = Path(simulation_dir)
    
    # Set up output directory
    if output_dir is None:
        output_dir = simulation_dir / "analysis"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Analyzing simulation in {simulation_dir}")
    logger.info(f"Saving results to {output_dir}")
    
    # Parse reference slopes
    ref_slopes = {}
    for label, slope in reference_slopes:
        ref_slopes[label] = float(slope)
    if not ref_slopes:
        ref_slopes = {r'$k^{-5/3}$': -5/3, r'$k^{-3}$': -3}
    
    # Find files
    diagnostics_file = simulation_dir / "diagnostics.h5"
    field_files = sorted(simulation_dir.glob("fields_*.nc"))
    
    if not diagnostics_file.exists():
        logger.warning("No diagnostics file found")
        stats_df = pd.DataFrame()
    else:
        # Analyze time series
        stats_df = analyze_time_series(diagnostics_file, output_dir)
    
    if not field_files:
        logger.warning("No field files found")
        spectral_slopes = {}
        config_info = {}
    else:
        # Get configuration from first field file
        ds = xr.open_dataset(field_files[0])
        config_info = dict(ds.attrs)
        ds.close()
        
        # Analyze spectrum evolution
        spectral_slopes = analyze_spectrum_evolution(
            simulation_dir, output_dir, n_snapshots, ref_slopes
        )
        
        # Create field snapshots
        create_field_snapshots(field_files, output_dir, n_snapshots=n_snapshots)
        
        # Create animation if requested
        if create_animation:
            logger.info(f"Creating animation of {animation_field}")
            animation_path = output_dir / f"{animation_field}_evolution.mp4"
            create_field_animation(
                field_files, animation_field, 
                output_path=animation_path
            )
            logger.info(f"Saved animation to {animation_path}")
    
    # Generate summary report
    generate_analysis_report(output_dir, stats_df, spectral_slopes, config_info)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()