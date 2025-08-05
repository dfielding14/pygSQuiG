"""Time series plotting functions for pygSQuiG simulations."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from .style import PlotStyle


def plot_time_series_multiplot(
    time_data: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    labels: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Optional[plt.Figure]:
    """Plot multiple time series in subplots.
    
    Args:
        time_data: Time array
        data_dict: Dictionary of {name: data_array}
        labels: Optional dictionary of {name: ylabel_label}
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object or None if saved
    """
    n_plots = len(data_dict)
    if figsize is None:
        figsize = (10, 3 * n_plots)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    for ax, (name, data) in zip(axes, data_dict.items()):
        ax.plot(time_data, data, linewidth=2)
        ylabel = labels.get(name, name) if labels else name
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Add zero line if data crosses zero
        if np.min(data) < 0 < np.max(data):
            ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Simulation Time Series')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_energy_balance(
    time_data: np.ndarray,
    energy: np.ndarray,
    injection_rate: Optional[np.ndarray] = None,
    dissipation_rate: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Optional[plt.Figure]:
    """Plot energy balance showing energy, injection, and dissipation.
    
    Args:
        time_data: Time array
        energy: Total energy array
        injection_rate: Energy injection rate (optional)
        dissipation_rate: Energy dissipation rate (optional)
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.5),
                                    sharex=True)
    
    # Energy evolution
    ax1.plot(time_data, energy, linewidth=2, label='Total Energy')
    ax1.set_ylabel('Energy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Energy rates
    if injection_rate is not None:
        ax2.plot(time_data, injection_rate, 'g-', linewidth=2, 
                 label='Injection')
    if dissipation_rate is not None:
        ax2.plot(time_data, dissipation_rate, 'r-', linewidth=2, 
                 label='Dissipation')
    if injection_rate is not None and dissipation_rate is not None:
        net_rate = injection_rate - dissipation_rate
        ax2.plot(time_data, net_rate, 'b--', linewidth=1.5, 
                 label='Net', alpha=0.7)
    
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle('Energy Balance')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_conservation_check(
    time_data: np.ndarray,
    quantities: Dict[str, np.ndarray],
    relative: bool = True,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Optional[plt.Figure]:
    """Plot conservation of quantities over time.
    
    Args:
        time_data: Time array
        quantities: Dict of {name: quantity_array}
        relative: If True, plot relative change from initial value
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, data in quantities.items():
        if relative:
            # Plot relative change
            relative_change = (data - data[0]) / abs(data[0])
            ax.plot(time_data, relative_change, linewidth=2, label=name)
        else:
            # Plot absolute values
            ax.plot(time_data, data, linewidth=2, label=name)
    
    ax.set_xlabel('Time')
    if relative:
        ax.set_ylabel('Relative Change')
        ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    else:
        ax.set_ylabel('Value')
    
    ax.set_title('Conservation Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig


def plot_regime_evolution(
    time_data: np.ndarray,
    reynolds: np.ndarray,
    rossby: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Optional[plt.Figure]:
    """Plot evolution of non-dimensional parameters.
    
    Args:
        time_data: Time array
        reynolds: Reynolds number array
        rossby: Rossby number array (optional)
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Figure object or None if saved
    """
    if figsize is None:
        figsize = PlotStyle.FIGSIZE_SINGLE
    
    n_plots = 1 if rossby is None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    # Reynolds number
    axes[0].semilogy(time_data, reynolds, linewidth=2)
    axes[0].set_ylabel('Reynolds Number')
    axes[0].grid(True, alpha=0.3, which='both')
    
    # Rossby number if provided
    if rossby is not None:
        axes[1].semilogy(time_data, rossby, linewidth=2, color='orange')
        axes[1].set_ylabel('Rossby Number')
        axes[1].grid(True, alpha=0.3, which='both')
    
    axes[-1].set_xlabel('Time')
    fig.suptitle('Regime Parameters')
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=PlotStyle.DPI, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig