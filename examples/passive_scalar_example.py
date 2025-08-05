#!/usr/bin/env python3
"""
Example demonstrating passive scalar evolution in gSQG turbulence.

This script shows how to:
1. Add passive scalars to a turbulence simulation
2. Use different source terms
3. Analyze scalar mixing and transport
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from pygsquig.core.grid import make_grid, ifft2
from pygsquig.core.solver_with_scalars import gSQGSolverWithScalars
from pygsquig.scalars.source_terms import (
    LocalizedSource,
    ExponentialGrowth,
    make_heating_source,
    make_cooling_source,
)
from pygsquig.scalars.diagnostics import compute_scalar_variance_spectrum, compute_scalar_flux


def main():
    """Run passive scalar demonstration."""

    # Setup
    N = 256
    L = 2 * np.pi
    grid = make_grid(N, L)

    # Create solver with multiple passive scalars
    passive_scalars = {
        # Dye with no diffusion (pure advection)
        "dye": {"kappa": 0.0},
        # Temperature with diffusion and localized heating
        "temperature": {
            "kappa": 0.01,
            "source": make_heating_source(x0=L / 4, y0=L / 2, radius=L / 8, power=1.0),
        },
        # Chemical species with decay
        "chemical": {"kappa": 0.005, "source": make_cooling_source(decay_time=10.0)},
    }

    solver = gSQGSolverWithScalars(
        grid=grid, alpha=1.0, nu_p=1e-6, p=8, passive_scalars=passive_scalars  # SQG
    )

    # Initial conditions
    # Active scalar: Random turbulence
    theta0 = np.random.randn(N, N)
    theta0 = np.real(np.fft.ifft2(np.fft.fft2(theta0) * (grid.k2 < 20**2)))  # Low-pass filter

    # Passive scalars
    scalar_init = {
        # Dye: Step function
        "dye": np.where(grid.x < L / 2, 1.0, 0.0),
        # Temperature: Gaussian blob
        "temperature": np.exp(-((grid.x - 3 * L / 4) ** 2 + (grid.y - L / 2) ** 2) / (L / 8) ** 2),
        # Chemical: Uniform concentration
        "chemical": np.ones((N, N)),
    }

    # Initialize
    state = solver.initialize(theta0=theta0, scalar_init=scalar_init)

    # Time stepping
    dt = 0.001
    t_end = 5.0
    output_interval = 0.5

    # Storage for diagnostics
    times = []
    dye_variance = []
    temp_flux_x = []
    chem_total = []

    # Create figure for snapshots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    print(f"Running simulation from t=0 to t={t_end}")

    t_output = 0
    while state.time < t_end:
        # Step forward
        state = solver.step(state, dt)

        # Output
        if state.time >= t_output:
            print(f"t = {state.time:.2f}")

            # Get fields
            theta = ifft2(state.theta_hat).real
            dye = ifft2(state.scalar_state.scalars["dye"]).real
            temp = ifft2(state.scalar_state.scalars["temperature"]).real
            chem = ifft2(state.scalar_state.scalars["chemical"]).real

            # Compute velocity
            u, v = solver.compute_velocity(state.theta_hat)

            # Store diagnostics
            times.append(state.time)

            # Dye variance (mixing metric)
            dye_var = np.var(dye)
            dye_variance.append(dye_var)

            # Temperature flux
            flux_x, flux_y = compute_scalar_flux(
                state.scalar_state.scalars["temperature"], u, v, grid
            )
            temp_flux_x.append(flux_x)

            # Total chemical (conservation check)
            chem_total.append(np.sum(chem))

            # Update plots
            if t_output == 0 or state.time >= t_end - dt:
                for ax in axes.flat:
                    ax.clear()

                # Active scalar (vorticity)
                im1 = axes[0, 0].imshow(theta, cmap="RdBu_r", origin="lower", extent=[0, L, 0, L])
                axes[0, 0].set_title(f"Vorticity (t={state.time:.1f})")

                # Passive dye
                im2 = axes[0, 1].imshow(
                    dye, cmap="viridis", origin="lower", extent=[0, L, 0, L], vmin=0, vmax=1
                )
                axes[0, 1].set_title("Passive Dye")

                # Temperature
                im3 = axes[0, 2].imshow(
                    temp, cmap="hot", origin="lower", extent=[0, L, 0, L], vmin=0
                )
                axes[0, 2].set_title("Temperature")

                # Velocity magnitude
                speed = np.sqrt(u**2 + v**2)
                im4 = axes[1, 0].imshow(speed, cmap="magma", origin="lower", extent=[0, L, 0, L])
                axes[1, 0].set_title("|u|")

                # Chemical concentration
                im5 = axes[1, 1].imshow(
                    chem, cmap="Blues", origin="lower", extent=[0, L, 0, L], vmin=0, vmax=1
                )
                axes[1, 1].set_title("Chemical")

                # Scalar variance spectra
                axes[1, 2].clear()
                for scalar_name, color in [
                    ("dye", "green"),
                    ("temperature", "red"),
                    ("chemical", "blue"),
                ]:
                    k, E_k = compute_scalar_variance_spectrum(
                        state.scalar_state.scalars[scalar_name], grid
                    )
                    axes[1, 2].loglog(k, E_k, color=color, label=scalar_name)
                axes[1, 2].set_xlabel("k")
                axes[1, 2].set_ylabel("Variance Spectrum")
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

                for ax in axes.flat:
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")

                plt.tight_layout()

            t_output += output_interval

    # Final plots
    plt.savefig("passive_scalar_snapshots.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Plot time series
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))

    # Dye variance (mixing)
    axes2[0].plot(times, dye_variance, "g-", linewidth=2)
    axes2[0].set_xlabel("Time")
    axes2[0].set_ylabel("Dye Variance")
    axes2[0].set_title("Mixing Rate")
    axes2[0].grid(True, alpha=0.3)

    # Temperature flux
    axes2[1].plot(times, temp_flux_x, "r-", linewidth=2)
    axes2[1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes2[1].set_xlabel("Time")
    axes2[1].set_ylabel("Heat Flux (x-component)")
    axes2[1].set_title("Turbulent Transport")
    axes2[1].grid(True, alpha=0.3)

    # Chemical conservation
    axes2[2].plot(times, chem_total, "b-", linewidth=2)
    axes2[2].set_xlabel("Time")
    axes2[2].set_ylabel("Total Chemical")
    axes2[2].set_title("Conservation with Decay")
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("passive_scalar_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\n=== Simulation Summary ===")
    print(f"Final time: {state.time:.2f}")
    print(f"Dye mixing: variance reduced by {(1 - dye_variance[-1]/dye_variance[0])*100:.1f}%")
    print(f"Mean temperature flux: {np.mean(temp_flux_x):.3e}")
    print(f"Chemical decay: {(1 - chem_total[-1]/chem_total[0])*100:.1f}%")

    # Get final diagnostics
    diags = solver.get_diagnostics(state)
    print("\nFinal diagnostics:")
    for key, value in sorted(diags.items()):
        if "scalar" in key or "dye" in key or "temp" in key or "chem" in key:
            print(f"  {key}: {value:.3e}")


if __name__ == "__main__":
    main()
