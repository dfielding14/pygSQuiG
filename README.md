# pygSQuiG

A Python/JAX solver for the **g**eneralised **S**urface-**Qu**asi-**G**eostrophic family (α ∈ [-2, 2]) with built-in turbulent forcing.

## Overview

pygSQuiG is a high-performance numerical solver for the generalized Surface-Quasi-Geostrophic (gSQG) equations, designed for research in geophysical turbulence. It features:

- **Flexible physics**: Supports the full gSQG family with α ∈ [-2, 2]
- **GPU acceleration**: Built on JAX for efficient GPU computation
- **Turbulent forcing**: Constant-energy-flux ring forcing with configurable parameters
- **Modern numerics**: Pseudo-spectral methods with proper dealiasing
- **Research-ready**: HDF5/xarray output, comprehensive diagnostics, and checkpoint/restart

## Equations

The code solves:

```
∂_t θ + u·∇θ = F - D
u = ∇^⊥(-Δ)^(-α/2)θ
```

where:
- θ is the active scalar (e.g., buoyancy)
- α controls the relationship between θ and velocity (α=1 for SQG, α=0 for 2D Euler)
- F is the turbulent forcing
- D represents dissipation (hyperviscosity and large-scale drag)

## Installation

### Basic Installation

```bash
pip install pygsquig
```

### GPU Support

```bash
pip install pygsquig[gpu]
```

### Development Installation

```bash
git clone https://github.com/yourusername/pygsquig.git
cd pygsquig
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run a forced SQG simulation
pygsquig-run examples/sqg_alpha1_forced.yml --device=gpu

# Analyze the output
pygsquig-analyse output/sqg_run_001.h5 --plot-spectra
```

## Example Configuration

```yaml
grid:
  N: 512
  L: 6.283185307179586  # 2π

solver:
  alpha: 1.0
  dissipation:
    type: hyperviscosity
    nu_p: 1.0e-16
    p: 8

forcing:
  type: ring
  kf: 20.0
  epsilon: 0.1

simulation:
  t_end: 100.0
  output_interval: 1.0
```

## Documentation

Full documentation is available at [Read the Docs](https://pygsquig.readthedocs.io).

## Development Status

⚠️ **Under Active Development**: This project is in early development. APIs may change.

Current focus:
- Core numerical implementation
- Basic forcing and dissipation
- Test suite and validation
- Documentation

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use pygSQuiG in your research, please cite:

```bibtex
@software{pygsquig,
  title = {pygSQuiG: A Python/JAX solver for generalised Surface-Quasi-Geostrophic turbulence},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/pygsquig}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This implementation follows the numerical methods and forcing approach described in Valdivieso da Costa & Tithof (2025).