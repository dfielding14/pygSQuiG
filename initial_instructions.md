# **PygSQuiG**  
*A Python / JAX solver for the **g**eneralised **S**urface-**Qu**asi-**G**eostrophic family (α ∈ [-2, 2]) with built-in turbulent forcing*

---

## 1. Numerical Model

| Item | Specification |
|------|---------------|
| **Governing equations** | $$\partial_t\theta + \mathbf u\!\cdot\!\nabla\theta = F - D,\qquad \mathbf u = \nabla^{\!\perp}(-\Delta)^{-\alpha/2}\theta$$ |
| **Grid** | Doubly periodic, \(N^2\) collocation points |
| **Spatial discretisation** | Pseudo-spectral with 2/3 dealiasing |
| **Time integration** | RK4 (optionally SSP-RK3) with CFL monitor |
| **Small-scale dissipation** | \(\nu_p(-\Delta)^{p/2}\theta\) ( \(p = 2,4,8\) configurable) |
| **Large-scale damping** | Linear drag \(-\mu\theta\) active for \(k < k_f/2\) |

---

## 2. Built-in Turbulent Driver

| Element | Specification |
|---------|---------------|
| **Type** | Constant-amplitude, random-phase *ring forcing* in Fourier space |
| **Shell** | Modes with \(|k-k_f| \le \Delta k/2\); default \(k_f = 20·2\pi/L\), \(\Delta k = 1\) |
| **Temporal correlation** | *White in time* (default) or OU process with user-set \(τ_f\) |
| **Injection control** | Target scalar-energy flux \(ε\); driver rescales so \(\langle \theta F\rangle = ε\) |
| **Reference** | Matches the statistically-stationary setup of Valadão et al. 2025 |

---

## 3. Repository Layout

pygsquig/
├─ pygsquig/
│  ├─ core/               # numerics
│  │   ├─ grid.py
│  │   ├─ operators.py
│  │   ├─ time_integrator.py
│  │   └─ solver.py
│  ├─ forcing/
│  │   ├─ ring_forcing.py
│  │   └─ damping.py
│  ├─ io/
│  │   ├─ config.py
│  │   └─ hdf5_io.py
│  ├─ utils/
│  │   ├─ diagnostics.py
│  │   └─ logging.py
│  └─ init.py
├─ scripts/
│  ├─ run.py
│  └─ analyse.py
├─ examples/
│  ├─ sqg_alpha1_forced.yml
│  └─ euler_2d.yml
├─ tests/
│  ├─ test_operators.py
│  ├─ test_forcing.py
│  ├─ test_time_integrator.py
│  └─ test_end2end.py
├─ docs/                  # Sphinx
├─ pyproject.toml
├─ .pre-commit-config.yaml
└─ README.md

---

## 4. Team Roles

| Agent | Responsibility |
|-------|----------------|
| **agent-core** | Implement `grid`, `operators`, `time_integrator`, `solver`; ensure JAX double-precision and `pmap`-readiness |
| **agent-forcing** | Build `ring_forcing` and `damping`; expose YAML interface; provide analytic notebook verifying constant \(ε\) --- following https://arxiv.org/html/2504.07914v2|
| **agent-io** | Design YAML → dataclass schema; implement HDF5 checkpoints & CLI `run.py` |
| **agent-testing** | Create unit & regression tests (conservation, forcing spectrum, \(⟨θF⟩ ≈ ε\)); configure GitHub Actions |
| **agent-docs** | Write Sphinx docs, quick-start guide, tutorial notebooks, maintain `examples/` |
| **agent-integrator** | Review PRs, enforce style via pre-commit, maintain roadmap and weekly stand-ups |

---

## 5. Coding Guidelines

* **Style**: PEP 8 + _black_ (99-char lines); enforced by pre-commit  
* **Typing**: Full PEP 484 hints; NumPy-style docstrings  
* **Backend**: All core numerics use `jax.numpy`; no bare NumPy  
* **Randomness**: Forcing relies on `jax.random` with explicit PRNG key flow  
* **Testing**: Every public function covered by `pytest`; CI runs a short CPU sanity job

---

## 6. Implementation Checklist

- [ ] Scaffold repo with Poetry/Hatch and pre-commit hooks  
- [ ] Validate `operators.py` against analytic \(|k|^{\alpha}\)  
- [ ] Implement `RingForcing.__call__(state, key, dt) → F̂` (JIT-safe)  
- [ ] Integrate forcing & damping into `solver.step()`  
- [ ] Unit tests: forcing spectrum peaks at \(k_f\); time-averaged \(⟨θF⟩ / ε ≈ 1\)  
- [ ] Example run (`sqg_alpha1_forced.yml`) reproduces \(k^{-5/3}\) KE spectrum  
- [ ] Tag **v0.1** after successful 2048², α = 1 benchmark

---

## 7. Post-v0.1 Roadmap

* Multi-GPU via `pjit` domain decomposition  
* Adaptive time stepping (CFL)  
* Real-time dashboard (Dash / Panel)  
* Vertical-mode stacks (3-D QG variants)  
* CUDA-accelerated I/O compression

---

## 8. Install & Run (End-User View)

```bash
pip install pygsquig
pygsquig-run examples/sqg_alpha1_forced.yml --device=gpu

