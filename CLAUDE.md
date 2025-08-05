# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**pygSQuiG** is a Python/JAX-based solver for the generalized Surface-Quasi-Geostrophic (gSQG) family of equations, designed for computational fluid dynamics research in geophysical turbulence. The project is currently in the planning phase with detailed specifications in `initial_instructions.md`.

## Key Technical Context

### Numerical Model
- Solves: ∂_t θ + u·∇θ = F - D, where u = ∇^⊥(-Δ)^(-α/2)θ
- Uses pseudo-spectral methods with JAX for GPU acceleration
- Implements RK4 time integration with CFL monitoring
- Supports turbulent forcing and various dissipation schemes

### Architecture Philosophy
- **JAX-first**: All numerical operations use JAX arrays and transformations
- **Functional programming**: Prefer pure functions, use JAX's functional patterns
- **Type safety**: Full typing with PEP 484 annotations throughout
- **Performance**: Design for GPU execution and potential multi-GPU scaling

## Development Commands

Since the project is not yet implemented, expected commands based on specifications:

```bash
# Testing (when implemented)
pytest                          # Run all tests
pytest tests/core/             # Run specific test module

# Code quality (when configured)
black .                        # Format code (99-char line limit)
pre-commit run --all-files     # Run all pre-commit hooks

# Running simulations (planned interface)
pygsquig-run config.yml --device=gpu
```

## Development Philosophy

### Test-Driven Development
**CRITICAL**: Every function must have tests BEFORE moving to the next feature.
- Write tests first or immediately after implementation
- Test edge cases, not just happy paths
- Numerical tests should verify against analytical solutions where possible
- Use property-based testing for mathematical invariants
- Aim for >95% coverage, but focus on meaningful tests

### Simplicity Over Cleverness
- **KISS Principle**: Keep implementations simple and readable
- **No Over-Engineering**: Avoid abstract base classes, complex inheritance, or unnecessary patterns
- **YAGNI**: Don't add features "just in case" - implement only what's needed
- **Clear > Clever**: Optimize for readability, not cleverness
- **Flat > Nested**: Prefer simple functions over deep class hierarchies

### Best Practices
- **Single Responsibility**: Each function does one thing well
- **Explicit > Implicit**: Clear parameter names and types
- **Fail Fast**: Validate inputs early with clear error messages
- **Pure Functions**: Minimize side effects, especially in numerical code

## High-Level Architecture

The codebase follows a modular structure designed for scientific computing:

1. **Core Numerics** (`pygsquig/core/`): Contains the mathematical heart of the solver
   - Grid management and spectral operations
   - Differential operators using JAX
   - Time integration schemes (RK4, SSP-RK3)
   - Main solver orchestration

2. **Forcing System** (`pygsquig/forcing/`): Implements turbulent drivers
   - Ring forcing in Fourier space
   - Energy injection control
   - Damping mechanisms

3. **I/O Layer** (`pygsquig/io/`): Handles configuration and data management
   - YAML-based configuration
   - HDF5 output with xarray integration
   - Checkpoint/restart functionality

4. **Utilities** (`pygsquig/utils/`): Supporting functionality
   - Diagnostics (energy spectra, fluxes)
   - Logging and monitoring
   - CFL tracking

## Important Implementation Guidelines

### JAX-Specific Patterns
- Use `jax.jit` for performance-critical functions
- Leverage `jax.vmap` for batch operations
- Implement functions as pure transformations of JAX arrays
- Use `chex` for testing JAX code

### Scientific Computing Considerations
- Maintain numerical precision in spectral operations
- Implement proper dealiasing (2/3 rule)
- Ensure energy conservation in inviscid limit
- Follow established conventions from the gSQG literature

### Development Workflow
When implementing features:
1. Start with the mathematical formulation from `initial_instructions.md`
2. Implement core numerics in JAX following functional patterns
3. Add comprehensive tests including convergence checks
4. Document with NumPy-style docstrings
5. Ensure GPU compatibility from the start

## Two-Agent Development Strategy

This project uses a two-agent development approach to parallelize implementation:

### Agent 1: Core & Forcing (Numerical Implementation)
- **Focus**: Mathematical/scientific computing aspects
- **Modules**: `pygsquig/core/` and `pygsquig/forcing/`
- **Responsibilities**: 
  - Implement grid, operators, time integrator, and solver
  - Build ring forcing and damping mechanisms
  - Ensure JAX double-precision and `pmap`-readiness
  - Verify forcing maintains constant energy flux ε
- **See**: `agent-1.md` for detailed specifications

### Agent 2: Infrastructure & Support (Software Engineering)
- **Focus**: Project infrastructure and software engineering
- **Modules**: `pygsquig/io/`, `pygsquig/utils/`, tests, docs, scripts
- **Responsibilities**:
  - Design YAML configuration and I/O system
  - Create comprehensive test suite
  - Write documentation and examples
  - Set up CI/CD and development tools
- **See**: `agent-2.md` for detailed specifications

### Coordination Points
- **Interfaces**: Agent 1 defines core data structures; Agent 2 consumes them for I/O
- **Testing**: Agent 2 writes tests for Agent 1's implementations
- **Integration**: Both agents coordinate on the main solver interface
- **Communication**: Use clear type hints and docstrings at module boundaries

## Agent Communication Protocol

Agents coordinate asynchronously through `COMM.md` file. This ensures clear handoffs and prevents blocking.

### How to Communicate
1. **Check COMM.md** at the start of each work session
2. **Post updates** when completing work that affects the other agent
3. **Add blockers** immediately when you need something from the other agent
4. **Answer questions** within 24 hours (or post a timeline)

### Key Sections in COMM.md
- **🔄 Active Blockers**: Urgent items blocking progress
- **📐 Interface Definitions**: Data structures and signatures (Agent 1 posts, Agent 2 consumes)
- **✅ Progress Updates**: Completed work now available
- **🔔 Change Notifications**: Breaking changes or important updates
- **❓ Questions & Answers**: Async discussions
- **📋 Task Coordination**: Current and upcoming work

### Entry Format
```
[2024-01-15 Agent1] Status: Title
- Details in bullet points
- Code blocks for interfaces
```

### Example Workflow
1. Agent 1 implements Grid dataclass
2. Agent 1 posts interface definition in COMM.md
3. Agent 2 sees update and creates configuration schema
4. Agent 2 posts progress and any questions
5. Both agents continue with unblocked work