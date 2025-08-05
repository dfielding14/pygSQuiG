"""Configuration system for pygSQuiG simulations.

This module provides dataclasses for configuring all aspects of a pygSQuiG simulation,
including grid parameters, solver settings, forcing, output options, and more.
Configuration files are written in YAML and converted to these dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
import yaml
from pathlib import Path
import numpy as np


@dataclass
class GridConfig:
    """Configuration for the computational grid.
    
    Attributes:
        N: Number of grid points in each direction (N x N grid)
        L: Domain size (square domain [0, L] x [0, L])
    """
    N: int = 256
    L: float = 2 * np.pi
    
    def __post_init__(self):
        if self.N <= 0 or self.N % 2 != 0:
            raise ValueError(f"N must be a positive even integer, got {self.N}")
        if self.L <= 0:
            raise ValueError(f"L must be positive, got {self.L}")


@dataclass
class DissipationConfig:
    """Configuration for small-scale dissipation.
    
    Attributes:
        type: Type of dissipation ('hyperviscosity' or 'viscosity')
        nu_p: Dissipation coefficient
        p: Order of dissipation operator (-Δ)^(p/2)
    """
    type: str = "hyperviscosity"
    nu_p: float = 1.0e-16
    p: int = 8
    
    def __post_init__(self):
        if self.type not in ["hyperviscosity", "viscosity"]:
            raise ValueError(f"Unknown dissipation type: {self.type}")
        if self.nu_p < 0:
            raise ValueError(f"nu_p must be non-negative, got {self.nu_p}")
        if self.p not in [2, 4, 8]:
            raise ValueError(f"p must be 2, 4, or 8, got {self.p}")


@dataclass 
class DampingConfig:
    """Configuration for large-scale damping.
    
    Attributes:
        type: Type of damping ('linear_drag' or 'none')
        mu: Linear drag coefficient
        k_cutoff_factor: Apply damping for k < k_f * k_cutoff_factor
    """
    type: str = "linear_drag"
    mu: float = 0.1
    k_cutoff_factor: float = 0.5
    
    def __post_init__(self):
        if self.type not in ["linear_drag", "none"]:
            raise ValueError(f"Unknown damping type: {self.type}")
        if self.mu < 0:
            raise ValueError(f"mu must be non-negative, got {self.mu}")
        if self.k_cutoff_factor <= 0:
            raise ValueError(f"k_cutoff_factor must be positive, got {self.k_cutoff_factor}")


@dataclass
class TimeIntegrationConfig:
    """Configuration for time integration.
    
    Attributes:
        method: Time integration method ('RK4' or 'SSP-RK3')
        dt: Fixed time step (ignored if adaptive_cfl is True)
        adaptive_cfl: Whether to use adaptive time stepping based on CFL
        cfl_safety: Safety factor for CFL condition (< 1)
        dt_max: Maximum allowed time step
    """
    method: str = "RK4"
    dt: float = 0.001
    adaptive_cfl: bool = True
    cfl_safety: float = 0.8
    dt_max: Optional[float] = None
    
    def __post_init__(self):
        if self.method not in ["RK4", "SSP-RK3"]:
            raise ValueError(f"Unknown time integration method: {self.method}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not 0 < self.cfl_safety < 1:
            raise ValueError(f"cfl_safety must be in (0, 1), got {self.cfl_safety}")
        if self.dt_max is not None and self.dt_max <= 0:
            raise ValueError(f"dt_max must be positive, got {self.dt_max}")


@dataclass
class SolverConfig:
    """Configuration for the gSQG solver.
    
    Attributes:
        alpha: Exponent in fractional Laplacian (alpha in [-2, 2])
        dissipation: Small-scale dissipation configuration
        damping: Large-scale damping configuration  
        time_integration: Time integration configuration
    """
    alpha: float
    dissipation: DissipationConfig = field(default_factory=DissipationConfig)
    damping: Optional[DampingConfig] = None
    time_integration: TimeIntegrationConfig = field(default_factory=TimeIntegrationConfig)
    
    def __post_init__(self):
        if not -2 <= self.alpha <= 2:
            raise ValueError(f"alpha must be in [-2, 2], got {self.alpha}")


@dataclass
class ForcingConfig:
    """Configuration for turbulent forcing.
    
    Attributes:
        type: Type of forcing ('ring' or 'none')
        kf: Forcing wavenumber
        dk: Width of forcing ring
        epsilon: Target energy injection rate
        tau_f: Correlation time (0 for white noise)
        seed: Random seed for reproducibility
    """
    type: str = "ring"
    kf: float = 20.0
    dk: float = 1.0
    epsilon: float = 0.1
    tau_f: float = 0.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.type not in ["ring", "none"]:
            raise ValueError(f"Unknown forcing type: {self.type}")
        if self.kf <= 0:
            raise ValueError(f"kf must be positive, got {self.kf}")
        if self.dk <= 0:
            raise ValueError(f"dk must be positive, got {self.dk}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.tau_f < 0:
            raise ValueError(f"tau_f must be non-negative, got {self.tau_f}")


@dataclass
class OutputConfig:
    """Configuration for output and diagnostics.
    
    Attributes:
        fields: List of fields to save ('theta', 'vorticity', 'streamfunction')
        diagnostics: List of diagnostics to compute
        save_every_n_steps: Save frequency in steps (if not None)
        compress: Whether to use HDF5 compression
    """
    fields: List[str] = field(default_factory=lambda: ["theta"])
    diagnostics: List[str] = field(default_factory=lambda: ["energy_spectrum", "scalar_flux"])
    save_every_n_steps: Optional[int] = None
    compress: bool = True
    
    def __post_init__(self):
        valid_fields = {"theta", "vorticity", "streamfunction", "velocity", "scalars"}
        invalid = set(self.fields) - valid_fields
        if invalid:
            raise ValueError(f"Unknown fields: {invalid}")
            
        valid_diagnostics = {"energy_spectrum", "scalar_flux", "enstrophy", "energy_flux"}
        invalid = set(self.diagnostics) - valid_diagnostics
        if invalid:
            raise ValueError(f"Unknown diagnostics: {invalid}")


@dataclass
class SimulationConfig:
    """Configuration for simulation control.
    
    Attributes:
        t_end: End time of simulation
        output_interval: Time interval between outputs
        checkpoint_interval: Time interval between checkpoints
        wall_time_limit: Maximum wall time in seconds (optional)
        log_interval: Time interval between log messages
    """
    t_end: float
    output_interval: float = 1.0
    checkpoint_interval: float = 10.0
    wall_time_limit: Optional[float] = None
    log_interval: float = 0.1
    
    def __post_init__(self):
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.output_interval <= 0:
            raise ValueError(f"output_interval must be positive, got {self.output_interval}")
        if self.checkpoint_interval <= 0:
            raise ValueError(f"checkpoint_interval must be positive, got {self.checkpoint_interval}")
        if self.wall_time_limit is not None and self.wall_time_limit <= 0:
            raise ValueError(f"wall_time_limit must be positive, got {self.wall_time_limit}")
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


@dataclass
class ScalarSourceConfig:
    """Configuration for passive scalar source terms.
    
    Attributes:
        type: Type of source term ('exponential', 'localized', 'chemical', 'periodic')
        parameters: Source-specific parameters
    """
    type: str
    parameters: dict = field(default_factory=dict)
    
    def __post_init__(self):
        valid_types = {"exponential", "localized", "chemical", "periodic", "none"}
        if self.type not in valid_types:
            raise ValueError(f"Unknown source type: {self.type}")


@dataclass
class PassiveScalarConfig:
    """Configuration for a single passive scalar species.
    
    Attributes:
        name: Identifier for the scalar
        kappa: Diffusivity coefficient
        source: Optional source term configuration
        initial_condition: Initial condition type ('zero', 'random', 'gaussian')
        initial_params: Parameters for initial condition
    """
    name: str
    kappa: float = 0.0
    source: Optional[ScalarSourceConfig] = None
    initial_condition: str = "zero"
    initial_params: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.kappa < 0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        
        valid_ic = {"zero", "random", "gaussian", "uniform"}
        if self.initial_condition not in valid_ic:
            raise ValueError(f"Unknown initial condition: {self.initial_condition}")


@dataclass
class ScalarsConfig:
    """Configuration for passive scalars.
    
    Attributes:
        enabled: Whether to include passive scalars
        species: List of scalar configurations
    """
    enabled: bool = False
    species: List[PassiveScalarConfig] = field(default_factory=list)
    
    def __post_init__(self):
        # Check for duplicate names
        names = [s.name for s in self.species]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate scalar names found")


@dataclass
class InitialConditionConfig:
    """Configuration for initial conditions.
    
    Attributes:
        type: Type of initial condition ('random', 'checkpoint', 'function')
        checkpoint_path: Path to checkpoint file (if type='checkpoint')
        amplitude: Amplitude for random initial conditions
        seed: Random seed for reproducibility
    """
    type: str = "random"
    checkpoint_path: Optional[str] = None
    amplitude: float = 1.0
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.type not in ["random", "checkpoint", "function"]:
            raise ValueError(f"Unknown initial condition type: {self.type}")
        if self.type == "checkpoint" and self.checkpoint_path is None:
            raise ValueError("checkpoint_path required when type='checkpoint'")
        if self.amplitude <= 0:
            raise ValueError(f"amplitude must be positive, got {self.amplitude}")


@dataclass
class RunConfig:
    """Main configuration for a pygSQuiG simulation run.
    
    This is the top-level configuration that includes all sub-configurations.
    """
    grid: GridConfig
    solver: SolverConfig
    forcing: Optional[ForcingConfig] = None
    scalars: Optional[ScalarsConfig] = None
    output: OutputConfig = field(default_factory=OutputConfig)
    simulation: SimulationConfig = field(default_factory=lambda: SimulationConfig(t_end=100.0))
    initial_condition: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RunConfig":
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            RunConfig instance
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration data
            
        Returns:
            RunConfig instance
        """
        # Create sub-configurations
        grid = GridConfig(**data.get("grid", {}))
        
        solver_data = data.get("solver", {})
        if "dissipation" in solver_data:
            solver_data["dissipation"] = DissipationConfig(**solver_data["dissipation"])
        if "damping" in solver_data:
            solver_data["damping"] = DampingConfig(**solver_data["damping"])
        if "time_integration" in solver_data:
            solver_data["time_integration"] = TimeIntegrationConfig(**solver_data["time_integration"])
        solver = SolverConfig(**solver_data)
        
        forcing = None
        if "forcing" in data:
            forcing = ForcingConfig(**data["forcing"])
            
        scalars = None
        if "scalars" in data:
            scalars_data = data["scalars"]
            if scalars_data.get("enabled", False):
                # Parse species configurations
                species_list = []
                for species_data in scalars_data.get("species", []):
                    # Parse source configuration if present
                    source = None
                    if "source" in species_data and species_data["source"] is not None:
                        source_data = species_data["source"]
                        source = ScalarSourceConfig(**source_data)
                    
                    # Create PassiveScalarConfig
                    species_config = PassiveScalarConfig(
                        name=species_data["name"],
                        kappa=species_data.get("kappa", 0.0),
                        source=source,
                        initial_condition=species_data.get("initial_condition", "zero"),
                        initial_params=species_data.get("initial_params", {})
                    )
                    species_list.append(species_config)
                
                scalars = ScalarsConfig(enabled=True, species=species_list)
            
        output = OutputConfig(**data.get("output", {}))
        simulation = SimulationConfig(**data.get("simulation", {"t_end": 100.0}))
        initial_condition = InitialConditionConfig(**data.get("initial_condition", {}))
        
        return cls(
            grid=grid,
            solver=solver,
            forcing=forcing,
            scalars=scalars,
            output=output,
            simulation=simulation,
            initial_condition=initial_condition
        )
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        data = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        from dataclasses import asdict
        
        data = asdict(self)
        
        # Remove None values and empty dicts
        def clean_dict(d):
            if isinstance(d, dict):
                return {k: clean_dict(v) for k, v in d.items() 
                        if v is not None and (not isinstance(v, dict) or v)}
            return d
        
        return clean_dict(data)


def load_config(path: Union[str, Path]) -> RunConfig:
    """Load configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        RunConfig instance
    """
    return RunConfig.from_yaml(path)