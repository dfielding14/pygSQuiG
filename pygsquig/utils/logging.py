"""Logging utilities for pygSQuiG simulations.

This module provides structured logging capabilities with support for
both console and file output, progress tracking, and performance monitoring.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class SimulationLogger:
    """Logger for pygSQuiG simulations with structured output."""

    def __init__(
        self,
        name: str = "pygsquig",
        console_level: str = "INFO",
        file_path: Optional[Path] = None,
        file_level: str = "DEBUG",
    ):
        """Initialize simulation logger.

        Args:
            name: Logger name
            console_level: Logging level for console output
            file_path: Optional path for log file
            file_level: Logging level for file output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all messages

        # Remove existing handlers
        self.logger.handlers = []

        # Console handler with custom formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler if requested
        if file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # Store metadata for structured logging
        self.metadata: Dict[str, Any] = {}

    def set_metadata(self, **kwargs):
        """Set metadata that will be included in structured log messages."""
        self.metadata.update(kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        extra_data = {**self.metadata, **kwargs}
        if extra_data:
            message = f"{message} | {json.dumps(extra_data, default=str)}"
        self.logger.info(message)

    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        extra_data = {**self.metadata, **kwargs}
        if extra_data:
            message = f"{message} | {json.dumps(extra_data, default=str)}"
        self.logger.debug(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        extra_data = {**self.metadata, **kwargs}
        if extra_data:
            message = f"{message} | {json.dumps(extra_data, default=str)}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        extra_data = {**self.metadata, **kwargs}
        if extra_data:
            message = f"{message} | {json.dumps(extra_data, default=str)}"
        self.logger.error(message)

    def log_simulation_start(self, config, device: str = "cpu"):
        """Log simulation start with configuration details."""
        self.info("=" * 60)
        self.info("pygSQuiG Simulation Starting")
        self.info(f"Grid: {config.grid.N}×{config.grid.N}, L={config.grid.L:.3f}")
        self.info(f"Solver: α={config.solver.alpha}, ν_p={config.solver.dissipation.nu_p:.2e}")
        self.info(f"Time: t_end={config.simulation.t_end}")
        self.info(f"Device: {device}")
        if config.forcing:
            self.info(f"Forcing: k_f={config.forcing.kf}, ε={config.forcing.epsilon}")
        self.info("=" * 60)

    def log_progress(
        self,
        time: float,
        step: int,
        dt: float,
        diagnostics: Dict[str, float],
        eta_seconds: Optional[float] = None,
    ):
        """Log simulation progress with diagnostics."""
        msg_parts = [f"t={time:8.3f}", f"step={step:6d}", f"dt={dt:.2e}"]

        # Add key diagnostics
        if "theta_rms" in diagnostics:
            msg_parts.append(f"θ_rms={diagnostics['theta_rms']:.3e}")
        if "energy" in diagnostics:
            msg_parts.append(f"E={diagnostics['energy']:.3e}")
        if "enstrophy" in diagnostics:
            msg_parts.append(f"Ω={diagnostics['enstrophy']:.3e}")
        if "cfl" in diagnostics:
            msg_parts.append(f"CFL={diagnostics['cfl']:.3f}")

        # Add ETA if available
        if eta_seconds is not None and eta_seconds > 0:
            hours = int(eta_seconds // 3600)
            minutes = int((eta_seconds % 3600) // 60)
            seconds = int(eta_seconds % 60)
            eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            msg_parts.append(f"ETA={eta_str}")

        self.info(" | ".join(msg_parts))

    def log_checkpoint(self, checkpoint_path: Path, time: float, step: int):
        """Log checkpoint save."""
        self.info(f"Checkpoint saved: {checkpoint_path.name}", time=time, step=step)

    def log_output(self, output_path: Path, time: float, step: int):
        """Log output save."""
        self.debug(f"Output saved: {output_path.name}", time=time, step=step)

    def log_simulation_complete(self, final_time: float, total_steps: int, wall_time: float):
        """Log simulation completion."""
        self.info("=" * 60)
        self.info("Simulation Complete!")
        self.info(f"Final time: {final_time:.3f}")
        self.info(f"Total steps: {total_steps}")
        self.info(f"Wall time: {wall_time:.1f} seconds")
        self.info(f"Performance: {total_steps/wall_time:.1f} steps/second")
        self.info("=" * 60)


class ProgressBar:
    """Simple progress bar for long-running simulations."""

    def __init__(self, total: float, width: int = 50):
        """Initialize progress bar.

        Args:
            total: Total value (e.g., t_end)
            width: Width of progress bar in characters
        """
        self.total = total
        self.width = width
        self.current = 0.0

    def update(self, current: float) -> str:
        """Update progress and return bar string.

        Args:
            current: Current value

        Returns:
            Progress bar string
        """
        self.current = current
        progress = min(current / self.total, 1.0)
        filled = int(progress * self.width)
        empty = self.width - filled

        bar = f"[{'=' * filled}{' ' * empty}] {progress*100:5.1f}%"
        return bar


def setup_logging(
    output_dir: Optional[Path] = None, console_level: str = "INFO", file_level: str = "DEBUG"
) -> SimulationLogger:
    """Set up logging for a simulation run.

    Args:
        output_dir: Directory for log files (if None, no file logging)
        console_level: Console logging level
        file_level: File logging level

    Returns:
        Configured SimulationLogger instance
    """
    log_file = None
    if output_dir:
        output_dir = Path(output_dir)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"simulation_{timestamp}.log"

    logger = SimulationLogger(
        console_level=console_level, file_path=log_file, file_level=file_level
    )

    return logger


def get_logger(name: str = "pygsquig") -> logging.Logger:
    """Get a standard logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
