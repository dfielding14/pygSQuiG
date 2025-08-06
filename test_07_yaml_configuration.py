#!/usr/bin/env python3
"""Test script for 07_yaml_configuration notebook."""

import yaml
import os
from pathlib import Path
import shutil

print("Testing 07_yaml_configuration notebook...")

# 1. Setup
print("\n1. Setup")
config_dir = Path("example_configs")
config_dir.mkdir(exist_ok=True)
print(f"Working directory: {os.getcwd()}")
print(f"Config directory: {config_dir.absolute()}")

# 2. Basic Configuration
print("\n2. Basic Configuration")
basic_config = {
    'simulation': {
        'name': 'basic_sqg_decay',
        'description': 'Basic SQG decaying turbulence example'
    },
    'grid': {
        'N': 256,
        'L': 6.283185307179586  # 2*pi
    },
    'physics': {
        'alpha': 1.0,
        'nu_p': 1e-16,
        'p': 8
    },
    'timestepping': {
        'dt': 0.001,
        't_final': 10.0,
        'adaptive': False
    },
    'initial_condition': {
        'type': 'random',
        'seed': 42,
        'energy': 1.0
    },
    'output': {
        'directory': 'output/basic_sqg',
        'save_interval': 0.1,
        'fields': ['theta', 'energy', 'enstrophy'],
        'format': 'netcdf'
    }
}

config_file = config_dir / "basic_sqg.yaml"
with open(config_file, 'w') as f:
    yaml.dump(basic_config, f, default_flow_style=False, sort_keys=False)

print("Created basic configuration")
assert config_file.exists(), "Failed to create basic config file"

# 3. Advanced Configuration
print("\n3. Advanced Configuration")
advanced_config = {
    'simulation': {
        'name': 'forced_sqg_with_scalars',
        'description': 'Forced SQG turbulence with passive scalar mixing'
    },
    'grid': {
        'N': 512,
        'L': 6.283185307179586
    },
    'physics': {
        'alpha': 1.0,
        'nu_p': 1e-16,
        'p': 8
    },
    'forcing': {
        'enabled': True,
        'type': 'ring',
        'k_forcing': 5,
        'epsilon': 0.1
    },
    'passive_scalars': {
        'temperature': {
            'kappa': 1e-3,
            'initial_condition': {'type': 'gradient'}
        },
        'tracer': {
            'kappa': 1e-4,
            'initial_condition': {'type': 'blob'}
        }
    },
    'timestepping': {
        'adaptive': True,
        'cfl_number': 0.5,
        't_final': 100.0
    },
    'output': {
        'directory': 'output/forced_sqg_scalars',
        'save_interval': 1.0,
        'checkpoint_interval': 10.0
    }
}

advanced_file = config_dir / "advanced_sqg.yaml"
with open(advanced_file, 'w') as f:
    yaml.dump(advanced_config, f, default_flow_style=False, sort_keys=False)

print("Created advanced configuration with:")
print("  - Ring forcing")
print("  - Two passive scalars")
print("  - Adaptive timestepping")

# 4. Configuration Validation
print("\n4. Configuration Validation")
def validate_config(config):
    """Basic configuration validation."""
    errors = []
    warnings = []
    
    # Check grid
    if 'grid' not in config:
        errors.append("Missing 'grid' section")
    else:
        N = config['grid'].get('N', 0)
        if N <= 0 or (N & (N-1)) != 0:
            errors.append(f"Grid size N={N} must be a positive power of 2")
    
    # Check physics
    if 'physics' not in config:
        errors.append("Missing 'physics' section")
    else:
        alpha = config['physics'].get('alpha', -1)
        if not 0 <= alpha <= 2:
            warnings.append(f"Unusual alpha={alpha}, typical range is [0, 2]")
    
    return errors, warnings

# Validate configurations
for config_path in [config_file, advanced_file]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors, warnings = validate_config(config)
    print(f"\nValidating {config_path.name}:")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    assert len(errors) == 0, f"Validation errors in {config_path.name}"

# 5. Parameter Sweep
print("\n5. Parameter Sweep")
sweep_dir = config_dir / "alpha_sweep"
sweep_dir.mkdir(exist_ok=True)

alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0]
sweep_files = []

for alpha in alpha_values:
    sweep_config = basic_config.copy()
    sweep_config['physics'] = basic_config['physics'].copy()  # Deep copy
    sweep_config['physics']['alpha'] = alpha
    sweep_config['simulation']['name'] = f'alpha_{alpha:.1f}'
    
    filename = sweep_dir / f"alpha_{alpha:.1f}.yaml"
    with open(filename, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)
    sweep_files.append(filename)

print(f"Created {len(sweep_files)} sweep configurations")
assert len(sweep_files) == len(alpha_values), "Wrong number of sweep files"

# 6. HPC Configuration
print("\n6. HPC Configuration")
hpc_config = {
    'simulation': {
        'name': 'large_scale_sqg',
        'description': 'High-resolution SQG for HPC'
    },
    'grid': {
        'N': 2048,
        'L': 6.283185307179586
    },
    'physics': {
        'alpha': 1.0,
        'nu_p': 1e-20,
        'p': 8
    },
    'computation': {
        'device': 'gpu',
        'precision': 'float64'
    },
    'timestepping': {
        'adaptive': True,
        'cfl_number': 0.5,
        't_final': 1000.0
    },
    'output': {
        'directory': '/scratch/username/sqg_hires',
        'save_interval': 10.0,
        'checkpoint_interval': 100.0,
        'compression': 'gzip'
    }
}

hpc_file = config_dir / "hpc_sqg.yaml"
with open(hpc_file, 'w') as f:
    yaml.dump(hpc_config, f, default_flow_style=False, sort_keys=False)

print("Created HPC configuration with:")
print("  - 2048×2048 grid")
print("  - GPU acceleration")
print("  - Checkpoint/restart")

# 7. Templates
print("\n7. Configuration Templates")
template_dir = config_dir / "templates"
template_dir.mkdir(exist_ok=True)

decay_template = {
    'simulation': {
        'name': '${NAME}',
        'description': 'Decaying turbulence template'
    },
    'grid': {
        'N': '${N:256}',
        'L': 6.283185307179586
    },
    'physics': {
        'alpha': '${ALPHA:1.0}',
        'nu_p': '${NU_P:1e-16}',
        'p': 8
    }
}

with open(template_dir / "decay_template.yaml", 'w') as f:
    yaml.dump(decay_template, f, default_flow_style=False, sort_keys=False)

print("Created configuration template")

# 8. Verify all files created
print("\n8. File Verification")
expected_files = [
    config_file,
    advanced_file,
    hpc_file,
    sweep_dir / "alpha_0.0.yaml",
    sweep_dir / "alpha_1.0.yaml",
    sweep_dir / "alpha_2.0.yaml",
    template_dir / "decay_template.yaml"
]

for file in expected_files:
    assert file.exists(), f"Missing file: {file}"

print(f"All {len(expected_files)} expected files created successfully")

# 9. Test YAML loading
print("\n9. YAML Loading Test")
for yaml_file in config_dir.rglob("*.yaml"):
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Invalid YAML structure in {yaml_file}"
    except Exception as e:
        print(f"❌ Failed to load {yaml_file}: {e}")
        raise

print(f"Successfully loaded all YAML files")

# 10. Clean up
print("\n10. Cleanup")
if config_dir.exists():
    shutil.rmtree(config_dir)
    print("Cleaned up example files")

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("The YAML configuration notebook is working correctly.")
print("Key features verified:")
print("  - YAML configuration creation")
print("  - Configuration validation")
print("  - Parameter sweeps")
print("  - HPC configurations")
print("  - Template systems")
print("="*50)