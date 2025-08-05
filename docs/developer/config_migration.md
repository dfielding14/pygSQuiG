# Configuration System Migration Guide

## Overview

pygSQuiG now offers a simplified configuration system alongside the original dataclass-based system. The new system provides the same functionality with less complexity and more intuitive usage.

## Key Improvements

1. **Single Config class** instead of 9 separate dataclasses
2. **Simpler validation** using a schema-based approach
3. **Same YAML structure** - existing config files work unchanged
4. **Automatic inference** - e.g., forcing is auto-enabled when parameters are provided
5. **Dot notation access** - `config.get('grid.N')` instead of `config.grid.N`

## Using the Simplified System

### Basic Usage

```python
from pygsquig.io.simple_config import Config

# Create with defaults
config = Config()

# Create with custom values
config = Config({
    'grid': {'N': 512, 'L': 10.0},
    'solver': {'alpha': 0.5}
})

# Load from YAML
config = Config.from_yaml('config.yml')

# Access values
N = config.get('grid.N')
alpha = config.get('solver.alpha')

# Modify values
config.set('solver.nu_p', 1e-12)

# Save to YAML
config.to_yaml('new_config.yml')
```

### Backward Compatibility

For code expecting the old dataclass system:

```python
from pygsquig.io.config_adapter import adapt_config

# Create simple config
config = Config.from_yaml('config.yml')

# Adapt to old interface
adapted = adapt_config(config)

# Now use like the old system
N = adapted.grid.N
alpha = adapted.solver.alpha
if adapted.forcing:
    epsilon = adapted.forcing.epsilon
```

## YAML Structure (Unchanged)

The YAML structure remains the same:

```yaml
grid:
  N: 256
  L: 6.283185307179586

solver:
  alpha: 1.0
  nu_p: 1.0e-12
  p: 8

forcing:
  kf: 30.0
  epsilon: 0.5
  # Note: 'enabled' is auto-set to True when parameters provided

time_integration:
  method: RK4
  adaptive: true
  cfl_safety: 0.8

simulation:
  t_end: 100.0
  output_interval: 1.0
```

## Migration Tips

1. **No code changes needed** - The adapter makes the new system compatible with existing code
2. **Gradual migration** - You can migrate module by module
3. **Simpler tests** - New config is easier to test and mock

## Comparison

### Old System (342 lines)
```python
# Complex nested dataclasses
@dataclass
class GridConfig:
    N: int = 256
    L: float = 2 * np.pi
    
    def __post_init__(self):
        if self.N <= 0 or self.N % 2 != 0:
            raise ValueError(...)
            
# ... 8 more dataclasses ...

# Complex deserialization
def from_dict(cls, data: dict):
    grid = GridConfig(**data.get("grid", {}))
    solver_data = data.get("solver", {})
    if "dissipation" in solver_data:
        solver_data["dissipation"] = DissipationConfig(...)
    # ... etc
```

### New System (150 lines)
```python
# Single config class with validation rules
DEFAULT_CONFIG = {...}  # Simple nested dict
VALIDATION_RULES = {
    'grid.N': lambda x: x > 0 and x % 2 == 0,
    'solver.alpha': lambda x: -2 <= x <= 2,
    # ... etc
}

class Config:
    def __init__(self, config_dict=None):
        self._config = merge_with_defaults(config_dict)
        self._validate()
```

## Benefits

- **50% less code** to maintain
- **Easier to extend** - just add to defaults and validation rules
- **Better error messages** - all validation errors shown at once
- **More flexible** - can add custom fields without changing classes
- **Auto-inference** - smart defaults based on what's provided