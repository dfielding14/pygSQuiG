# Running Simulations

This guide covers how to run pygSQuiG simulations using the command-line interface and Python scripts.

## Command-Line Interface

### Basic Usage

Run a simulation from a YAML configuration file:

```bash
pygsquig-run config.yml
```

### Command-Line Options

```bash
pygsquig-run config.yml [OPTIONS]

Options:
  --device [cpu|gpu|tpu]  Device to run on (default: cpu)
  --checkpoint PATH       Resume from checkpoint file
  --output-dir PATH       Output directory (default: ./output)
  --dry-run              Validate configuration without running
  --log-level LEVEL      Logging level (DEBUG|INFO|WARNING|ERROR)
  --help                 Show help message
```

### Examples

#### GPU Simulation
```bash
pygsquig-run config.yml --device=gpu
```

#### Resume from Checkpoint
```bash
pygsquig-run config.yml --checkpoint=output/checkpoints/step_00001000.h5
```

#### Dry Run (Validation Only)
```bash
pygsquig-run config.yml --dry-run
```

#### Custom Output Directory
```bash
pygsquig-run config.yml --output-dir=results/run_001
```

## Output Structure

The simulation creates the following directory structure:

```
output/
├── config.yml              # Copy of configuration used
├── simulation.log          # Detailed log file
├── fields/                 # Field snapshots
│   ├── fields_00000000.nc
│   ├── fields_00000100.nc
│   └── ...
├── diagnostics/            # Time series data
│   └── timeseries.h5
└── checkpoints/            # Restart files
    ├── step_00001000.h5
    ├── step_00002000.h5
    └── final_step_00010000.h5
```

## Output Files

### Field Files (NetCDF)

Field snapshots contain:
- `theta`: Active scalar (buoyancy) field
- `u`, `v`: Velocity components (if requested)
- `scalar_{name}`: Passive scalar fields (if enabled)
- `time`: Simulation time
- Metadata: step number, parameters

Access with xarray:
```python
import xarray as xr

ds = xr.open_dataset("output/fields/fields_00001000.nc")
theta = ds.theta.values
time = ds.time.item()
```

### Diagnostics (HDF5)

Time series of integrated quantities:
- `time`: Time array
- `energy`: Total energy
- `enstrophy`: Total enstrophy  
- `theta_rms`: RMS of active scalar
- `theta_max`: Maximum of active scalar
- `{scalar}_mean`: Mean of each passive scalar
- `{scalar}_variance`: Variance of each passive scalar

Access with h5py:
```python
import h5py

with h5py.File("output/diagnostics/timeseries.h5", "r") as f:
    time = f["time"][:]
    energy = f["energy"][:]
```

### Checkpoints (HDF5)

Complete state for restart:
- Full spectral fields
- Simulation time and step
- Configuration used

## Monitoring Progress

### Console Output

During the run, progress is logged:
```
2024-01-15 10:30:15 | INFO | Starting simulation from t=0.00 to t=100.00
2024-01-15 10:30:16 | INFO | t=   1.000 | step=     100 | dt=1.00e-02 | E=1.234 | Z=5.678 | CFL=0.45 | ETA: 0:15:23
2024-01-15 10:30:17 | INFO | t=   2.000 | step=     200 | dt=1.00e-02 | E=1.235 | Z=5.679 | CFL=0.46 | ETA: 0:14:55
```

### Log File

Detailed logging saved to `output/simulation.log`:
- All console output
- Additional debug information
- Warnings and errors
- Performance metrics

### Real-Time Monitoring

Monitor a running simulation:
```bash
# Watch latest output
watch -n 1 "ls -lht output/fields/ | head -5"

# Monitor diagnostics
python -c "
import h5py
import numpy as np
with h5py.File('output/diagnostics/timeseries.h5', 'r') as f:
    t = f['time'][-1]
    E = f['energy'][-1]
    print(f't={t:.2f}, E={E:.3f}')
"
```

## Graceful Shutdown

### Ctrl+C Handling

The simulation can be stopped gracefully with Ctrl+C:
1. Press Ctrl+C once
2. Current step completes
3. Checkpoint is saved
4. Output files are finalized

```
^C
2024-01-15 10:35:42 | WARNING | Shutdown requested. Saving checkpoint before exiting...
2024-01-15 10:35:43 | INFO | Checkpoint saved: step_00005432.h5
2024-01-15 10:35:43 | INFO | Simulation stopped at t=54.32
```

### Wall Time Limit

Set a maximum wall time in the configuration:
```yaml
simulation:
  wall_time_limit: 3600  # 1 hour in seconds
```

The simulation will checkpoint and exit cleanly when approaching the limit.

## Performance Tips

### Optimal Settings

1. **Adaptive Timestepping**: Enable for efficiency
   ```yaml
   solver:
     time_integration:
       adaptive_cfl: true
       cfl_safety: 0.8
   ```

2. **Output Frequency**: Balance data needs with I/O cost
   ```yaml
   simulation:
     output_interval: 10.0      # Save fields every 10 time units
     checkpoint_interval: 100.0  # Checkpoint every 100 time units
   ```

3. **Field Selection**: Only save needed fields
   ```yaml
   output:
     fields: [theta]  # Don't save velocity if not needed
   ```

### GPU Acceleration

For GPU runs:
```bash
# Check GPU availability
python -c "import jax; print(jax.devices())"

# Run on specific GPU
export CUDA_VISIBLE_DEVICES=0
pygsquig-run config.yml --device=gpu
```

### Memory Management

For large simulations:
```bash
# Limit GPU memory usage
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Use less memory-intensive settings
pygsquig-run large_sim.yml --device=gpu
```

## Batch Runs

### Sequential Runs

```bash
#!/bin/bash
for alpha in 0.5 1.0 1.5 2.0; do
    sed "s/alpha: 1.0/alpha: $alpha/" base_config.yml > config_$alpha.yml
    pygsquig-run config_$alpha.yml --output-dir=results/alpha_$alpha
done
```

### Parallel Runs

Using GNU Parallel:
```bash
parallel -j 4 pygsquig-run config_{}.yml --output-dir=results/run_{} ::: 1 2 3 4
```

## Troubleshooting

### Common Issues

1. **Simulation becomes unstable (NaN values)**
   - Reduce timestep or enable adaptive timestepping
   - Increase dissipation (`nu_p`)
   - Check forcing parameters

2. **Out of memory**
   - Reduce grid resolution
   - Save fields less frequently
   - Use `--device=cpu` for large grids

3. **Slow performance**
   - Enable JIT compilation (automatic)
   - Use GPU acceleration
   - Reduce output frequency

### Debug Mode

Enable detailed debugging:
```bash
pygsquig-run config.yml --log-level=DEBUG
```

### Validation

Check configuration without running:
```bash
pygsquig-run config.yml --dry-run
```

## Post-Processing

### Quick Visualization

```python
import xarray as xr
import matplotlib.pyplot as plt

# Load final field
ds = xr.open_dataset("output/fields/fields_final_00010000.nc")

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(ds.x, ds.y, ds.theta, cmap='RdBu_r')
plt.colorbar(label='θ')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Active scalar at t={ds.time.item():.1f}')
plt.show()
```

### Analyze Time Series

```python
import h5py
import matplotlib.pyplot as plt

with h5py.File("output/diagnostics/timeseries.h5", "r") as f:
    time = f["time"][:]
    energy = f["energy"][:]
    enstrophy = f["enstrophy"][:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1.plot(time, energy)
ax1.set_ylabel('Energy')

ax2.plot(time, enstrophy)
ax2.set_xlabel('Time')
ax2.set_ylabel('Enstrophy')

plt.tight_layout()
plt.show()
```

## Advanced Usage

### Custom Analysis During Run

Create a wrapper script:
```python
import subprocess
import time
import h5py

# Start simulation
proc = subprocess.Popen(['pygsquig-run', 'config.yml'])

# Monitor while running
while proc.poll() is None:
    time.sleep(10)
    try:
        with h5py.File('output/diagnostics/timeseries.h5', 'r') as f:
            current_time = f['time'][-1]
            print(f"Current simulation time: {current_time:.2f}")
    except:
        pass

print("Simulation complete!")
```

### Automated Restart

For long runs with job time limits:
```bash
#!/bin/bash
# restart_simulation.sh

CONFIG="long_run.yml"
OUTPUT_DIR="output"
MAX_RESTARTS=10

for i in $(seq 1 $MAX_RESTARTS); do
    echo "Run $i starting..."
    
    # Find latest checkpoint
    CHECKPOINT=$(ls -t $OUTPUT_DIR/checkpoints/step_*.h5 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT" ]; then
        # First run
        pygsquig-run $CONFIG --output-dir=$OUTPUT_DIR
    else
        # Restart from checkpoint
        pygsquig-run $CONFIG --checkpoint=$CHECKPOINT --output-dir=$OUTPUT_DIR
    fi
    
    # Check if we reached t_end
    if [ -f "$OUTPUT_DIR/checkpoints/final_step_*.h5" ]; then
        echo "Simulation complete!"
        break
    fi
done
```