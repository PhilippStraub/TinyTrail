# TinyLCM Pi Zero 2W Deployment System

Complete deployment and management system for running drift detection scenarios on Raspberry Pi Zero 2W devices.

## Quick Start

```bash
# First time setup
make setup-pi PI=192.168.1.100
make deploy PI=192.168.1.100

# Create initial states
make create-state1 PI=192.168.1.100

# Run scenarios
make scenario1 PI=192.168.1.100
make scenario2 PI=192.168.1.100
```

## System Overview

### Components

- **Makefile**: Main deployment and execution interface
- **setup_pi.sh**: Pi initialization and dependency installation
- **monitor.py**: Real-time drift detection monitoring
- **pi_config.env**: Configuration constants

### Deployment Targets

| Target | Description |
|--------|-------------|
| `make deploy` | Full project deployment |
| `make deploy-quick` | Python code only (fast) |
| `make deploy-models` | Model files only |

### Execution Targets

| Target | Description |
|--------|-------------|
| `make scenario1` | Run Scenario 1 (Lego drift) |
| `make scenario2` | Run Scenario 2 (Ball drift) |
| `make create-state1/2` | Create initial KNN states |

### Monitoring Targets

| Target | Description |
|--------|-------------|
| `make monitor` | Live drift monitoring |
| `make logs` | Stream real-time logs |
| `make sync-logs` | Download logs to local |
| `make ssh` | Open SSH session |

## Setup Process

### 1. Pi Preparation

Ensure your Pi Zero 2W has:
- Raspberry Pi OS Lite (64-bit recommended)
- SSH enabled
- Camera connected and enabled
- Network connectivity

### 2. Initial Setup

```bash
# Setup Pi (installs dependencies)
make setup-pi PI=<your-pi-ip>
```

This will:
- Install Python dependencies
- Enable camera
- Create virtual environment
- Set up project directories

### 3. Deploy Code

```bash
# Full deployment
make deploy PI=<your-pi-ip>
```

### 4. Create Initial States

```bash
# For Scenario 1 (without lego in training)
make create-state1 PI=<your-pi-ip>

# For Scenario 2 (with lego in training)  
make create-state2 PI=<your-pi-ip>
```

### 5. Run Scenarios

```bash
# Scenario 1: Lego triggers drift
make scenario1 PI=<your-pi-ip>

# Scenario 2: Ball triggers drift, lego recognized
make scenario2 PI=<your-pi-ip>
```

## Multiple Pi Management

The system supports deploying to multiple Pi devices:

```bash
# Add multiple Pis
make deploy PI=192.168.1.100  # Pi 1
make deploy PI=192.168.1.101  # Pi 2

# Run scenarios on different Pis
make scenario1 PI=192.168.1.100
make scenario2 PI=192.168.1.101
```

IP addresses are stored in `.pi_hosts` for easy selection.

## Monitoring

### Real-time Monitoring

```bash
make monitor PI=<your-pi-ip>
```

Shows live:
- System stats (CPU, memory, temperature)
- Drift detection status
- Classification results
- Distance metrics
- Page-Hinkley values

### Log Management

```bash
# Stream logs in real-time
make logs PI=<your-pi-ip>

# Download logs to local machine
make sync-logs PI=<your-pi-ip>
# Logs saved to: ./pi_logs/<pi-ip>/scenario1|2/
```

## Development Workflow

### Typical Development Cycle

1. **Code in VS Code** (local development)
2. **Quick Deploy**: `make deploy-quick PI=<ip>`
3. **Test**: `make scenario1 PI=<ip>`
4. **Monitor**: `make monitor PI=<ip>`
5. **Debug**: `make logs PI=<ip>` or `make ssh PI=<ip>`

### Fast Iteration

```bash
# Only deploy Python changes
make deploy-quick PI=<ip>

# Only update models
make deploy-models PI=<ip>
```

## Troubleshooting

### Connection Issues

```bash
# Test SSH connection
make test-connection PI=<ip>
```

### Pi Setup Issues

```bash
# SSH into Pi for manual debugging
make ssh PI=<ip>

# Check logs
make logs PI=<ip>
```

### Python Environment

On the Pi, activate the environment:
```bash
source /home/pi/tinylcm_drift_detection/activate.sh
```

### Camera Issues

If camera is not working:
```bash
# On Pi, check camera
vcgencmd get_camera

# Enable camera manually
sudo raspi-config
# Interface Options -> Camera -> Enable
```

## File Structure on Pi

```
/home/pi/tinylcm_drift_detection/
├── tinylcm_python/                 # Core library
├── examples/
│   └── python_examples/
│       ├── utils/                  # Shared utilities
│       ├── scenario_lego_drift/    # Scenario 1
│       │   ├── model/             # TFLite models
│       │   ├── initial_state/     # KNN states
│       │   ├── logs/              # Runtime logs
│       │   ├── config.json
│       │   └── main.py
│       └── scenario_ball_drift/    # Scenario 2
├── venv/                          # Python virtual environment
└── activate.sh                   # Environment activation
```

## Configuration

### Scenarios

- **Scenario 1**: Lego objects trigger drift (not in training)
- **Scenario 2**: Ball objects trigger drift (lego recognized)

### Performance

- Target: ~7 FPS (142ms per frame)
- Memory: ~200MB typical usage
- CPU: ~60-80% on Pi Zero 2W

### Logging

- Automatic log rotation
- Debug frame saving optional
- Drift event logging enabled by default

## Tips

1. **Use deploy-quick** during development for faster iteration
2. **Monitor system temperature** - Pi Zero 2W can get warm
3. **Use quality camera** - Better images = better drift detection  
4. **Test with various objects** - Ensure robust detection
5. **Check logs regularly** - Early detection of issues

## Support

For issues:
1. Check logs: `make logs PI=<ip>`
2. SSH debug: `make ssh PI=<ip>`  
3. Monitor resources: `make monitor PI=<ip>`
4. Clean and redeploy: `make clean-pi && make deploy PI=<ip>`