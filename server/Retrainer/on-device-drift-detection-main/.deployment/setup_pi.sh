#!/bin/bash
# Setup script for Raspberry Pi Zero 2W
# Installs dependencies for TinyLCM drift detection

set -e

echo "========================================="
echo "TinyLCM Pi Zero 2W Setup Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored status
status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on Raspberry Pi
if [ ! -f /proc/cpuinfo ] || ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    warning "Not running on a Raspberry Pi, continuing anyway..."
fi

# Update system packages
status "Updating package lists..."
sudo apt-get update || warning "Failed to update package lists"

# Install system dependencies
status "Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-opencv \
    python3-sklearn \
    python3-psutil \
    libatlas-base-dev \
    libopenjp2-7 \
    libtiff5 \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libjasper1 \
    libilmbase23 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec-extra58 \
    libavformat58 \
    libswscale5 \
    git \
    htop \
    2>/dev/null || warning "Some system packages might be missing"

# Camera-related packages (prefer apt over pip for picamera2/libcamera on Pi OS)
status "Installing camera libraries (apt)..."
sudo apt-get install -y libcamera-apps python3-libcamera python3-picamera2 2>/dev/null || \
  warning "libcamera/picamera2 apt packages not available; ensure camera support manually"

# Configure camera stack (libcamera) based on OS
status "Checking camera configuration..."

# Detect correct config file path (Bookworm uses /boot/firmware)
CONFIG_FILE="/boot/config.txt"
if [ -f /boot/firmware/config.txt ]; then
  CONFIG_FILE="/boot/firmware/config.txt"
fi

# Disable legacy stack if enabled and favor libcamera
if grep -q "^start_x=1" "$CONFIG_FILE"; then
  warning "Legacy camera stack detected; disabling for libcamera compatibility"
  sudo sed -i 's/^start_x=1/# start_x disabled for libcamera/' "$CONFIG_FILE" || true
fi

# Ensure camera auto-detect is enabled (harmless if already present)
if ! grep -q "^camera_auto_detect=1" "$CONFIG_FILE"; then
  echo "camera_auto_detect=1" | sudo tee -a "$CONFIG_FILE" >/dev/null || true
fi

# Ensure user is in video group for camera access
if ! groups $USER | grep -q "video"; then
  status "Adding $USER to 'video' group (reboot required)"
  sudo usermod -aG video "$USER" || true
fi

# Check Python version
status "Checking Python version..."
python3 --version

# Project path setup
PROJECT_PATH="/home/pi/tinylcm_drift_detection"

# NO VIRTUAL ENVIRONMENT - Use system Python directly
status "Using system Python (no virtual environment; apt-managed per PEP 668)"

# Install Python modules via APT (avoid pip on system Python)
status "Installing Python modules via apt..."
sudo apt-get install -y \
  python3-pil \
  python3-tflite-runtime \
  2>/dev/null || warning "Some Python modules via apt may be unavailable (check python3-tflite-runtime)"

if ! python3 -c "import tflite_runtime" 2>/dev/null; then
  warning "python3-tflite-runtime not available via apt. Options:"
  echo "  - Use a virtualenv: python3 -m venv ~/venv && source ~/venv/bin/activate && pip install tflite-runtime"
  echo "  - Or use pip with --break-system-packages (not recommended)"
fi

# No pip fallbacks to honor PEP 668 and keep system clean

# Create necessary directories
status "Creating project directories..."
mkdir -p "${PROJECT_PATH}/logs"
mkdir -p "${PROJECT_PATH}/debug"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_lego_drift/logs"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_lego_drift/debug"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_lego_drift/initial_state"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_ball_drift/logs"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_ball_drift/debug"
mkdir -p "${PROJECT_PATH}/examples/python_examples/scenario_ball_drift/initial_state"

# Set permissions
status "Setting permissions..."
chmod -R 755 "${PROJECT_PATH}"

# Create environment setup script (no venv needed)
status "Creating environment setup script..."
cat > "${PROJECT_PATH}/activate.sh" << 'EOF'
#!/bin/bash
# Set up TinyLCM environment (system Python)
export PYTHONPATH=/home/pi/tinylcm_drift_detection:$PYTHONPATH
echo "TinyLCM environment set up (system Python)"
echo "Python: $(which python3)"
echo "PYTHONPATH: $PYTHONPATH"
EOF
chmod +x "${PROJECT_PATH}/activate.sh"

# Test imports
status "Testing Python imports..."
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" || error "NumPy import failed"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" || error "OpenCV import failed"
python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" || error "scikit-learn import failed"
python3 -c "import psutil; print(f'psutil: {psutil.__version__}')" || error "psutil import failed"

# Try to import tflite_runtime
python3 -c "
try:
    import tflite_runtime.interpreter as tflite
    print('TFLite runtime: OK')
except ImportError:
    print('TFLite runtime: Not available (will need to install manually)')
" || warning "TFLite runtime not available"

echo ""
echo "========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. If camera was enabled, reboot the Pi: sudo reboot"
echo "2. Deploy the project: make deploy"
echo "3. Create initial states: make create-state1"
echo "4. Run scenarios: make scenario1"
echo ""
echo "To set up environment variables manually:"
echo "  source ${PROJECT_PATH}/activate.sh"
echo ""
echo "Note: Using system Python (no virtual environment)"
echo ""
