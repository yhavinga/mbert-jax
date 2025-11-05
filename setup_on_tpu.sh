#!/bin/bash
# Setup script to run on TPU VM after deployment
# Installs JAX for TPU and all project dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== mbert-jax TPU Setup ===${NC}"
echo ""

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 failed"
        exit 1
    fi
}

# 1. Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓${NC} Python $python_version (meets requirement >= $required_version)"
else
    echo -e "${RED}✗${NC} Python $python_version is too old (need >= $required_version)"
    exit 1
fi

# 2. Install venv if not available (common on minimal cloud images)
echo ""
echo -e "${YELLOW}Checking for python3-venv...${NC}"
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "Installing python3-venv package..."
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y python3.10-venv > /dev/null 2>&1
    check_status "python3-venv installed"
else
    echo -e "${GREEN}✓${NC} python3-venv is available"
fi

# 3. Create virtual environment
echo ""
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists, recreating..."
    rm -rf venv
fi
python3 -m venv venv
check_status "Virtual environment created"

# 3. Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
check_status "Virtual environment activated"

# 4. Upgrade pip
echo ""
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools > /dev/null 2>&1
check_status "pip upgraded"

# 5. Install JAX for TPU
echo ""
echo -e "${YELLOW}Installing JAX for TPU...${NC}"
echo "This may take a few minutes..."
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html > /dev/null 2>&1
check_status "JAX for TPU installed"

# 6. Install project requirements
echo ""
echo -e "${YELLOW}Installing project dependencies...${NC}"
if [ -f "requirements_tpu.txt" ]; then
    pip install -r requirements_tpu.txt > /dev/null 2>&1
    check_status "Project dependencies installed"
else
    echo -e "${RED}Warning: requirements_tpu.txt not found${NC}"
    echo "Installing basic requirements..."
    pip install -r requirements.txt > /dev/null 2>&1
    pip install flax optax datasets tqdm > /dev/null 2>&1
    check_status "Basic dependencies installed"
fi

# 7. Verify TPU availability
echo ""
echo -e "${YELLOW}Verifying TPU configuration...${NC}"
python3 -c "
import jax
import sys

devices = jax.devices()
backend = jax.default_backend()

if backend == 'tpu':
    print(f'✓ TPU detected: {len(devices)} devices')
    for i, d in enumerate(devices):
        print(f'  Device {i}: {d}')
    sys.exit(0)
else:
    print(f'✗ No TPU detected. Backend: {backend}')
    print(f'  Available devices: {devices}')
    sys.exit(1)
" || {
    echo -e "${RED}TPU verification failed!${NC}"
    echo "Continuing anyway - JAX will use CPU/GPU fallback"
}

# 8. Set up environment variables for optimal TPU performance
echo ""
echo -e "${YELLOW}Setting TPU environment variables...${NC}"
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=0  # Enable TPU logs
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Better memory management
echo -e "${GREEN}✓${NC} Environment variables set"

# 9. Create directories for outputs
echo ""
echo -e "${YELLOW}Creating output directories...${NC}"
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs
check_status "Output directories created"

# 10. Display system information
echo ""
echo -e "${BLUE}=== System Information ===${NC}"
echo -e "Hostname: ${YELLOW}$(hostname)${NC}"
echo -e "TPU Type: ${YELLOW}$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type 2>/dev/null || echo "Unknown")${NC}"
echo -e "Zone: ${YELLOW}$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | cut -d'/' -f4 || echo "Unknown")${NC}"
echo -e "Memory: ${YELLOW}$(free -h | grep Mem | awk '{print $2}')${NC}"
echo -e "Disk Space: ${YELLOW}$(df -h . | tail -1 | awk '{print $4}' ) available${NC}"

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: ${YELLOW}source venv/bin/activate${NC}"
echo "  2. Test TPU setup: ${YELLOW}python scripts/test_tpu_setup.py${NC}"
echo "  3. Start training: ${YELLOW}python scripts/train_minimal.py${NC}"
echo ""
echo "For monitoring:"
echo "  - Watch logs: ${YELLOW}tail -f logs/training.log${NC}"
echo "  - TPU utilization: ${YELLOW}gcloud compute tpus tpu-vm ssh tpu-v6 --command='top'${NC}"
echo ""

# Create activation reminder
cat > activate.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "Environment activated for TPU training"
EOF
chmod +x activate.sh

echo -e "${BLUE}Tip:${NC} Run ${YELLOW}source activate.sh${NC} to quickly activate the environment next time"