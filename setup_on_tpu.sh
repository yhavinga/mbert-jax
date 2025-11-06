#!/bin/bash
# Setup script to run on TPU VM after deployment
# Installs JAX for TPU and all project dependencies
#
# Usage: ./setup_on_tpu.sh [--force]
#   --force: Force reinstall all dependencies even if already installed

set -e  # Exit on error

# Parse arguments
FORCE_REINSTALL=false
if [[ "$1" == "--force" ]]; then
    FORCE_REINSTALL=true
    echo "Force reinstall mode enabled"
fi

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

# 1. Check/Install Python 3.12
echo -e "${YELLOW}Checking Python version...${NC}"

# Check if python3.12 is available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
    python_version=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    echo -e "${GREEN}✓${NC} Python $python_version available"
else
    echo -e "${YELLOW}Python 3.12 not found. Installing...${NC}"
    # Add deadsnakes PPA for newer Python versions
    sudo add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
    sudo apt-get update > /dev/null 2>&1
    sudo apt-get install -y python3.12 python3.12-venv python3.12-distutils > /dev/null 2>&1

    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        python_version=$($PYTHON_CMD --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        echo -e "${GREEN}✓${NC} Python $python_version installed successfully"
    else
        echo -e "${RED}✗${NC} Failed to install Python 3.12"
        exit 1
    fi
fi

# 2. Create or verify virtual environment
echo ""
echo -e "${YELLOW}Setting up virtual environment...${NC}"
RECREATE_VENV=false

# Force reinstall if requested
if [ "$FORCE_REINSTALL" = true ]; then
    echo "Force reinstall requested, will recreate virtual environment"
    RECREATE_VENV=true
# Check if venv exists and is functional
elif [ -d "venv" ]; then
    if venv/bin/python -c "import sys; sys.exit(0)" 2>/dev/null; then
        venv_python_version=$(venv/bin/python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        echo "Virtual environment exists with Python $venv_python_version"
        # Check if venv has the right Python version
        if [ "$venv_python_version" != "$python_version" ]; then
            echo "Virtual environment has Python $venv_python_version but we need Python $python_version"
            RECREATE_VENV=true
        # Check if JAX is already installed for TPU
        elif venv/bin/python -c "import jax; import sys; backend = jax.default_backend(); sys.exit(0 if backend == 'tpu' else 1)" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} JAX for TPU already configured"
        else
            echo "JAX not configured for TPU, will reinstall..."
            RECREATE_VENV=true
        fi
    else
        echo "Virtual environment is corrupted, recreating..."
        RECREATE_VENV=true
    fi
else
    RECREATE_VENV=true
fi

if [ "$RECREATE_VENV" = true ]; then
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    $PYTHON_CMD -m venv venv
    check_status "Virtual environment created with Python $python_version"
else
    echo -e "${GREEN}✓${NC} Using existing virtual environment"
fi

# 3. Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
check_status "Virtual environment activated"

# 4. Upgrade pip (if needed)
echo ""
echo -e "${YELLOW}Checking pip...${NC}"
if [ "$RECREATE_VENV" = true ]; then
    pip install --upgrade pip wheel setuptools > /dev/null 2>&1
    check_status "pip upgraded"
else
    echo -e "${GREEN}✓${NC} pip already configured"
fi

# 5. Install JAX for TPU (if needed)
echo ""
echo -e "${YELLOW}Checking JAX for TPU...${NC}"
if [ "$RECREATE_VENV" = true ] || ! python -c "import jax; import sys; sys.exit(0 if jax.default_backend() == 'tpu' else 1)" 2>/dev/null; then
    echo "Installing JAX for TPU..."
    echo "This may take a few minutes..."
    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html > /dev/null 2>&1
    check_status "JAX for TPU installed"
else
    echo -e "${GREEN}✓${NC} JAX for TPU already installed"
fi

# 6. Install or verify project requirements
echo ""
echo -e "${YELLOW}Checking project dependencies...${NC}"
if [ -f "requirements_tpu.txt" ]; then
    # Check if key packages are installed
    NEED_INSTALL=false
    if [ "$RECREATE_VENV" = true ]; then
        NEED_INSTALL=true
    else
        # Check for key packages
        if ! python -c "import flax, optax, transformers, datasets" 2>/dev/null; then
            NEED_INSTALL=true
        fi
    fi

    if [ "$NEED_INSTALL" = true ]; then
        echo "Installing project dependencies..."
        pip install -r requirements_tpu.txt > /dev/null 2>&1
        check_status "Project dependencies installed"
    else
        echo -e "${GREEN}✓${NC} Project dependencies already installed"
    fi
else
    echo -e "${RED}Warning: requirements_tpu.txt not found${NC}"
    echo "Installing basic requirements..."
    pip install -r requirements.txt > /dev/null 2>&1
    pip install flax optax datasets tqdm wandb > /dev/null 2>&1
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

# 10. Verify authentication tokens
echo ""
echo -e "${YELLOW}Verifying authentication tokens...${NC}"

# Check HuggingFace token
if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo -e "${GREEN}✓${NC} HuggingFace token found"
    # Try to validate token by checking it can be read
    if python3 -c "
import sys
try:
    from huggingface_hub import HfApi
    api = HfApi()
    user = api.whoami()
    print(f'  Authenticated as: {user[\"name\"]}')
except Exception as e:
    print(f'  Token validation skipped: {str(e)}')
" 2>/dev/null; then
        :
    fi
else
    echo -e "${YELLOW}⚠${NC} No HuggingFace token found"
    echo "  To add: huggingface-cli login (on local machine, then redeploy)"
fi

# Check Wandb credentials
if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Wandb credentials found in .netrc"
elif [ -d "$HOME/.config/wandb" ]; then
    echo -e "${GREEN}✓${NC} Wandb config directory found"
else
    echo -e "${YELLOW}⚠${NC} No Wandb credentials found"
    echo "  To add: wandb login (on local machine, then redeploy)"
fi

# 11. Display system information
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

# Create activation reminder with token paths
cat > activate.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export JAX_PLATFORMS=tpu
export TF_CPP_MIN_LOG_LEVEL=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Set HuggingFace token path if it exists
if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
    export HUGGING_FACE_HUB_TOKEN=$(cat "$HF_TOKEN_PATH" 2>/dev/null || echo "")
fi

echo "Environment activated for TPU training"
echo "TPU backend: $(python -c 'import jax; print(jax.default_backend())' 2>/dev/null || echo 'Not detected')"
echo "TPU devices: $(python -c 'import jax; print(len(jax.devices()))' 2>/dev/null || echo '0')"
EOF
chmod +x activate.sh

echo -e "${BLUE}Tip:${NC} Run ${YELLOW}source activate.sh${NC} to quickly activate the environment next time"