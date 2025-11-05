#!/bin/bash
# Fast deployment script for TPU v6 spot instance
# Syncs project code while excluding large/unnecessary files

set -e  # Exit on error

# Configuration
TPU_HOST="tpu-v6"
REMOTE_DIR="~/mbert-jax"
LOCAL_DIR="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== mbert-jax TPU Deployment ===${NC}"
echo -e "Deploying to: ${YELLOW}${TPU_HOST}:${REMOTE_DIR}${NC}"
echo ""

# Check if TPU host is reachable
echo -n "Checking TPU connectivity... "
if ssh -o ConnectTimeout=5 "${TPU_HOST}" "echo 'connected'" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo -e "${RED}Error: Cannot connect to ${TPU_HOST}${NC}"
    echo "Please ensure:"
    echo "  1. TPU VM is running"
    echo "  2. SSH config has 'Host tpu-v6' configured"
    echo "  3. You have network connectivity"
    exit 1
fi

# Create remote directory if it doesn't exist
echo -n "Creating remote directory... "
ssh "${TPU_HOST}" "mkdir -p ${REMOTE_DIR}"
echo -e "${GREEN}✓${NC}"

# Sync files with rsync
echo ""
echo "Syncing project files..."
echo "Excluding: venv/, cache files, large artifacts"
echo ""

rsync -avz --progress --delete \
    --exclude='venv/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='.cache/' \
    --exclude='artifacts/reference-configs/*.npz' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='.git/' \
    --exclude='*.egg-info/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.pth' \
    --exclude='*.pkl' \
    --exclude='*.h5' \
    --exclude='*.safetensors' \
    --exclude='wandb/' \
    --exclude='.ipynb_checkpoints/' \
    "${LOCAL_DIR}/" "${TPU_HOST}:${REMOTE_DIR}/"

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Next steps:"
echo "  1. SSH to TPU: ${YELLOW}ssh ${TPU_HOST}${NC}"
echo "  2. Navigate to: ${YELLOW}cd ${REMOTE_DIR}${NC}"
echo "  3. Run setup: ${YELLOW}./setup_on_tpu.sh${NC}"
echo "  4. Test TPU: ${YELLOW}python scripts/test_tpu_setup.py${NC}"
echo "  5. Start training: ${YELLOW}python scripts/train_minimal.py${NC}"
echo ""

# Optional: Show remote disk usage
echo "Remote disk usage:"
ssh "${TPU_HOST}" "cd ${REMOTE_DIR} && du -sh ."
echo ""

# Optional: Automatically run setup if flag is passed
if [[ "$1" == "--setup" ]]; then
    echo -e "${YELLOW}Running remote setup...${NC}"
    ssh "${TPU_HOST}" "cd ${REMOTE_DIR} && chmod +x setup_on_tpu.sh && ./setup_on_tpu.sh"
fi