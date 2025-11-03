#!/bin/bash

# TPU v6e Setup Script
# Available configurations based on your quota: 64 spot Cloud TPU v6e chips in europe-west4-a

# Project configuration
export GCP_PROJECT="t5-dutch-english"
export GCP_ZONE="europe-west4-a"
export STORAGE_BUCKET="gs://t5-dutch-english"

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TPU v6e Configuration Setup${NC}"
echo "================================"
echo ""
echo "Available TPU v6e configurations:"
echo "  1. v6e-1  - Single chip (testing/development)"
echo "  2. v6e-4  - 4 chips (small workloads)"
echo "  3. v6e-8  - 8 chips (optimized for inference)"
echo "  4. v6e-16 - 16 chips (medium workloads)"
echo "  5. v6e-32 - 32 chips (large workloads)"
echo "  6. v6e-64 - 64 chips (maximum available in your quota)"
echo ""
echo -n "Select configuration (1-6): "
read selection

# Set accelerator type based on selection
case $selection in
    1)
        ACCELERATOR_TYPE="v6e-1"
        CHIPS="1"
        ;;
    2)
        ACCELERATOR_TYPE="v6e-4"
        CHIPS="4"
        ;;
    3)
        ACCELERATOR_TYPE="v6e-8"
        CHIPS="8"
        ;;
    4)
        ACCELERATOR_TYPE="v6e-16"
        CHIPS="16"
        ;;
    5)
        ACCELERATOR_TYPE="v6e-32"
        CHIPS="32"
        ;;
    6)
        ACCELERATOR_TYPE="v6e-64"
        CHIPS="64"
        ;;
    *)
        echo -e "${RED}Invalid selection. Exiting.${NC}"
        exit 1
        ;;
esac

# TPU name with configuration suffix
export GCP_TPU_NAME="tpu-v6e-${CHIPS}"

echo ""
echo -e "${YELLOW}Configuration selected:${NC}"
echo "  Project: $GCP_PROJECT"
echo "  Zone: $GCP_ZONE"
echo "  TPU Name: $GCP_TPU_NAME"
echo "  Accelerator: $ACCELERATOR_TYPE ($CHIPS chips)"
echo "  Type: Spot instance"
echo ""

echo -n "Do you want to create a queued resource? (y/n): "
read create_queued

if [[ "$create_queued" == "y" || "$create_queued" == "Y" ]]; then
    echo ""
    echo -e "${GREEN}Creating queued resource for TPU v6e...${NC}"

    # Create a queued resource for TPU v6e
    gcloud compute tpus queued-resources create ${GCP_TPU_NAME}-queued \
        --node-id ${GCP_TPU_NAME} \
        --zone ${GCP_ZONE} \
        --project ${GCP_PROJECT} \
        --accelerator-type ${ACCELERATOR_TYPE} \
        --runtime-version v2-alpha-tpuv6e \
        --spot

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Queued resource created successfully!${NC}"
        echo ""
        echo "Commands for managing your TPU:"
        echo ""
        echo "# List all queued resources:"
        echo "gcloud compute tpus queued-resources list --project ${GCP_PROJECT} --zone ${GCP_ZONE}"
        echo ""
        echo "# Check status of your queued resource:"
        echo "gcloud compute tpus queued-resources describe ${GCP_TPU_NAME}-queued --zone ${GCP_ZONE} --project ${GCP_PROJECT}"
        echo ""
        echo "# Delete queued resource (if needed):"
        echo "gcloud compute tpus queued-resources delete ${GCP_TPU_NAME}-queued --zone ${GCP_ZONE} --project ${GCP_PROJECT}"
        echo ""
        echo "# Once the TPU is created, SSH into it:"
        echo "gcloud compute tpus tpu-vm ssh ${GCP_TPU_NAME} --zone ${GCP_ZONE} --project ${GCP_PROJECT}"
        echo ""
        echo "# Delete the TPU VM (when done):"
        echo "gcloud compute tpus tpu-vm delete ${GCP_TPU_NAME} --zone ${GCP_ZONE} --project ${GCP_PROJECT}"
    else
        echo -e "${RED}Failed to create queued resource.${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${YELLOW}Alternative: Create TPU v6e VM directly (if quota available)${NC}"
    echo ""
    echo "Run this command to create the TPU VM directly:"
    echo ""
    echo "gcloud compute tpus tpu-vm create ${GCP_TPU_NAME} \\"
    echo "    --zone ${GCP_ZONE} \\"
    echo "    --project ${GCP_PROJECT} \\"
    echo "    --accelerator-type ${ACCELERATOR_TYPE} \\"
    echo "    --version v2-alpha-tpuv6e \\"
    echo "    --spot"
fi

echo ""
echo -e "${GREEN}TPU v6e Specifications:${NC}"
echo "• 4.7X peak performance vs TPU v5e"
echo "• 32 GB HBM per chip ($(($CHIPS * 32)) GB total for your configuration)"
echo "• Optimized for transformer models and inference workloads"
echo "• Doubled HBM capacity/bandwidth and ICI bandwidth vs v5e"