#!/bin/bash
set -e

# Configuration
REGISTRY="10.11.122.100:5000"
IMAGE_NAME="mlbox"
TAG="latest"
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"

# HuggingFace credentials - Set these as environment variables or hardcode them
HF_TOKEN="${HF_TOKEN:-YOUR_TOKEN_HERE}"
HF_PEANUT_SEG_REPO_ID="${HF_PEANUT_SEG_REPO_ID:-PolovyyVitaliy/peanuts-seg-yolov8m}"
HF_PEANUT_SEG_FILE="${HF_PEANUT_SEG_FILE:-best.pt}"
HF_PEANUT_CLS_REPO_ID="${HF_PEANUT_CLS_REPO_ID:-PolovyyVitaliy/peanuts-cls-yolov8s}"
HF_PEANUT_CLS_FILE="${HF_PEANUT_CLS_FILE:-best.pt}"

echo "Building Docker image with HuggingFace models pre-downloaded..."
echo "Registry: ${FULL_IMAGE_NAME}"

# Build the image with build arguments for HuggingFace
docker build \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  --build-arg HF_PEANUT_SEG_REPO_ID="${HF_PEANUT_SEG_REPO_ID}" \
  --build-arg HF_PEANUT_SEG_FILE="${HF_PEANUT_SEG_FILE}" \
  --build-arg HF_PEANUT_CLS_REPO_ID="${HF_PEANUT_CLS_REPO_ID}" \
  --build-arg HF_PEANUT_CLS_FILE="${HF_PEANUT_CLS_FILE}" \
  -t "${FULL_IMAGE_NAME}" \
  .

echo "✓ Image built successfully"

echo "Pushing image to registry..."
docker push "${FULL_IMAGE_NAME}"
echo "✓ Image pushed successfully to ${FULL_IMAGE_NAME}"

