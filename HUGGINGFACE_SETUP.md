# HuggingFace Model Setup

## Problem
The production server cannot download models from HuggingFace at runtime because of network restrictions.

## Solution
Models are now downloaded during Docker image build time and cached inside the image.

## Setup Instructions

### 1. Set Environment Variables

Before building the image, export these environment variables:

```bash
export HF_TOKEN="your_huggingface_token_here"
export HF_PEANUT_SEG_REPO_ID="PolovyyVitaliy/peanuts-seg-yolov8m"
export HF_PEANUT_SEG_FILE="best.pt"
export HF_PEANUT_CLS_REPO_ID="PolovyyVitaliy/peanuts-cls-yolov8s"
export HF_PEANUT_CLS_FILE="best.pt"
```

**To get your HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Copy the token value

### 2. Build and Push Image

Run the build script:

```bash
./build-and-push.sh
```

Or build manually:

```bash
docker build \
  --build-arg HF_TOKEN="${HF_TOKEN}" \
  --build-arg HF_PEANUT_SEG_REPO_ID="${HF_PEANUT_SEG_REPO_ID}" \
  --build-arg HF_PEANUT_SEG_FILE="${HF_PEANUT_SEG_FILE}" \
  --build-arg HF_PEANUT_CLS_REPO_ID="${HF_PEANUT_CLS_REPO_ID}" \
  --build-arg HF_PEANUT_CLS_FILE="${HF_PEANUT_CLS_FILE}" \
  -t 10.11.122.100:5000/mlbox:latest \
  .

docker push 10.11.122.100:5000/mlbox:latest
```

### 3. Deploy to Production

On the production server, run:

```bash
./deploy-mlbox.sh
```

## How It Works

1. During Docker build, the models are downloaded from HuggingFace and cached in `~/.cache/huggingface/` inside the container
2. At runtime, when the code calls `hf_hub_download()`, it finds the models in the local cache and doesn't need internet access
3. The production server only needs to pull the Docker image from the local registry

## Troubleshooting

**Build fails with authentication error:**
- Check that your HF_TOKEN is valid and has read access
- Verify the repository IDs are correct

**Models still trying to download at runtime:**
- Verify the build completed successfully and models were downloaded
- Check Docker build logs for "Models downloaded successfully"
- Ensure environment variables match between build-time and runtime

**Can't access private models:**
- Make sure your HuggingFace token has access to the private repositories
- Check repository permissions on HuggingFace

