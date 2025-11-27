#!/bin/bash
# Copy PaddleX models from dev environment to project for Docker packaging
# Run this once on dev server after models are downloaded

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$PROJECT_ROOT/assets/models/paddlex"
SOURCE_DIR="$HOME/.paddlex/official_models/PP-DocLayout_plus-L"

echo "Copying PaddleX models to project directory..."
echo "Source: $SOURCE_DIR"
echo "Destination: $MODELS_DIR/PP-DocLayout_plus-L/"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    echo "Please run LabelGuard once to download the models first."
    exit 1
fi

# Create models directory
mkdir -p "$MODELS_DIR"

# Copy model files (excluding .cache if you want, but including it is safer)
echo "Copying files..."
rsync -av "$SOURCE_DIR/" "$MODELS_DIR/PP-DocLayout_plus-L/"

echo ""
echo "✅ Models copied successfully!"
echo "Total size: $(du -sh "$MODELS_DIR" | cut -f1)"
echo ""
echo "Files ready for Docker build. Next: update Dockerfile to copy these models."


