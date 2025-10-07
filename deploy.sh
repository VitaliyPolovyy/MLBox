#!/bin/bash

# Deploy script for mlbox with local registry
# This script generates requirements.txt from Poetry, builds Docker image, and manages local registry

set -e  # Exit on any error

# Configuration
REGISTRY_HOST="localhost:5000"
MAX_VERSIONS=3

echo "🚀 Starting deployment process..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first."
    echo "   Visit: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    echo "   Try: sudo systemctl start docker"
    echo "   Or: sudo service docker start"
    exit 1
fi

echo "📦 Generating requirements.txt from Poetry..."
# Export dependencies from Poetry to requirements.txt
poetry run pip freeze > requirements.txt

# Remove editable install lines that Docker can't handle
echo "🔧 Cleaning requirements.txt for Docker..."
grep -v "^-e " requirements.txt > requirements_clean.txt
mv requirements_clean.txt requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Successfully generated requirements.txt"
else
    echo "❌ Failed to generate requirements.txt"
    exit 1
fi

echo "🐳 Building Docker image..."
# Build Docker image
docker build -t mlbox:latest .

if [ $? -eq 0 ]; then
    echo "✅ Successfully built Docker image: mlbox:latest"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Registry management
echo "📤 Managing local registry..."

# Start registry if not running
if ! docker ps | grep -q local-registry; then
    echo "⚠️  Starting local registry..."
    docker run -d --name local-registry -p 5000:5000 -v registry-data:/var/lib/registry registry:2
    sleep 3
fi

# Get creation date and time (more robust)
CREATION_DATE=$(docker inspect mlbox:latest --format='{{.Created}}' | sed 's/T.*//')
CREATION_TIME=$(docker inspect mlbox:latest --format='{{.Created}}' | sed 's/.*T//' | cut -d'.' -f1 | sed 's/:/_/g')
TAG_NAME="${CREATION_DATE}_${CREATION_TIME}"
echo "📅 Tagging with date and time: $TAG_NAME"

# Tag and push with date and time
docker tag mlbox:latest $REGISTRY_HOST/mlbox:$TAG_NAME
docker push $REGISTRY_HOST/mlbox:$TAG_NAME

# Tag and push latest
docker tag mlbox:latest $REGISTRY_HOST/mlbox:latest
docker push $REGISTRY_HOST/mlbox:latest

# Simple cleanup (remove oldest if too many)
VERSION_COUNT=$(docker images $REGISTRY_HOST/mlbox --format "{{.Tag}}" | grep -v latest | wc -l)
if [ $VERSION_COUNT -gt $MAX_VERSIONS ]; then
    OLDEST=$(docker images $REGISTRY_HOST/mlbox --format "{{.Tag}}" | grep -v latest | sort | head -1)
    echo "🗑️  Removing oldest: $OLDEST"
    docker rmi $REGISTRY_HOST/mlbox:$OLDEST
fi

echo "✅ Registry updated successfully!"
echo "🎉 Deployment completed successfully!"

echo ""
echo "📋 Next steps:"
echo "   • Run locally: docker run -p 8000:8000 mlbox:latest"
echo "   • Pull from registry: docker pull $REGISTRY_HOST/mlbox:latest"
echo "   • Run from registry: docker run -p 8000:8000 $REGISTRY_HOST/mlbox:latest"
echo "   • List registry versions: docker images $REGISTRY_HOST/mlbox"
