#!/bin/bash

# Deploy script for mlbox with local registry
# This script generates requirements.txt from Poetry, builds Docker image, and manages local registry

set -e  # Exit on any error

# Configuration
REGISTRY_HOST="localhost:5000"
MAX_VERSIONS=1  # Keep only the newest timestamped version (plus 'latest' = 2 tags total)

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

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "❌ jq is not installed. Please install jq first."
    echo "   Ubuntu/Debian: sudo apt-get install jq"
    echo "   macOS: brew install jq"
    exit 1
fi

echo "📦 Generating requirements.txt from Poetry..."
if poetry export -f requirements.txt --output requirements.txt --without-hashes 2>/dev/null; then
    echo "✅ Successfully generated requirements.txt"
else
    echo "⚠️  poetry export failed, falling back to pip freeze..."
    poetry run pip freeze > requirements.txt
    echo "🔧 Cleaning requirements.txt for Docker..."
    grep -v "^[-e]" requirements.txt | grep -v "^file://" > requirements_clean.txt
    mv requirements_clean.txt requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Successfully generated requirements.txt"
    else
        echo "❌ Failed to generate requirements.txt"
        exit 1
    fi
fi

# Registry management
echo "📤 Managing local registry..."

if ! docker ps | grep -q local-registry; then
    echo "⚠️  Starting local registry with deletion enabled..."
    docker run -d --name local-registry --restart=unless-stopped -p 5000:5000 \
        -e REGISTRY_STORAGE_DELETE_ENABLED=true \
        -v registry-data:/var/lib/registry \
        registry:2
    sleep 3
else
    # Check if deletion is enabled, if not, recreate the container
    if ! docker inspect local-registry | grep -q "REGISTRY_STORAGE_DELETE_ENABLED.*true"; then
        echo "⚠️  Registry exists but deletion is not enabled. Recreating with deletion enabled..."
        docker stop local-registry 2>/dev/null || true
        docker rm local-registry 2>/dev/null || true
        docker run -d --name local-registry --restart=unless-stopped -p 5000:5000 \
            -e REGISTRY_STORAGE_DELETE_ENABLED=true \
            -v registry-data:/var/lib/registry \
            registry:2
        sleep 3
    fi
fi

# Archive current mlbox:latest from registry to timestamped version before building new one
echo "📦 Archiving current mlbox:latest from registry..."
CURRENT_DATE=$(date +%Y-%m-%d)
CURRENT_TIME=$(date +%H_%M_%S)
ARCHIVE_TAG="${CURRENT_DATE}_${CURRENT_TIME}"
echo "📅 Tagging current latest as: $ARCHIVE_TAG"

# Pull current latest from registry (if it exists) and tag it
if docker pull $REGISTRY_HOST/mlbox:latest 2>/dev/null; then
    docker tag $REGISTRY_HOST/mlbox:latest $REGISTRY_HOST/mlbox:$ARCHIVE_TAG
    docker push $REGISTRY_HOST/mlbox:$ARCHIVE_TAG
    echo "✅ Archived current latest as $ARCHIVE_TAG"
    # Remove the pulled image to save space
    docker rmi $REGISTRY_HOST/mlbox:latest 2>/dev/null || true
else
    echo "ℹ️  No existing mlbox:latest in registry, skipping archive"
fi

echo "🐳 Building new Docker image..."
docker build -t mlbox:latest .

if [ $? -eq 0 ]; then
    echo "✅ Successfully built new Docker image: mlbox:latest"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Push new mlbox:latest
echo "📤 Pushing new mlbox:latest to registry..."
docker tag mlbox:latest $REGISTRY_HOST/mlbox:latest
docker push $REGISTRY_HOST/mlbox:latest
echo "✅ Pushed new mlbox:latest"

# Clean up ALL local registry-tagged images (keep only mlbox:latest)
echo "🧹 Removing local registry-tagged images..."
# Remove all registry-tagged images for mlbox
docker images --format "{{.Repository}}:{{.Tag}}" | grep "^$REGISTRY_HOST/mlbox:" | while read -r image; do
    docker rmi "$image" 2>/dev/null || true
done

# Remove dangling images (<none> tags)
echo "🧹 Removing dangling images..."
docker image prune -f

# ------------------------------------------
# 🧹 Registry cleanup using Registry HTTP API
# ------------------------------------------
echo "🧹 Checking registry for old tags..."
TAGS_JSON=$(curl -s http://$REGISTRY_HOST/v2/mlbox/tags/list || echo "{}")
# Sort tags in ascending order (oldest first), so we can remove from the beginning
TAGS_RAW=$(echo "$TAGS_JSON" | jq -r '.tags[]? | select(. != "latest")' 2>/dev/null | sort)
if [ -z "$TAGS_RAW" ]; then
    TAGS=()
else
    readarray -t TAGS <<< "$TAGS_RAW"
fi

TAG_COUNT=${#TAGS[@]}
if (( TAG_COUNT > MAX_VERSIONS )); then
    REMOVE_COUNT=$((TAG_COUNT - MAX_VERSIONS))
    echo "🗑️  Found $TAG_COUNT timestamped tags. Keeping $MAX_VERSIONS newest, removing $REMOVE_COUNT oldest tag(s)..."
    # Remove oldest tags (from the beginning of sorted array) using Registry API
    for ((i=0; i<REMOVE_COUNT; i++)); do
        OLD_TAG=${TAGS[$i]}
        echo "   → Deleting tag: $OLD_TAG"
        
        # Get manifest digest using HEAD request
        DIGEST=$(curl -sI -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
            "http://$REGISTRY_HOST/v2/mlbox/manifests/$OLD_TAG" 2>/dev/null | \
            grep -i "Docker-Content-Digest" | awk '{print $2}' | tr -d $'\r\n')
        
        if [ -n "$DIGEST" ]; then
            # Delete manifest by digest using DELETE API
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
                -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
                "http://$REGISTRY_HOST/v2/mlbox/manifests/$DIGEST" 2>/dev/null)
            
            if [ "$HTTP_CODE" = "202" ] || [ "$HTTP_CODE" = "200" ]; then
                echo "   ✅ Deleted tag $OLD_TAG (digest: ${DIGEST:0:20}...)"
            elif [ "$HTTP_CODE" = "405" ]; then
                echo "   ⚠️  DELETE API not supported (405). Registry may need REGISTRY_STORAGE_DELETE_ENABLED=true"
                echo "   → Falling back to filesystem deletion..."
                docker exec local-registry rm -rf "/docker/registry/v2/repositories/mlbox/_manifests/tags/$OLD_TAG" 2>/dev/null
                if [ $? -eq 0 ]; then
                    echo "   ✅ Deleted tag $OLD_TAG via filesystem"
                else
                    echo "   ❌ Failed to delete tag $OLD_TAG"
                fi
            else
                echo "   ⚠️  Failed to delete tag $OLD_TAG (HTTP $HTTP_CODE)"
            fi
        else
            echo "   ⚠️  Could not resolve digest for tag $OLD_TAG"
        fi
    done
    echo "✅ Kept $MAX_VERSIONS newest timestamped tags (plus 'latest')"
else
    echo "✅ No cleanup needed ($TAG_COUNT timestamped tags ≤ $MAX_VERSIONS, plus 'latest')"
fi

# -------------------------------
# 🧹 Garbage collection & restart
# -------------------------------
echo "🧹 Running garbage collection in local registry..."
docker exec local-registry registry garbage-collect /etc/docker/registry/config.yml --delete-untagged=true 2>&1 || echo "⚠️  GC failed or skipped"

echo "🔁 Restarting local registry..."
docker restart local-registry || echo "⚠️  Restart failed"
sleep 2

# Verify cleanup
echo "🔍 Verifying registry cleanup..."
FINAL_TAGS_JSON=$(curl -s http://$REGISTRY_HOST/v2/mlbox/tags/list || echo "{}")
FINAL_TAG_COUNT=$(echo "$FINAL_TAGS_JSON" | jq -r '.tags | length' 2>/dev/null || echo "0")
FINAL_TIMESTAMPED_COUNT=$(echo "$FINAL_TAGS_JSON" | jq -r '[.tags[]? | select(. != "latest")] | length' 2>/dev/null || echo "0")
echo "   Current tags in registry: $FINAL_TAG_COUNT total ($FINAL_TIMESTAMPED_COUNT timestamped + latest)"

if [ "$FINAL_TIMESTAMPED_COUNT" -le "$MAX_VERSIONS" ]; then
    echo "✅ Registry cleanup verified successfully!"
else
    echo "⚠️  Warning: Registry still has $FINAL_TIMESTAMPED_COUNT timestamped tags (expected ≤ $MAX_VERSIONS)"
fi

echo "✅ Registry updated and cleaned successfully!"
echo "🎉 Deployment completed successfully!"

echo ""
echo "📋 Next steps:"
echo "   • Run locally: docker run -p 8000:8000 mlbox:latest"
echo ""
echo "📋 Production (from other servers):"
echo "   • Configure insecure registry: Add '\"insecure-registries\": [\"$REGISTRY_HOST\"]' to /etc/docker/daemon.json"
echo "   • Pull from registry: docker pull $REGISTRY_HOST/mlbox:latest"
echo "   • Run from registry: docker run -d -p 8000:8000 $REGISTRY_HOST/mlbox:latest"
echo "   • List registry versions: curl http://$REGISTRY_HOST/v2/mlbox/tags/list"
