# MLOps Deployment Guide

## Overview
This guide explains how to deploy the `mlbox` application in production by pulling Docker images from the development registry.

---

## Development Registry Information

- **Registry Host**: `10.11.122.100:5000`
- **Image Name**: `mlbox`
- **Protocol**: HTTP (insecure registry)
- **Available Tags**: 
  - `latest` - most recent build
  - Timestamped versions (e.g., `2025-10-07_12_13_23`)

---

## Production Server Setup

### Prerequisites
- Docker installed on production server
- Network connectivity to dev server (`10.11.122.100`)
- Port `5000` accessible from production to dev server

### Step 1: Configure Docker for Insecure Registry

On the **production server**, configure Docker to trust the dev registry:

```bash
# Create or edit Docker daemon configuration
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
```

Add the following content:

```json
{
  "insecure-registries": ["10.11.122.100:5000"]
}
```

**Note**: If the file already exists and has other configurations, just add the `insecure-registries` entry to the existing JSON object.

Restart Docker to apply changes:

```bash
sudo systemctl restart docker
```

### Step 2: Verify Registry Connectivity

Test connectivity to the dev registry:

```bash
# Check registry catalog
curl http://10.11.122.100:5000/v2/_catalog

# Expected output: {"repositories":["mlbox"]}

# List available mlbox versions
curl http://10.11.122.100:5000/v2/mlbox/tags/list

# Expected output: {"name":"mlbox","tags":["latest","2025-10-07_12_13_23","2025-09-24"]}
```

### Step 3: Pull Docker Image

Pull the latest image from dev registry:

```bash
docker pull 10.11.122.100:5000/mlbox:latest
```

Or pull a specific version:

```bash
docker pull 10.11.122.100:5000/mlbox:2025-10-07_12_13_23
```

### Step 4: Run Container in Production

Run the container with auto-restart:

```bash
docker run -d \
  --name mlbox-app \
  --restart=unless-stopped \
  -p 8000:8000 \
  10.11.122.100:5000/mlbox:latest
```

**Parameters explained**:
- `-d`: Run in detached mode (background)
- `--name mlbox-app`: Container name
- `--restart=unless-stopped`: Auto-restart on server reboot
- `-p 8000:8000`: Map container port 8000 to host port 8000
- `10.11.122.100:5000/mlbox:latest`: Image to run

---

## Container Management

### View Running Containers
```bash
docker ps
```

### View Container Logs
```bash
docker logs mlbox-app

# Follow logs in real-time
docker logs -f mlbox-app

# Last 100 lines
docker logs --tail 100 mlbox-app
```

### Stop Container
```bash
docker stop mlbox-app
```

### Start Container
```bash
docker start mlbox-app
```

### Restart Container
```bash
docker restart mlbox-app
```

### Remove Container
```bash
# Stop and remove
docker stop mlbox-app
docker rm mlbox-app
```

---

## Updating to New Version

When a new version is available:

```bash
# Pull the new version
docker pull 10.11.122.100:5000/mlbox:latest

# Stop and remove old container
docker stop mlbox-app
docker rm mlbox-app

# Run new container
docker run -d \
  --name mlbox-app \
  --restart=unless-stopped \
  -p 8000:8000 \
  10.11.122.100:5000/mlbox:latest
```

Or use a one-liner with automatic cleanup:

```bash
docker pull 10.11.122.100:5000/mlbox:latest && \
docker stop mlbox-app && \
docker rm mlbox-app && \
docker run -d --name mlbox-app --restart=unless-stopped -p 8000:8000 10.11.122.100:5000/mlbox:latest
```

---

## Troubleshooting

### Issue: Cannot connect to registry

**Symptoms**:
```
Error response from daemon: Get "https://10.11.122.100:5000/v2/": http: server gave HTTP response to HTTPS client
```

**Solution**: Ensure `insecure-registries` is configured in `/etc/docker/daemon.json` and Docker is restarted.

---

### Issue: Network connectivity problems

**Check network connectivity**:
```bash
# Ping dev server
ping 10.11.122.100

# Check port accessibility
telnet 10.11.122.100 5000
# or
nc -zv 10.11.122.100 5000
```

**If firewall is blocking**:
- Contact dev team to open port 5000 on dev server
- Or configure VPN/network access

---

### Issue: Container fails to start

**Check logs**:
```bash
docker logs mlbox-app
```

**Check container status**:
```bash
docker ps -a
```

**Inspect container**:
```bash
docker inspect mlbox-app
```

---

## Health Checks

### Verify Application is Running

```bash
# Check if port 8000 is listening
curl http://localhost:8000

# Or from another machine
curl http://<production-server-ip>:8000
```

### Check Container Health

```bash
# Container stats (CPU, memory usage)
docker stats mlbox-app

# Container processes
docker top mlbox-app

# Container resource usage
docker inspect mlbox-app --format='{{.State.Status}}'
```

---

## Registry Information Commands

### List All Available Versions

```bash
# From any machine with access to dev server
curl http://10.11.122.100:5000/v2/mlbox/tags/list
```

### Check Image Details

```bash
# List local images
docker images 10.11.122.100:5000/mlbox

# Image details
docker inspect 10.11.122.100:5000/mlbox:latest
```

---

## Production Best Practices

1. **Always use specific version tags** in production (not `latest`) for better control:
   ```bash
   docker run -d --name mlbox-app -p 8000:8000 10.11.122.100:5000/mlbox:2025-10-07_12_13_23
   ```

2. **Set resource limits** to prevent container from consuming all resources:
   ```bash
   docker run -d \
     --name mlbox-app \
     --restart=unless-stopped \
     --memory="4g" \
     --cpus="2" \
     -p 8000:8000 \
     10.11.122.100:5000/mlbox:latest
   ```

3. **Use volumes** for persistent data if needed:
   ```bash
   docker run -d \
     --name mlbox-app \
     --restart=unless-stopped \
     -v /path/to/data:/app/data \
     -p 8000:8000 \
     10.11.122.100:5000/mlbox:latest
   ```

4. **Monitor logs** and set up log rotation:
   ```bash
   docker run -d \
     --name mlbox-app \
     --restart=unless-stopped \
     --log-opt max-size=10m \
     --log-opt max-file=3 \
     -p 8000:8000 \
     10.11.122.100:5000/mlbox:latest
   ```

---

## Contact & Support

**Dev Registry Location**: `10.11.122.100:5000`  
**Registry Versions Kept**: 3 most recent versions  
**Auto-cleanup**: Yes, old versions automatically removed

For issues or questions, contact the development team.

---

## Quick Reference

```bash
# Pull latest
docker pull 10.11.122.100:5000/mlbox:latest

# Run in production
docker run -d --name mlbox-app --restart=unless-stopped -p 8000:8000 10.11.122.100:5000/mlbox:latest

# View logs
docker logs -f mlbox-app

# Update to new version
docker pull 10.11.122.100:5000/mlbox:latest && docker stop mlbox-app && docker rm mlbox-app && docker run -d --name mlbox-app --restart=unless-stopped -p 8000:8000 10.11.122.100:5000/mlbox:latest

# Check available versions
curl http://10.11.122.100:5000/v2/mlbox/tags/list
```

