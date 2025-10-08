# MLBox Production Deployment - Quick Reference

## Dev Registry
- **Host**: `10.11.122.100:5000`
- **Image**: `mlbox`

---

## One-Time Setup (Production Server)

```bash
# 1. Configure Docker
echo '{
  "insecure-registries": ["10.11.122.100:5000"]
}' | sudo tee /etc/docker/daemon.json

sudo systemctl restart docker

# 2. Test connectivity
curl http://10.11.122.100:5000/v2/mlbox/tags/list
```

---

## Deploy/Update Application

```bash
# Pull latest version
docker pull 10.11.122.100:5000/mlbox:latest

# Stop old container (if exists)
docker stop mlbox-app 2>/dev/null || true
docker rm mlbox-app 2>/dev/null || true

# Run new container
docker run -d \
  --name mlbox-app \
  --restart=unless-stopped \
  -p 8000:8000 \
  10.11.122.100:5000/mlbox:latest

# Check logs
docker logs -f mlbox-app
```

---

## Common Commands

```bash
# View available versions
curl http://10.11.122.100:5000/v2/mlbox/tags/list

# Check container status
docker ps | grep mlbox-app

# View logs
docker logs -f mlbox-app

# Restart container
docker restart mlbox-app

# Check application health
curl http://localhost:8000
```

---

## Production Best Practice

Use specific version tags instead of `latest`:

```bash
docker run -d \
  --name mlbox-app \
  --restart=unless-stopped \
  --memory="4g" \
  --cpus="2" \
  -p 8000:8000 \
  10.11.122.100:5000/mlbox:2025-10-07_12_13_23
```

