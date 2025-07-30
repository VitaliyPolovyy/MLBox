# Docker Setup for MLBox

This project includes Docker configuration for both production and development environments.

## Quick Start

### Production
```bash
# Build and run the production container
docker-compose up --build

# Or build and run manually
docker build -t mlbox .
docker run -p 8000:8000 mlbox
```

### Development
```bash
# Build and run the development container
docker-compose --profile dev up --build

# Or build and run manually
docker build -f Dockerfile.dev -t mlbox-dev .
docker run -it -p 8000:8000 -p 8888:8888 -v $(pwd):/app mlbox-dev
```

## Docker Files

- `Dockerfile` - Production-ready image with minimal dependencies
- `Dockerfile.dev` - Development image with all dependencies and tools
- `docker-compose.yml` - Orchestration for both production and development

## Features

### Production Dockerfile
- Uses Python 3.11 slim image
- Poetry for dependency management
- Non-root user for security
- Health checks
- Optimized for production deployment

### Development Dockerfile
- Includes development dependencies
- Interactive shell for development
- Volume mounting for live code changes
- Jupyter notebook support

## Volume Mounts

The following directories are mounted as volumes:
- `./data` - Persistent data storage
- `./logs` - Application logs
- `./assets` - Static assets

## Environment Variables

- `RAY_RUNTIME_ENV_MODE=host` - Ray runtime environment mode
- `PYTHONUNBUFFERED=1` - Unbuffered Python output
- `PYTHONDONTWRITEBYTECODE=1` - Don't write bytecode files

## Health Checks

The production container includes health checks that verify the Ray Serve application is running:
```bash
curl -f http://localhost:8000/peanuts/health
```

## Building for Different Platforms

To build for ARM64 (Apple Silicon, etc.):
```bash
docker buildx build --platform linux/arm64 -t mlbox .
```

## Troubleshooting

1. **Port conflicts**: If port 8000 is already in use, change the port mapping in docker-compose.yml
2. **Permission issues**: The container runs as non-root user (appuser)
3. **Memory issues**: Ray Serve may require significant memory for ML models

## Development Workflow

1. Start development container: `docker-compose --profile dev up --build`
2. Access container shell: `docker exec -it <container_name> bash`
3. Run tests: `pytest`
4. Start Jupyter: `jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root`
5. Access Jupyter at: `http://localhost:8888`