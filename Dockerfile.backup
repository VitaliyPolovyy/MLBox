# -------- Builder: compile deps to wheels --------
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt

# -------- Runtime: slim final image --------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libgl1-mesa-dri libglib2.0-0 libxext6 libxrender1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install prebuilt wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /wheels/requirements.txt \
 && rm -rf /wheels


# App code (avoid recursive chown layer)
RUN useradd -m -u 1000 appuser

WORKDIR /app
WORKDIR /app
COPY --chown=1000:1000 mlbox/ ./mlbox/
COPY --chown=1000:1000 deployments/ ./deployments/
COPY --chown=1000:1000 assets/ ./assets/
COPY --chown=1000:1000 setup.py ./

RUN pip install -e .
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/peanuts/health || exit 1

# Start Ray then your app; 'exec' forwards signals to Python
CMD ["bash","-lc","ray start --head --dashboard-host=0.0.0.0 && python deployments/peanut_deployment.py && tail -f /dev/null"]
