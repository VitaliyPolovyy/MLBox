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

# Download HuggingFace models at build time (before switching to appuser)
ARG HF_TOKEN
ARG HF_PEANUT_SEG_REPO_ID
ARG HF_PEANUT_SEG_FILE
ARG HF_PEANUT_CLS_REPO_ID
ARG HF_PEANUT_CLS_FILE

RUN if [ -n "$HF_TOKEN" ] && [ -n "$HF_PEANUT_CLS_REPO_ID" ]; then \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='${HF_PEANUT_CLS_REPO_ID}', filename='${HF_PEANUT_CLS_FILE}', token='${HF_TOKEN}'); \
    hf_hub_download(repo_id='${HF_PEANUT_SEG_REPO_ID}', filename='${HF_PEANUT_SEG_FILE}', token='${HF_TOKEN}'); \
    print('Models downloaded successfully')"; \
    fi

# Create cache directory for matplotlib with proper permissions
RUN mkdir -p /tmp/matplotlib && \
    chown -R 1000:1000 /tmp/matplotlib && \
    chmod -R 755 /tmp/matplotlib

USER appuser

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD serve status | grep -E '(Peanuts|LabelGuard):' | grep 'status:' | grep -q 'HEALTHY' || exit 1

# Start Ray then deploy all services from YAML config
CMD ["bash","-lc","ray start --head --dashboard-host=0.0.0.0 && serve run deployments/ray_serv.yaml"]
