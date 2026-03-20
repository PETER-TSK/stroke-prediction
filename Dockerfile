# ── Stroke Prediction — Docker Image ──────────────────────────────────────────
#
# Builds a reproducible image for running the stroke prediction pipeline.
#
# Build
# -----
#   docker build -t stroke-prediction .
#
# Train (requires GPU; mount data and output dirs)
# -----
#   docker run --gpus all \
#     -v $(pwd)/healthcare-dataset-stroke-data.csv:/app/healthcare-dataset-stroke-data.csv \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/results:/app/results \
#     stroke-prediction python src/main.py --time-limit 300
#
# Predict (CPU only; mount trained models and input file)
# -------
#   docker run \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/new_patients.csv:/app/new_patients.csv \
#     -v $(pwd)/results:/app/results \
#     stroke-prediction python src/predict.py \
#       --input new_patients.csv \
#       --output results/predictions.csv \
#       --threshold 0.3
#
# Notes
# -----
# * AutoGluon is large (~2 GB installed). The first build will take several minutes.
# * GPU training requires the NVIDIA Container Toolkit on the host.
# * The trained models/ directory is NOT baked into the image (gitignored).
#   Mount it at runtime with -v $(pwd)/models:/app/models.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── system dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only requirements first to leverage Docker layer cache.
# A code change won't invalidate the (slow) dependency install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── application source ─────────────────────────────────────────────────────────
COPY src/ src/

# ── runtime placeholders for mounted volumes ───────────────────────────────────
# models/ and results/ are mounted at runtime — create empty dirs as mount points.
RUN mkdir -p models results

# ── default entrypoint ─────────────────────────────────────────────────────────
# Override with 'python src/predict.py ...' for inference-only usage.
ENTRYPOINT ["python"]
CMD ["src/main.py", "--help"]
