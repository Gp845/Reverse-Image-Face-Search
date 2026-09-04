# Runs the web UI. The pipeline is long-running (~75s per job) and holds two
# ONNX models in memory, so it wants a normal container rather than a
# serverless function -- see the deployment note in the README.
FROM python:3.13-slim

# opencv-python-headless still links libGL and libglib at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so edits to the source do not invalidate the layer.
COPY requirements.txt .
# Headless swaps in for opencv-python: no GUI libraries in a container, and it
# is a much smaller image for identical cv2 APIs.
RUN sed 's/^opencv-python==/opencv-python-headless==/' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt gunicorn==23.0.0

COPY pipeline/ ./pipeline/
COPY web/ ./web/
COPY models/ ./models/
COPY face_detection_yunet_2026may.onnx ./

ENV PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    # glibc gives each thread its own malloc arena, which inflates RSS well
    # beyond live data once the download pool spins up. Capping the arenas
    # costs a little contention and is worth it under a hard memory limit.
    MALLOC_ARENA_MAX=2 \
    # Sized for a 512MB container. Raise on a larger host.
    MAX_IMAGE_SIDE=1600 \
    MAX_CANDIDATE_SIDE=1280 \
    MATCH_WORKERS=4

EXPOSE 7860

# One worker with threads, not multiple workers: each worker would load its own
# copy of both models, and jobs are held in that process's memory, so a request
# routed to a different worker would not find its own job. The long timeout is
# the pipeline's ~75s run plus headroom for a slow Sepolia block.
CMD ["sh", "-c", "gunicorn --bind $HOST:$PORT --workers 1 --threads 8 --timeout 300 web.server:app"]
