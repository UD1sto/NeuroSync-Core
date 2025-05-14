FROM nvcr.io/nvidia/pytorch:23.04-py3

# ---------- Paths & environment ----------
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/NeuroBridge:/app/NeuroBridge/NeuroSync_Local_API:/app/NeuroBridge/NeuroSync_Player:/app/neurosync-worker:${PYTHONPATH}"

# ---------- System deps (Install these first for better caching) ----------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # GStreamer stack
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        python3-gi \
        python3-gst-1.0 \
        gir1.2-gstreamer-1.0 \
        # Audio / PyAudio dependencies
        libasound2 \
        libsndfile1 \
        libportaudio2 \
        portaudio19-dev \
        libasound2-dev \
        # For building Python packages with C extensions (like PyAudio if no wheel isfound)
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ---------- Copy source (Copy after system deps for better caching of above layer) ----------
COPY ./NeuroBridge /app/NeuroBridge
COPY ./neurosync-worker /app/neurosync-worker

# ---------- Python deps ----------
RUN pip install --no-cache-dir flask flask-cors numpy pygame jsonschema && \
    pip install --no-cache-dir -r /app/NeuroBridge/NeuroSync_Player/requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn httpx

# Verify keyboard installation and list all packages after successful install
RUN python -c "import sys; print(f'Python version: {sys.version}'); import keyboard; print('Keyboard module imported successfully during build!'); import pandas; print(f'Pandas version: {pandas.__version__}'); import fastapi; print('FastAPI imported successfully'); import uvicorn; print('Uvicorn imported successfully')" && \
    pip list && \
    # Clean pip cache after successful installs
    rm -rf /root/.cache/pip

# ---------- Entrypoint ----------
COPY ./NeuroBridge/entrypoint_bridge.sh /app/entrypoint_bridge.sh
RUN chmod +x /app/entrypoint_bridge.sh

EXPOSE 5000 9876

ENTRYPOINT ["/app/entrypoint_bridge.sh"]