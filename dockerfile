# -----------------------------------------------------------------------------
# NeuroSync code + BYOC adapter
# -----------------------------------------------------------------------------

# Use NVIDIA's PyTorch container as base
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Set working directory
WORKDIR /app

# Copy NeuroSync core source first so that editable install can resolve deps
COPY ./NeuroSync-Core /app/NeuroSync-Core

# Copy worker adapter (server_adapter, entry scripts)
COPY ./neurosync-worker/ /app/

# Python path so that neurosync modules are discoverable even without pip install
ENV PYTHONPATH="/app/NeuroSync-Core:${PYTHONPATH}"

# Install PortAudio library
RUN apt-get update && apt-get install -y libportaudio2 libasound2-dev && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies for BYOC adapter and (optionally) editable NeuroSync
RUN pip install --no-cache-dir \
    requests \
    requests-toolbelt \
    fastapi \
    "uvicorn[standard]" \
    jsonschema && \
    pip install --no-cache-dir -r /app/NeuroSync-Core/requirements.txt && \
    pip install -e /app/NeuroSync-Core || true

# Ensure entrypoint scripts are executable
RUN chmod +x /app/entrypoint.sh /app/combined_entrypoint.sh

# Final entrypoint: runs both NeuroSync core API (Flask) and BYOC adapter
ENTRYPOINT ["/app/combined_entrypoint.sh"]

# No CMD – combined entrypoint launches both processes.