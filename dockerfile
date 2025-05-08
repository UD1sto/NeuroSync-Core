# Use a general Python base image
FROM python:3.10-slim AS base

WORKDIR /app

# --------------------------------------------------------
# Keep existing system libs install for audio stack
# --------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libportaudio2 libasound2-dev alsa-utils build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements first (for better layer caching)
COPY requirements.txt ./

# Upgrade pip before installing requirements
RUN pip install --upgrade pip

# Install requirements from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Uninstall PyTorch, torchvision, torchaudio, bitsandbytes if they were in requirements.txt,
# as boot.sh will install the correct versions.
# Add other GPU-specific packages here if necessary.
RUN pip uninstall -y torch torchvision torchaudio bitsandbytes || echo "PyTorch/related packages not found, skipping uninstall."

# Copy full NeuroSync source
COPY . .

# Copy the boot script and make it executable
COPY scripts/boot.sh /usr/local/bin/boot.sh
RUN chmod +x /usr/local/bin/boot.sh

# Expose ports: 5000 = Flask API, 5055 = F5-TTS inference socket (optional)
EXPOSE 5000 5055

ENV FLASK_HOST=0.0.0.0

# Use boot.sh as the entrypoint
ENTRYPOINT ["boot.sh"]
# CMD is removed as ENTRYPOINT now handles startup