# Use NVIDIA's PyTorch container as base
FROM nvcr.io/nvidia/pytorch:23.04-py3

# Set working directory
WORKDIR /app

# Copy F5-TTS source code
COPY ./F5-TTS /app/F5-TTS

# Install the package in development mode
WORKDIR /app/F5-TTS
RUN pip install -e .

# Make sure the f5-tts_infer-socket command is available
# The exact path depends on how it's installed, might need adjustment
RUN ln -s /app/F5-TTS/src/f5_tts/runtime/triton_trtllm/scripts/infer_socket.py /usr/local/bin/f5-tts_infer-socket && \
    chmod +x /usr/local/bin/f5-tts_infer-socket

# Set the entrypoint
ENTRYPOINT ["f5-tts_infer-socket"]

# Default command (can be overridden)
CMD ["--port", "5055", "--host", "0.0.0.0"]