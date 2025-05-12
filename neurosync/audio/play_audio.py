import io
import time
import os
import logging
import pygame # Keep pygame for potential non-RTMP fallback or other uses
import threading # Added for Event

# Keep existing convert_audio import if needed
# from .convert_audio import convert_to_wav

# Added GStreamer import
from .gst_stream import stream_wav_to_rtmp

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --- Helper Functions --- (Keep existing Pygame helpers if needed)

def init_pygame_mixer():

    if not pygame.mixer.get_init():
        try:
             pygame.mixer.init()
        except Exception as e:
             logger.error(f"Failed to initialize Pygame mixer: {e}")

# --- Configuration Helpers --- (Copied from original working version)

def _audio_mode() -> str:
    # \"\"\"Return the requested audio mode.

    # • \"rtmp\"  → stream with GStreamer (default)
    # • \"pygame\" → local playback via SDL/ALSA
    # \"\"\"
    return os.getenv("AUDIO_MODE", "rtmp").lower()

def _rtmp_url() -> str:
    # \"\"\"Return the RTMP url to push to.

    # Prioritizes Twitch if TWITCH_STREAM_KEY is set, otherwise defaults to local RTMP server.
    # \"\"\"
    twitch_stream_key = os.getenv("TWITCH_STREAM_KEY")
    # Use LIVELINK_TARGET_IP from Core's env vars if OBS_HOST_IP isn't set
    obs_host_ip = os.getenv("OBS_HOST_IP") or os.getenv("LIVELINK_TARGET_IP", "127.0.0.1")

    if twitch_stream_key:
        twitch_broadcast_mode = os.getenv("TWITCH_BROADCAST_MODE", "test").lower()
        logger.info("Twitch stream key found. Target: Twitch.")
        # NOTE: Do NOT log the stream_key itself for security!
        if twitch_broadcast_mode == "live":
            logger.info("TWITCH_BROADCAST_MODE=live. Streaming to Twitch for public broadcast.")
            return f"rtmp://live.twitch.tv/app/{twitch_stream_key}"
        else:
            logger.info("TWITCH_BROADCAST_MODE=test (or not set). Streaming to Twitch in bandwidth test mode.")
            return f"rtmp://live.twitch.tv/app/{twitch_stream_key}?bandwidthtest=true"
    else:
        # Use NeuroSync-Core env var if available, else default
        default_local_rtmp = os.getenv("RTMP_URL", f"rtmp://{obs_host_ip}/live/audiostream")
        logger.info(f"No Twitch key found. Target: Local RTMP server at {default_local_rtmp}")
        return default_local_rtmp

# --- Playback Functions --- (New/Modified)

def play_audio_from_path(audio_path, start_event: threading.Event):
    # \"\"\
    # Play audio from a file path, prioritizing RTMP streaming.
    # Uses the start_event to synchronize with other threads (like blendshape sender).
    # \"\"\"
    mode = _audio_mode()

    # -------------------------------------------------------------
    # Primary path: GStreamer streaming (default)
    # -------------------------------------------------------------
    if mode != "pygame":
        rtmp_url_target = _rtmp_url()
        logger.info(f"[Audio] Waiting for start signal to stream {audio_path} to {rtmp_url_target} (mode={mode})")
        start_event.wait() # Wait for signal before starting stream
        logger.info(f"[Audio] Start signal received. Streaming {audio_path} to {rtmp_url_target}...")
        try:
            # Use blocking=True because the caller (e.g., run_audio_animation)
            # should wait for audio to finish before proceeding/cleanup.
            success = stream_wav_to_rtmp(audio_path, rtmp_url_target, blocking=True)
            if not success:
                logger.error(f"[Audio] GStreamer streaming failed for {audio_path}.")
                # Optionally fallback to pygame here if needed
        except Exception as stream_error:
            logger.error(f"[Audio] GStreamer streaming failed with exception: {stream_error}", exc_info=True)
            # Optionally fallback to pygame here if needed
        return

    # -------------------------------------------------------------
    # Secondary path: local playback via pygame (if AUDIO_MODE=pygame)
    # -------------------------------------------------------------
    logger.warning("AUDIO_MODE=pygame selected. Attempting local playback.")
    try:
        init_pygame_mixer() # Ensure mixer is ready
        if not pygame.mixer.get_init():
            logger.error("[Audio] Pygame mixer failed to initialize. Cannot play locally.")
            return

        try:
            pygame.mixer.music.load(audio_path)
        except pygame.error as load_error:
            logger.error(f"[Audio] Pygame failed to load {audio_path}: {load_error}. Cannot play locally.")
            # Consider conversion fallback if needed: audio_path = convert_to_wav(audio_path)
            return

        logger.info(f"[Audio] Waiting for start signal to play {audio_path} via Pygame...")
        start_event.wait() # Wait for signal before starting playback
        logger.info(f"[Audio] Start signal received. Playing {audio_path} via Pygame...")
        pygame.mixer.music.play()

        # Use a simple loop - sync loop might be needed if timing is critical
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # Check roughly 30 times per second
        logger.info(f"[Audio] Pygame playback finished for {audio_path}.")

    except Exception as e:
        logger.error(f"[Audio] Pygame playback failed unexpectedly: {e}", exc_info=True)

# --- Existing functions below (if any) --- 

# Example: Keep read_audio_file_as_bytes if used elsewhere in Core
def read_audio_file_as_bytes(file_path):
    # \"\"\
    # Read a WAV audio file from disk as bytes.
    # Only WAV files are supported.
    # \"\"\"
    if not file_path.lower().endswith('.wav'):
        logger.error(f"Unsupported file format: {file_path}. Only WAV files are supported.")
        return None
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading audio file: {e}")
        return None 