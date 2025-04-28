"""Core components for NeuroSync runtime.

This package contains the Player class and supporting modules for audio playback,
LiveLink communication, model interaction, and runtime execution.
"""

# Import key components to make them available at the neurosync.core namespace

# Runtime
from .runtime.player import Player, run_audio_animation # Player class and legacy runner

# Model related
from .model.blendshape_sequence import BlendshapeSequence
from .generate_face_shapes import generate_facial_data_from_bytes
from .model.model import load_model

# Audio
from .audio.play_audio import play_audio_bytes, play_audio_from_memory, play_audio_from_path
from .audio.convert_audio import convert_to_wav, pcm_to_wav, audio_to_bytes
from .audio.extraction import extract_audio_features # Core feature extraction
from .audio.processing import process_audio_features # Core feature processing

# LiveLink
from .livelink import (
    PyLiveLinkFace, FaceBlendShape, create_socket_connection, initialize_py_face,
    default_animation_loop, stop_default_animation, emotion_animations,
    send_sequence_to_unreal # Key function for sending sequences
)

# Legacy compatibility (to be removed after full migration)
# import utils.generate_face_shapes # Now imported above
# import utils.audio.play_audio # Covered by direct imports above

# Core NeuroSync modules
from .config import config
from .bridge import BridgeCache
from .color_text import ColorText
