"""Core LiveLink communication and animation handling for NeuroSync.

This package contains:
- PyLiveLinkFace class for encoding ARKit blendshapes.
- Helpers for UDP connection, initialization, and scaling.
- Animation loading, blending, and sending logic.
- Default idle animation loop.
"""

# Core LiveLink components (originally from livelink/connect)
from .faceblendshapes import FaceBlendShape
from .livelink_init import create_socket_connection, initialize_py_face, UDP_PORT
from .pylivelinkface import PyLiveLinkFace
from .dimension_scalars import scale_blendshapes_by_section, MOUTH_BLENDSHAPES, EYE_BLENDSHAPES, EYEBROW_BLENDSHAPES

# Animation components (originally from livelink/animations & send_to_unreal)
from .animation_loader import emotion_animations, load_animation, load_emotion_animations
from .animation_emotion import determine_highest_emotion, merge_emotion_data_into_facial_data_wrapper
from .blending_anims import blend_animation_start_end, blend_animation_data_to_loop_by_dimension
from .default_animation import default_animation_loop, stop_default_animation, default_animation_data
from .send_to_unreal import apply_blink_to_facial_data, pre_encode_facial_data, send_pre_encoded_data_to_unreal, send_sequence_to_unreal

# Core LiveLink integration modules 