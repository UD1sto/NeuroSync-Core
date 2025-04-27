# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

from threading import Thread, Event, Lock
import numpy as np
import random
from typing import Union, Tuple, Iterator, List
from pathlib import Path

from utils.audio.play_audio import play_audio_from_path, play_audio_from_memory
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal, send_sequence_to_unreal
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.connect.livelink_init import initialize_py_face 
from livelink.animations.animation_emotion import determine_highest_emotion, merge_emotion_data_into_facial_data_wrapper
from livelink.animations.animation_loader import emotion_animations
from utils.model.blendshape_sequence import BlendshapeSequence

queue_lock = Lock()

class Player:
    """
    Unified player class for audio and blendshape playback with sample-accurate timing.
    """
    def __init__(self, py_face=None, socket_connection=None):
        self.py_face = py_face
        self.socket_connection = socket_connection
        self.default_animation_thread = None
        self.encoding_face = initialize_py_face()
    
    def play(self, audio: Union[bytes, Path], sequence: BlendshapeSequence):
        """
        Play audio and corresponding blendshapes with sample-accurate timing.
        
        Args:
            audio: Audio data as bytes or path to audio file
            sequence: BlendshapeSequence containing frames and timing info
        """
        # Apply emotion overlay if needed
        if (sequence.frames is not None and 
            len(sequence.frames) > 0 and 
            len(sequence.frames[0]) > 61):
            
            facial_data_array = np.array(sequence.frames)
            dominant_emotion = determine_highest_emotion(facial_data_array)

            if dominant_emotion in emotion_animations and len(emotion_animations[dominant_emotion]) > 0:
                selected_animation = random.choice(emotion_animations[dominant_emotion])
                sequence.frames = merge_emotion_data_into_facial_data_wrapper(sequence.frames, selected_animation)

        with queue_lock:
            stop_default_animation.set()
            if self.default_animation_thread and self.default_animation_thread.is_alive():
                self.default_animation_thread.join()

        start_event = Event()

        # Start audio playback thread
        if isinstance(audio, bytes):
            audio_thread = Thread(target=play_audio_from_memory, args=(audio, start_event))
        else:
            audio_thread = Thread(target=play_audio_from_path, args=(audio, start_event))

        # Start blendshape playback thread with sample-accurate timing
        data_thread = Thread(
            target=send_sequence_to_unreal, 
            args=(sequence, self.encoding_face, start_event, self.socket_connection)
        )

        audio_thread.start()
        data_thread.start()

        start_event.set()

        audio_thread.join()
        data_thread.join()

        # Restore default animation
        with queue_lock:
            stop_default_animation.clear()
            self.default_animation_thread = Thread(target=default_animation_loop, args=(self.py_face,))
            self.default_animation_thread.start()
    
    def stream(self, generator: Iterator[Tuple[bytes, BlendshapeSequence]]):
        """
        Stream audio and blendshapes from a generator.
        
        Args:
            generator: Iterator yielding (audio_bytes, BlendshapeSequence) tuples
        """
        for audio_bytes, sequence in generator:
            if audio_bytes is None or sequence is None:
                break
            self.play(audio_bytes, sequence)

# Legacy function for backward compatibility
def run_audio_animation(audio_input, generated_facial_data, py_face, socket_connection, default_animation_thread, sr=16000, fps=60):
    """Legacy function maintained for backward compatibility."""
    # Convert to BlendshapeSequence
    if isinstance(generated_facial_data, np.ndarray):
        frames = generated_facial_data.tolist()
    else:
        frames = generated_facial_data
        
    sequence = BlendshapeSequence(fps=fps, sr=sr, frames=frames)
    
    # Use the new Player
    player = Player(py_face, socket_connection)
    player.default_animation_thread = default_animation_thread
    player.play(audio_input, sequence)


