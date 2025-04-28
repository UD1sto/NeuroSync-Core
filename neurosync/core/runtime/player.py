# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
from threading import Thread, Event, Lock
import numpy as np
import random
from typing import Union, Tuple, Iterator, List
from pathlib import Path
import socket # Import needed for socket handling in Player

# Import from new locations within neurosync.core
from ..audio.play_audio import play_audio_from_path, play_audio_from_memory
from ..livelink import (
    pre_encode_facial_data, 
    send_pre_encoded_data_to_unreal, 
    send_sequence_to_unreal,
    default_animation_loop, 
    stop_default_animation,
    initialize_py_face, 
    create_socket_connection, # Needed if Player manages its own connection
    determine_highest_emotion,
    merge_emotion_data_into_facial_data_wrapper,
    emotion_animations
)
from ..model.blendshape_sequence import BlendshapeSequence

queue_lock = Lock()

class Player:
    """
    Unified player class for audio and blendshape playback with sample-accurate timing.
    Manages the default idle animation thread and LiveLink connection.
    """
    def __init__(self, py_face=None, socket_connection=None, manage_connection=True):
        """
        Initialize the Player.

        Args:
            py_face: Optional pre-initialized PyLiveLinkFace instance for default animation.
            socket_connection: Optional pre-existing socket connection.
            manage_connection: If True and socket_connection is None, Player creates and manages its own socket.
        """
        self.py_face = py_face if py_face else initialize_py_face() # Initialize if not provided
        self.manage_connection = manage_connection
        self.socket_connection = socket_connection
        self.own_socket = False
        
        if self.socket_connection is None and self.manage_connection:
            try:
                self.socket_connection = create_socket_connection()
                self.own_socket = True
            except Exception as e:
                print(f"Player failed to create socket connection: {e}")
                self.socket_connection = None # Ensure it's None if creation failed

        self.default_animation_thread = None
        self.encoding_face = initialize_py_face() # Separate instance for encoding active sequences
        self.start_default_animation() # Start idle anim immediately
    
    def start_default_animation(self):
        """Starts the default idle animation thread."""
        with queue_lock:
            if self.default_animation_thread is None or not self.default_animation_thread.is_alive():
                stop_default_animation.clear()
                self.default_animation_thread = Thread(
                    target=default_animation_loop, 
                    args=(self.py_face, self.socket_connection), 
                    daemon=True # Make daemon so it doesn't block exit
                )
                self.default_animation_thread.start()

    def stop_default_animation(self):
        """Stops the default idle animation thread."""
        with queue_lock:
            stop_default_animation.set()
            if self.default_animation_thread and self.default_animation_thread.is_alive():
                self.default_animation_thread.join(timeout=0.5) # Wait briefly
            self.default_animation_thread = None # Reset thread variable

    def play(self, audio: Union[bytes, Path], sequence: BlendshapeSequence):
        """
        Play audio and corresponding blendshapes with sample-accurate timing.
        Manages pausing and resuming the default animation.
        
        Args:
            audio: Audio data as bytes or path to audio file.
            sequence: BlendshapeSequence containing frames and timing info.
        """
        if sequence is None or sequence.frames is None or len(sequence.frames) == 0:
            print("Player received empty or invalid BlendshapeSequence. Skipping playback.")
            return
            
        # Ensure audio is valid Path object if not bytes
        if not isinstance(audio, bytes):
            audio = Path(audio)
            if not audio.is_file():
                print(f"Player received invalid audio path: {audio}. Skipping playback.")
                return

        # Apply emotion overlay if needed
        if len(sequence.frames[0]) > 61: # Check if emotion data is present
            try:
                facial_data_array = np.array(sequence.frames)
                dominant_emotion = determine_highest_emotion(facial_data_array)

                if dominant_emotion in emotion_animations and len(emotion_animations[dominant_emotion]) > 0:
                    selected_animation = random.choice(emotion_animations[dominant_emotion])
                    # Ensure selected_animation is a numpy array before passing
                    if isinstance(selected_animation, list):
                         selected_animation = np.array(selected_animation)
                         
                    # Ensure facial_data_array is mutable if it's not already
                    if not facial_data_array.flags.writeable:
                        facial_data_array = facial_data_array.copy()
                        
                    merged_frames = merge_emotion_data_into_facial_data_wrapper(facial_data_array, selected_animation)
                    sequence.frames = merged_frames.tolist() # Convert back to list for sequence
            except Exception as e:
                print(f"Error applying emotion overlay: {e}")
                # Continue without overlay

        # Stop default animation before starting playback
        self.stop_default_animation()

        start_event = Event()
        playback_successful = True

        # Start audio playback thread
        try:
            if isinstance(audio, bytes):
                audio_thread = Thread(target=play_audio_from_memory, args=(audio, start_event), daemon=True)
            else:
                audio_thread = Thread(target=play_audio_from_path, args=(str(audio), start_event), daemon=True)
            audio_thread.start()
        except Exception as e:
            print(f"Error starting audio thread: {e}")
            playback_successful = False

        # Start blendshape playback thread
        try:
            data_thread = Thread(
                target=send_sequence_to_unreal, 
                args=(sequence, self.encoding_face, start_event, self.socket_connection),
                daemon=True
            )
            data_thread.start()
        except Exception as e:
            print(f"Error starting blendshape thread: {e}")
            playback_successful = False
            # Attempt to stop audio thread if data thread failed
            if 'audio_thread' in locals() and audio_thread.is_alive():
                 # This is difficult to stop cleanly, Pygame doesn't offer direct stop
                 pass
                 
        if playback_successful:
            start_event.set() # Signal threads to start processing
            # Wait for both threads to complete
            if 'audio_thread' in locals():
                audio_thread.join()
            if 'data_thread' in locals():
                data_thread.join()
        else:
             print("Playback aborted due to thread start failure.")

        # Restore default animation after playback finishes or fails
        self.start_default_animation()
    
    def stream(self, generator: Iterator[Tuple[Union[bytes, Path], BlendshapeSequence]]):
        """
        Stream audio and blendshapes from a generator.
        
        Args:
            generator: Iterator yielding (audio, BlendshapeSequence) tuples.
                       Audio can be bytes or Path.
        """
        try:
             for audio, sequence in generator:
                if audio is None or sequence is None:
                     print("Stream generator finished or yielded None. Stopping stream.")
                     break
                self.play(audio, sequence)
        except Exception as e:
            print(f"Error during stream processing: {e}")
        finally:
             # Ensure default animation is restarted even if stream errors out
             self.start_default_animation()

    def close(self):
         """Clean up resources, stop threads and close owned socket."""
         print("Closing Player resources...")
         self.stop_default_animation()
         if self.own_socket and self.socket_connection:
             try:
                 self.socket_connection.close()
                 print("Player closed owned socket connection.")
             except Exception as e:
                 print(f"Error closing player socket: {e}")
         self.socket_connection = None

    def __del__(self):
        # Attempt cleanup when object is garbage collected
        self.close()

# Legacy function for backward compatibility
# Consider deprecating or removing if no longer used externally
def run_audio_animation(audio_input: Union[bytes, Path], 
                      generated_facial_data: Union[List[List[float]], np.ndarray], 
                      py_face, 
                      socket_connection, 
                      default_animation_thread, # This parameter is now managed by Player
                      sr: int = 16000, 
                      fps: int = 60):
    """Legacy function maintained for backward compatibility. Uses the Player class internally."""
    print("Warning: run_audio_animation is deprecated. Use Player class directly.")
    # Convert to BlendshapeSequence
    if isinstance(generated_facial_data, np.ndarray):
        frames = generated_facial_data.tolist()
    else:
        frames = generated_facial_data
        
    sequence = BlendshapeSequence(fps=fps, sr=sr, frames=frames)
    
    # Use the new Player, letting it manage the default animation thread and connection if needed
    # Pass existing py_face and socket_connection if provided
    player = Player(py_face=py_face, socket_connection=socket_connection, manage_connection=False) 
    try:
        player.play(audio_input, sequence)
    finally:
        # Player now handles its own cleanup via close() or __del__
        # We don't manually close the connection here as Player might not own it.
        pass 