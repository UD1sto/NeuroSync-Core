# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
import socket
import pandas as pd
from threading import Event
from pathlib import Path

from neurosync.core.livelink.livelink_init import FaceBlendShape, UDP_PORT
from neurosync.core.livelink.animations.blending_anims import blend_animation_start_end

def load_animation(csv_path):
    """
    Loads the default animation CSV file
    Returns the animation data as a NumPy array.
    """
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['Timecode', 'BlendshapeCount'])
    return data.values
# ==================== DEFAULT ANIMATION SETUP ====================

# Determine path relative to this Python file so it works regardless of CWD
ground_truth_path = Path(__file__).parent / "default_anim" / "default.csv"

# Load the default animation data
default_animation_data = load_animation(ground_truth_path)

# Create the blended default animation data
default_animation_data = blend_animation_start_end(default_animation_data, blend_frames=8)

# Event to signal stopping of the default animation loop
stop_default_animation = Event()

def default_animation_loop(py_face, socket_connection=None):
    """
    Loops through the blended default animation data and sends it to the face.
    Uses the provided socket_connection (if None, creates a dummy socket).
    """
    # Create a dummy socket that does nothing if we have no connection
    if socket_connection is None:
        class DummySocket:
            def sendall(self, *args, **kwargs):
                pass
        socket_connection = DummySocket()
        print("Default animation using dummy socket - no data will be sent")
        
    # Loop through animation frames and send them
    while not stop_default_animation.is_set():
        for frame in default_animation_data:
            if stop_default_animation.is_set():
                break
            for i, value in enumerate(frame):
                py_face.set_blendshape(FaceBlendShape(i), float(value))            
            try:
                socket_connection.sendall(py_face.encode())
            except Exception as e:
                print(f"Error in default animation sending: {e}")
            total_sleep = 1 / 60 
            sleep_interval = 0.005  
            while total_sleep > 0 and not stop_default_animation.is_set():
                time.sleep(min(sleep_interval, total_sleep))
                total_sleep -= sleep_interval

