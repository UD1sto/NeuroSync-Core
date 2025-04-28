# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import time
import socket
import pandas as pd
from threading import Event

# Corrected import for FaceBlendShape
from .faceblendshapes import FaceBlendShape
# Import for blending still needed
from .blending_anims import blend_animation_start_end # Relative import

def load_animation(csv_path):
    """
    Loads the default animation CSV file
    Returns the animation data as a NumPy array.
    """
    data = pd.read_csv(csv_path)
    data = data.drop(columns=['Timecode', 'BlendshapeCount'])
    return data.values

# ==================== DEFAULT ANIMATION SETUP ====================

# Path to the default animation CSV file (relative to this module)
_ANIMATIONS_DIR = os.path.join(os.path.dirname(__file__), "animations")
ground_truth_path = os.path.join(_ANIMATIONS_DIR, "default_anim", "default.csv")

# Load the default animation data
default_animation_data = None
try:
    default_animation_data = load_animation(ground_truth_path)
    # Create the blended default animation data
    default_animation_data = blend_animation_start_end(default_animation_data, blend_frames=8)
except FileNotFoundError:
    print(f"Error: Default animation file not found at {ground_truth_path}")
    # Initialize with dummy data if file not found
    import numpy as np
    default_animation_data = np.zeros((10, 61)) # Placeholder for 10 frames, 61 blendshapes
except Exception as e:
    print(f"Error loading or blending default animation: {e}")
    import numpy as np
    default_animation_data = np.zeros((10, 61))

# Event to signal stopping of the default animation loop
stop_default_animation = Event()

def default_animation_loop(py_face, socket_connection=None):
    """
    Loops through the blended default animation data and sends it to the face.
    Uses the provided socket_connection (if None, creates a dummy socket).
    """
    # Check if default animation data was loaded
    if default_animation_data is None or len(default_animation_data) == 0:
        print("Error: Default animation data not loaded. Cannot start loop.")
        return
        
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
                if i < len(FaceBlendShape):
                    try:
                        py_face.set_blendshape(FaceBlendShape(i), float(value))
                    except ValueError as e:
                        print(f"Error setting blendshape {i} with value {value}: {e}")
                        continue # Skip this blendshape if invalid index
            try:
                socket_connection.sendall(py_face.encode())
            except Exception as e:
                print(f"Error in default animation sending: {e}")
            total_sleep = 1 / 60 
            sleep_interval = 0.005  
            while total_sleep > 0 and not stop_default_animation.is_set():
                time.sleep(min(sleep_interval, total_sleep))
                total_sleep -= sleep_interval 