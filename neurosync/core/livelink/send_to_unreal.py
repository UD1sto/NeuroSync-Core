# send_to_unreal.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
from typing import List

from .livelink_init import create_socket_connection, FaceBlendShape # Relative import
from .default_animation import default_animation_data # Relative import
from .blending_anims import blend_in, blend_out  # Relative import
from ..model.blendshape_sequence import BlendshapeSequence


def apply_blink_to_facial_data(facial_data: List, default_animation_data: List[List[float]]):
    """
    Updates each frame in facial_data in-place by setting the blink indices (EyeBlinkLeft, EyeBlinkRight)
    to the values from default_animation_data. This ensures that the blink values are present before any blending.
    """
    blink_indices = {FaceBlendShape.EyeBlinkLeft.value, FaceBlendShape.EyeBlinkRight.value}
    default_len = len(default_animation_data)
    for idx, frame in enumerate(facial_data):
        default_idx = idx % default_len
        for blink_idx in blink_indices:
            if blink_idx < len(frame) and default_idx < len(default_animation_data) and blink_idx < len(default_animation_data[default_idx]): # Bounds check
                frame[blink_idx] = default_animation_data[default_idx][blink_idx]

def smooth_facial_data(facial_data: list) -> list:
    if len(facial_data) < 2:
        return facial_data.copy()  

    smoothed_data = [facial_data[0]]
    for i in range(1, len(facial_data)):
        previous_frame = facial_data[i - 1]
        current_frame = facial_data[i]
        averaged_frame = [(a + b) / 2 for a, b in zip(previous_frame, current_frame)]
        smoothed_data.append(averaged_frame)
    
    return smoothed_data

# smoothing shouldnt be needed, its just there if you scale too much and want to dial it back without losing scale.

def pre_encode_facial_data(facial_data: list, py_face, fps: int = 60, smooth: bool = False) -> list:
    apply_blink_to_facial_data(facial_data, default_animation_data)
    
    # If smoothing is enabled, apply smoothing to the facial data
    if smooth:
        facial_data = smooth_facial_data(facial_data)  

    encoded_data = []
    blend_in_frames = int(0.1 * fps)
    blend_out_frames = int(0.2 * fps)

    blend_in(facial_data, fps, py_face, encoded_data, blend_in_frames, default_animation_data)

    for frame_index, frame_data in enumerate(facial_data[blend_in_frames:-blend_out_frames]):
        for i in range(min(len(frame_data), 51)): # Ensure we don't exceed 51 blendshapes
            if i < len(FaceBlendShape):
                try:
                    py_face.set_blendshape(FaceBlendShape(i), frame_data[i])
                except ValueError as e:
                     print(f"Error setting blendshape {i} with value {frame_data[i]}: {e}")
                     continue
        encoded_data.append(py_face.encode())
    
    blend_out(facial_data, fps, py_face, encoded_data, blend_out_frames, default_animation_data)
    return encoded_data

def send_pre_encoded_data_to_unreal(encoded_facial_data: List[bytes], start_event, fps: int = 60, sr: int = 16000, socket_connection=None):
    try:
        own_socket = False
        if socket_connection is None:
            socket_connection = create_socket_connection()
            own_socket = True

        start_event.wait()  
        samples_per_frame = sr / fps
        start_time = time.time()  

        for frame_index, frame_data in enumerate(encoded_facial_data):
            current_time = time.time()
            elapsed_time = current_time - start_time
            expected_time = frame_index * samples_per_frame / sr
            if elapsed_time < expected_time:
                # Sleep precisely until the expected time
                sleep_duration = expected_time - elapsed_time
                time.sleep(max(0, sleep_duration)) # Ensure sleep is non-negative
            elif elapsed_time > expected_time + (samples_per_frame / sr):
                 # Frame is too late, skip it
                 print(f"Skipping frame {frame_index} due to timing delay")
                 continue

            try:
                socket_connection.sendall(frame_data) 
            except socket.error as e:
                print(f"Socket error sending frame {frame_index}: {e}")
                # Consider attempting to reconnect or break the loop
                break
            except Exception as e:
                 print(f"Unexpected error sending frame {frame_index}: {e}")
                 break

    except KeyboardInterrupt:
        print("Interrupted. Stopping UDP send loop.")
    except Exception as e:
        print(f"Error in send_pre_encoded_data_to_unreal: {e}")
    finally:
        if own_socket and socket_connection:
            try:
                socket_connection.close()
            except Exception as e:
                print(f"Error closing socket: {e}")

def send_sequence_to_unreal(sequence: BlendshapeSequence, py_face, start_event, socket_connection=None):
    """
    New function to send a BlendshapeSequence to Unreal engine using sample-accurate timing.
    """
    # Pre-encode the facial data
    encoded_data = pre_encode_facial_data(sequence.frames, py_face, sequence.fps)
    
    # Send with sample-accurate timing
    send_pre_encoded_data_to_unreal(encoded_data, start_event, sequence.fps, sequence.sr, socket_connection) 