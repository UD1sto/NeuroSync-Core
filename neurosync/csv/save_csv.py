# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import numpy as np
import pandas as pd
import io

# Base columns (Blendshape data)
BASE_COLUMNS = [
    'Timecode', 'BlendshapeCount', 'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft',
    'EyeSquintLeft', 'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight', 'EyeLookUpRight',
    'EyeSquintRight', 'EyeWideRight', 'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose', 'MouthFunnel', 'MouthPucker',
    'MouthRight', 'MouthLeft', 'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight', 'MouthDimpleLeft',
    'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower',
    'MouthShrugUpper', 'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight', 'MouthUpperUpLeft',
    'MouthUpperUpRight', 'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft', 'BrowOuterUpRight', 'CheekPuff',
    'CheekSquintLeft', 'CheekSquintRight', 'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch', 'HeadRoll',
    'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll'
]

# Emotion columns
EMOTION_COLUMNS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

def _prepare_csv_data(generated):
    """Internal helper to prepare data and columns for CSV generation."""
    # Convert the generated list to a NumPy array
    generated_np = np.array(generated)

    if generated_np.ndim == 1:
         # Handle case where only one frame is generated
         generated_np = generated_np.reshape(1, -1)
    elif generated_np.ndim == 0 or generated_np.size == 0:
         # Handle empty input
         print("Warning: Empty data provided for CSV generation.")
         return None, None, None

    # Determine the number of dimensions
    num_dimensions = generated_np.shape[1]
    if num_dimensions == 68:
        selected_columns = BASE_COLUMNS + EMOTION_COLUMNS
        selected_data = generated_np
    elif num_dimensions == 61:
        selected_columns = BASE_COLUMNS
        selected_data = generated_np[:, :61]
    else:
        print(f"Error: Unexpected number of columns: {num_dimensions}. Expected 61 or 68.")
        return None, None, None
        # Or raise ValueError("Unexpected number of columns...")

    # Generate timecodes
    frame_count = generated_np.shape[0]
    frame_rate = 60  # Assume 60 FPS
    frame_duration = 1 / frame_rate

    timecodes = []
    for i in range(frame_count):
        total_seconds = i * frame_duration
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = (seconds - int(seconds)) * 1000
        frame_number = int(milliseconds / (1000 / frame_rate)) # Frame number within the second
        timecode = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{frame_number:02}.{int(milliseconds):03}"
        timecodes.append(timecode)

    timecodes_np = np.array(timecodes).reshape(-1, 1)
    blendshape_counts = np.full((frame_count, 1), selected_data.shape[1])

    # Stack the data together
    data_with_timecode = np.hstack((timecodes_np, blendshape_counts, selected_data))

    return data_with_timecode, selected_columns, selected_data.shape[1]

def save_generated_data_as_csv(generated, output_path):
    """ Saves generated blendshape data (61 or 68 dims) to a CSV file with timecodes. """
    data_with_timecode, selected_columns, _ = _prepare_csv_data(generated)

    if data_with_timecode is None:
        print(f"Failed to prepare data for CSV: {output_path}")
        return

    try:
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(data_with_timecode, columns=selected_columns)
        df.to_csv(output_path, index=False)
        print(f"Generated data saved to {output_path}")
    except Exception as e:
        print(f"Error saving CSV file {output_path}: {e}")

def generate_csv_in_memory(generated):
    """Generates CSV content and returns it as a BytesIO object."""
    data_with_timecode, selected_columns, _ = _prepare_csv_data(generated)

    if data_with_timecode is None:
        print("Failed to prepare data for in-memory CSV.")
        return None

    try:
        # Convert to DataFrame
        df = pd.DataFrame(data_with_timecode, columns=selected_columns)

        # Save CSV content in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        return csv_bytes
    except Exception as e:
        print(f"Error generating in-memory CSV: {e}")
        return None

def save_or_return_csv(generated, output_path=None, return_in_memory=False):
    """Saves to disk or returns a CSV as a BytesIO object based on the flag."""
    if return_in_memory:
        return generate_csv_in_memory(generated)
    else:
        if output_path is None:
             print("Error: output_path must be provided when not returning in memory.")
             return None
        save_generated_data_as_csv(generated, output_path)
        # Return None explicitly to indicate no in-memory object is returned
        return None 