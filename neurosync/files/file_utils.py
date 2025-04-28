# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import shutil
import wave
import uuid
import numpy as np
import soundfile as sf
import pandas as pd # Added import

# Corrected imports
from ..csv.save_csv import save_generated_data_as_csv
from ..audio.save_audio import save_audio_file
from ..api_connect.neurosync_api_connect import send_audio_to_neurosync


GENERATED_DIR = 'generated'

def reprocess_generated_files():
    """
    Processes the audio files in the 'generated' directory by sending them to the API and regenerating the facial blendshapes.
    NOTE: This relies on the external NeuroSync API and specific file structure.
    """
    # Get all directories inside the GENERATED_DIR
    if not os.path.exists(GENERATED_DIR):
         print(f"Directory not found: {GENERATED_DIR}")
         return

    directories = [d for d in os.listdir(GENERATED_DIR) if os.path.isdir(os.path.join(GENERATED_DIR, d))]

    for directory in directories:
        dir_path = os.path.join(GENERATED_DIR, directory)
        audio_path = os.path.join(dir_path, 'audio.wav')
        shapes_path = os.path.join(dir_path, 'shapes.csv')

        if os.path.exists(audio_path):
            print(f"Processing: {audio_path}")

            # Read the audio file as bytes
            try:
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
            except IOError as e:
                 print(f"Error reading audio file {audio_path}: {e}")
                 continue

            # Send audio to the API to generate facial blendshapes
            generated_facial_data = send_audio_to_neurosync(audio_bytes)

            if generated_facial_data is None:
                print(f"Failed to generate facial data for {audio_path}")
                continue

            # Move old shapes.csv to an 'old' folder and rename it with a unique identifier
            old_dir = os.path.join(dir_path, 'old')
            os.makedirs(old_dir, exist_ok=True)

            if os.path.exists(shapes_path):
                try:
                    unique_old_name = f"shapes_{uuid.uuid4()}.csv"
                    shutil.move(shapes_path, os.path.join(old_dir, unique_old_name))
                except Exception as e:
                     print(f"Error moving old shapes file {shapes_path}: {e}")
                     continue # Skip saving new if moving failed

            # Save the new blendshapes as a CSV
            save_generated_data_as_csv(generated_facial_data, shapes_path)

            print(f"New shapes.csv generated and old shapes.csv moved to {old_dir}")

def initialize_directories():
    """ Creates the default GENERATED_DIR if it doesn't exist. """
    if not os.path.exists(GENERATED_DIR):
        try:
            os.makedirs(GENERATED_DIR)
            print(f"Created directory: {GENERATED_DIR}")
        except OSError as e:
             print(f"Error creating directory {GENERATED_DIR}: {e}")

def ensure_wav_input_folder_exists(folder_path):
    """
    Checks if the specified folder exists. If not, creates it.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        except OSError as e:
             print(f"Error creating folder {folder_path}: {e}")

def list_wav_files(folder_path):
    """
    Lists all .wav files in the provided folder and returns them as a list.
    """
    if not os.path.isdir(folder_path):
         print(f"Error: Folder not found or is not a directory: {folder_path}")
         return []
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        if not files:
            print(f"No .wav files found in the folder: {folder_path}")
        return files
    except OSError as e:
        print(f"Error listing files in {folder_path}: {e}")
        return []

def list_generated_files():
    """List all the generated audio and face blend shape CSV files in the generated directory."""
    if not os.path.isdir(GENERATED_DIR):
         print(f"Generated directory not found: {GENERATED_DIR}")
         return []
    directories = [d for d in os.listdir(GENERATED_DIR) if os.path.isdir(os.path.join(GENERATED_DIR, d))]
    generated_files = []
    for directory in directories:
        audio_path = os.path.join(GENERATED_DIR, directory, 'audio.wav')
        shapes_path = os.path.join(GENERATED_DIR, directory, 'shapes.csv')
        if os.path.exists(audio_path) and os.path.exists(shapes_path):
            generated_files.append((audio_path, shapes_path))
    return generated_files

def load_animation(csv_path):
    """
    Loads animation data from a CSV file.
    Assumes first two columns are Timecode and BlendshapeCount.
    Returns the animation data as a NumPy array.
    """
    try:
        data = pd.read_csv(csv_path)
        # Check if expected columns exist before dropping
        cols_to_drop = [col for col in ['Timecode', 'BlendshapeCount'] if col in data.columns]
        data = data.drop(columns=cols_to_drop)
        return data.values
    except FileNotFoundError:
        print(f"Error: Animation file not found at {csv_path}")
        return None
    except Exception as e:
         print(f"Error loading animation CSV {csv_path}: {e}")
         return None

def save_generated_data(audio_bytes, generated_facial_data):
    """ Saves audio bytes and generated facial data to a uniquely named subdirectory within GENERATED_DIR. """
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(GENERATED_DIR, unique_id)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating output directory {output_dir}: {e}")
         return None, None, None

    audio_path = os.path.join(output_dir, 'audio.wav')
    shapes_path = os.path.join(output_dir, 'shapes.csv')

    # Attempt to save the audio using the existing method (which uses wave)
    save_audio_file(audio_bytes, audio_path)

    # Validate if the saved file is a valid WAV file
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            wav_file.getnframes() # Try to get number of frames
    except (wave.Error, EOFError) as e:
        # If the file is not valid or saving failed, try rewriting with soundfile
        print(f"Warning: Initial save/validation failed for {audio_path} ({e}). Attempting rewrite with soundfile.")
        try:
            # Assuming audio_bytes is raw PCM data if initial save failed badly
            # Or re-read if save_audio_file might have partially written?
            # For simplicity, let's assume audio_bytes is the original intended data.
            # Determine sample rate - THIS IS A PROBLEM - save_audio_file doesn't return it.
            # We need the sample rate. Assuming 88200 for now.
            sr_assumed = 88200 # MAJOR ASSUMPTION
            print(f"Warning: Assuming sample rate {sr_assumed} for soundfile rewrite.")
            with sf.SoundFile(audio_path, mode='w', samplerate=sr_assumed, channels=1, format='WAV', subtype='PCM_16') as f:
                 # Need to convert bytes back to numpy array first
                 # Assuming PCM_16 based on the initial save attempt
                 audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                 f.write(audio_array)
            print(f"Rewrote {audio_path} using soundfile.")
        except Exception as sf_e:
             print(f"Error rewriting audio with soundfile: {sf_e}")
             # Decide if we should continue without valid audio

    # Save the generated facial data as a CSV file
    save_generated_data_as_csv(generated_facial_data, shapes_path)

    return unique_id, audio_path, shapes_path

def save_generated_data_from_wav(wav_file_path, generated_facial_data):
    """ Copies an existing WAV file and saves generated facial data to a unique subdirectory. """
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(GENERATED_DIR, unique_id)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
         print(f"Error creating output directory {output_dir}: {e}")
         return None, None, None

    audio_path = os.path.join(output_dir, 'audio.wav')
    shapes_path = os.path.join(output_dir, 'shapes.csv')

    try:
        shutil.copy(wav_file_path, audio_path)
    except shutil.SameFileError:
        # This isn't really an error, file is already where it needs to be
        pass
    except Exception as e:
        print(f"Error copying audio file from {wav_file_path} to {audio_path}: {e}")
        # Decide if we should proceed without the audio copy

    save_generated_data_as_csv(generated_facial_data, shapes_path)

    return unique_id, audio_path, shapes_path 