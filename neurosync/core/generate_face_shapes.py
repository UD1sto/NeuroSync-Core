# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# generate_face_shapes.py

import numpy as np

# Updated imports to use neurosync.core paths
from neurosync.core.audio.extraction.extract_features import extract_audio_features
from neurosync.core.audio.processing.audio_processing import process_audio_features

def generate_facial_data_from_bytes(audio_bytes, model, device, config):
    
    audio_features, y = extract_audio_features(audio_bytes, from_bytes=True)
    
    if audio_features is None or y is None:
        return [], np.array([])
  
    final_decoded_outputs = process_audio_features(audio_features, model, device, config)

    return final_decoded_outputs

