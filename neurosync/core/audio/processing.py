# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# audio_processing.py

import numpy as np
import torch
from torch.cuda.amp import autocast

def decode_audio_chunk(audio_chunk, model, device, config):
    # Use precision based on config
    use_half_precision = config.get("use_half_precision", True)
    
    # Force float16 if half precision is desired; else float32
    dtype = torch.float16 if use_half_precision else torch.float32

    # Convert audio chunk directly to the desired precision
    src_tensor = torch.tensor(audio_chunk, dtype=dtype).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_half_precision and device.type == 'cuda': # Autocast only works on CUDA
            with autocast(dtype=torch.float16):
                encoder_outputs = model.encoder(src_tensor)
                output_sequence = model.decoder(encoder_outputs)
        else:
            # Run in float32 if not CUDA or half precision disabled
            src_tensor = src_tensor.float() # Ensure float32 for non-CUDA or full precision
            encoder_outputs = model.encoder(src_tensor)
            output_sequence = model.decoder(encoder_outputs)

        # Convert output tensor back to numpy array
        decoded_outputs = output_sequence.squeeze(0).cpu().float().numpy() # Ensure output is float32 numpy
    return decoded_outputs


def concatenate_outputs(all_decoded_outputs, num_frames):
    final_decoded_outputs = np.concatenate(all_decoded_outputs, axis=0)
    final_decoded_outputs = final_decoded_outputs[:num_frames]
    return final_decoded_outputs

def ensure_2d(final_decoded_outputs):
    if final_decoded_outputs.ndim == 3:
        final_decoded_outputs = final_decoded_outputs.reshape(-1, final_decoded_outputs.shape[-1])
    return final_decoded_outputs

def pad_audio_chunk(audio_chunk, frame_length, num_features, pad_mode='replicate'):
    """
    Pads the audio_chunk to ensure it has a number of frames equal to frame_length.
    
    Parameters:
        audio_chunk (np.array): Input audio data with shape (num_frames, num_features).
        frame_length (int): Desired number of frames.
        num_features (int): Number of features per frame.
        pad_mode (str): Type of padding to use. Options are:
                        - 'reflect': Pads using reflection.
                        - 'replicate': Pads by replicating the last frame.
    
    Returns:
        np.array: Padded audio_chunk with shape (frame_length, num_features).
    """
    if audio_chunk.shape[0] < frame_length:
        pad_length = frame_length - audio_chunk.shape[0]
        
        if pad_mode == 'reflect':
            # --- Original reflect padding method ---
            # Need to handle case where pad_length is larger than chunk length
            if pad_length >= audio_chunk.shape[0]:
                 # Fallback to replicate if reflect pad is too large
                 last_frame = audio_chunk[-1:] 
                 replication = np.tile(last_frame, (pad_length, 1))
                 audio_chunk = np.vstack((audio_chunk, replication))
            else:
                 padding = np.pad(
                     audio_chunk,
                     pad_width=((0, pad_length), (0, 0)),
                     mode='reflect'
                 )
                 # Using the last pad_length frames from the reflected padding
                 # This logic seems flawed, reflect padding adds to both ends.
                 # Let's correct to take from the end of the padded array.
                 audio_chunk = padding[:frame_length, :]
        
        elif pad_mode == 'replicate':
            # --- Replicate padding method ---
            last_frame = audio_chunk[-1:]  # Select the last frame (shape: (1, num_features))
            replication = np.tile(last_frame, (pad_length, 1))  # Replicate it pad_length times
            audio_chunk = np.vstack((audio_chunk, replication))
        
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}. Choose 'reflect' or 'replicate'.")
    
    # Ensure the final shape is exactly (frame_length, num_features)
    return audio_chunk[:frame_length]


def blend_chunks(chunk1, chunk2, overlap):
    actual_overlap = min(overlap, len(chunk1), len(chunk2))
    if actual_overlap <= 0:
        return np.vstack((chunk1, chunk2))
    
    blended_chunk = np.copy(chunk1)
    for i in range(actual_overlap):
        alpha = (i + 1) / (actual_overlap + 1) # Linear blend from 0 to 1 (exclusive of end points)
        blended_chunk[-actual_overlap + i] = (1 - alpha) * chunk1[-actual_overlap + i] + alpha * chunk2[i]
        
    # Combine the non-overlapping part of chunk1, the blended overlap, and the non-overlapping part of chunk2
    return np.vstack((chunk1[:-actual_overlap], blended_chunk[-actual_overlap:], chunk2[actual_overlap:]))

def process_audio_features(audio_features, model, device, config):
    # Configuration settings
    frame_length = config['frame_size']  # Number of frames per chunk (e.g., 64)
    overlap = config.get('overlap', 32)  # Number of overlapping frames between chunks
    num_features = audio_features.shape[1]
    num_frames = audio_features.shape[0]
    all_decoded_outputs = []

    # Set model to evaluation mode
    model.eval()

    # Process chunks with the specified overlap
    start_idx = 0
    processed_chunk_list = [] # Store processed chunks before blending
    
    while start_idx < num_frames:
        end_idx = min(start_idx + frame_length, num_frames)
        actual_chunk_length = end_idx - start_idx

        # Select and pad chunk if needed
        audio_chunk = audio_features[start_idx:end_idx]
        # Padding should ensure the chunk has frame_length for the model
        padded_chunk = pad_audio_chunk(audio_chunk, frame_length, num_features)

        # Pass config to dynamically choose precision
        decoded_outputs = decode_audio_chunk(padded_chunk, model, device, config)
        
        # Trim the decoded output to match the original (pre-padding) chunk length
        processed_chunk_list.append(decoded_outputs[:actual_chunk_length])

        # Move start index forward by (frame_length - overlap)
        # Ensure step is at least 1
        step = max(1, frame_length - overlap)
        start_idx += step

    # Blend the processed chunks
    if not processed_chunk_list:
        return np.array([]) # Return empty if no chunks were processed
        
    final_output = processed_chunk_list[0]
    for i in range(1, len(processed_chunk_list)):
         # Overlap for blending is based on the step size used during chunking
         blend_overlap = frame_length - step # This is the overlap used in generation
         final_output = blend_chunks(final_output, processed_chunk_list[i], blend_overlap)

    # Trim final output to the original number of frames
    final_decoded_outputs = final_output[:num_frames]

    # Normalize or apply any post-processing
    final_decoded_outputs = ensure_2d(final_decoded_outputs)
    # Normalization factor might need adjustment or be made configurable
    final_decoded_outputs[:, :61] /= 100.0 # Normalize specific columns

    # Easing effect for smooth start (fades in first 0.1 seconds)
    ease_duration_seconds = 0.1 # seconds
    fps = config.get('fps', 60) # Get FPS from config, default 60
    ease_duration_frames = min(int(ease_duration_seconds * fps), final_decoded_outputs.shape[0])
    if ease_duration_frames > 0:
        easing_factors = np.linspace(0, 1, ease_duration_frames)[:, None]
        final_decoded_outputs[:ease_duration_frames] *= easing_factors

    # Zero out unnecessary columns (optional post-processing)
    final_decoded_outputs = zero_columns(final_decoded_outputs)

    return final_decoded_outputs


def zero_columns(data):
    # Indices based on FaceBlendShape enum might be more robust
    columns_to_zero = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    if data.shape[1] > max(columns_to_zero):
        modified_data = np.copy(data) 
        modified_data[:, columns_to_zero] = 0
        return modified_data
    else:
        print("Warning: zero_columns indices exceed data dimensions. Skipping.")
        return data 