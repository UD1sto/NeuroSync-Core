import soundfile as sf
import wave
import io
import numpy as np
import scipy.signal

def save_audio_file(audio_bytes, output_path, target_sr=88200):
    # Read the audio data and sampling rate from the bytes using soundfile
    data, sr = sf.read(io.BytesIO(audio_bytes))

    # If the audio has more than one channel, convert to mono by averaging channels
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample the audio if the original sample rate doesn't match the target
    if sr != target_sr:
        # Using resample_poly for efficient and high-quality resampling.
        # It rescales the audio by treating target_sr as the "up" factor and sr as the "down" factor.
        try:
             data = scipy.signal.resample_poly(data, target_sr, sr)
             sr = target_sr
        except ValueError as e:
             print(f"Warning: Resampling failed ({e}). Saving with original sample rate {sr}.")

    # Write the processed audio data to a WAV file
    try:
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)         # Output mono audio
            wf.setsampwidth(2)         # 16-bit PCM audio (2 bytes per sample)
            wf.setframerate(sr)
            # Scale float audio (typically in range [-1, 1]) to int16
            wf.writeframes((data * 32767).astype(np.int16).tobytes())
        print(f"Audio data saved to {output_path}")
    except Exception as e:
         print(f"Error saving audio file to {output_path}: {e}") 