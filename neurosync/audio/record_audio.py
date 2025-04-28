import io
import pyaudio
import numpy as np
import keyboard # Consider making this optional or providing alternatives
import soundfile as sf

# for the best results, record in 88200

def record_audio_until_release(sr=88200):
    """Record audio from the default microphone until the right Ctrl key is released."""
    p = pyaudio.PyAudio()
    stream = None
    try:
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sr,
                        input=True,
                        frames_per_buffer=1024)

        print("Recording... Press and hold Right Ctrl to record, release to stop.")

        frames = []

        # Use a loop that checks key state without blocking indefinitely
        while keyboard.is_pressed('right ctrl'):
            try:
                 data = stream.read(1024, exception_on_overflow=False)
                 frames.append(data)
            except IOError as e:
                 # Handle potential input overflow errors if needed
                 print(f"Warning: Input overflow ({e})")
                 pass # Or break, depending on desired behavior

        print("Finished recording.")

        audio_bytes = b''.join(frames)

        audio_file = io.BytesIO()
        with sf.SoundFile(audio_file, mode='w', samplerate=sr, channels=1, format='WAV', subtype='PCM_16') as f:
            f.write(np.frombuffer(audio_bytes, dtype=np.int16))

        audio_file.seek(0)
        return audio_file.read()

    except Exception as e:
         print(f"Error during recording: {e}")
         return None # Return None on error
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()
