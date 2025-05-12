import gi, sys, time, logging
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initialize GStreamer
try:
    Gst.init(None)
except Exception as e:
     logging.warning(f"GStreamer initialization failed: {e}. RTMP streaming will not work.")
     # Allow the rest of the app to potentially run without GStreamer if RTMP isn't used.

logging.basicConfig(level=logging.INFO)
# Default RTMP URL (can be overridden by environment variables)
DEFAULT_RTMP_URL = "rtmp://localhost/live/audiostream"

def stream_wav_to_rtmp(wav_path: str,
                       rtmp_url: str = DEFAULT_RTMP_URL,
                       blocking: bool = True) -> bool:
    # \"\"\"
    # Streams a local WAV file to an RTMP server. Non-blocking mode
    # returns immediately after starting the pipeline. Returns True on
    # successful pipeline start, False otherwise.
    # \"\"\"
    # Check if Gst was initialized successfully
    if not Gst.is_initialized():
        logging.error("GStreamer is not initialized. Cannot stream to RTMP.")
        return False

    # Construct the pipeline description
    # Ensure audio is converted and encoded correctly for RTMP
    pipeline_desc = (
        f"filesrc location=\\\"{wav_path}\\\" ! wavparse ! audioconvert ! "
        f"avenc_aac ! queue ! flvmux name=mux streamable=true ! " # Use avenc_aac for broader compatibility
        f"rtmpsink location=\\\"{rtmp_url}\\\" sync=false async=false" # Added sync/async=false for potentially better sync
    )

    logging.info("Launching GStreamer pipeline for RTMP:\n%s", pipeline_desc)
    try:
        pipeline = Gst.parse_launch(pipeline_desc)
    except Exception as e:
        logging.error(f"Failed to parse GStreamer pipeline: {e}")
        return False

    pipeline.set_state(Gst.State.PLAYING)

    # Check if pipeline reached PLAYING state
    state_change_ret = pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
    if state_change_ret != Gst.State.PLAYING:
         logging.error(f"Failed to set pipeline to PLAYING state. Current state: {state_change_ret}")
         pipeline.set_state(Gst.State.NULL) # Clean up
         return False

    if not blocking:
        logging.info("GStreamer pipeline started in non-blocking mode.")
        # In non-blocking, we might want a way to monitor/stop later
        # For now, just return True indicating successful start
        return True # Indicate successful start

    # Wait until EOS or ERROR in blocking mode
    logging.info("GStreamer pipeline running in blocking mode...")
    bus = pipeline.get_bus()
    keep_running = True
    success = True
    while keep_running:
        msg = bus.timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR | Gst.MessageType.STATE_CHANGED
        )
        if msg:
            if msg.src == pipeline: # Only handle messages from the pipeline itself
                 if msg.type == Gst.MessageType.EOS:
                     logging.info("GStreamer pipeline reached End-Of-Stream.")
                     keep_running = False
                 elif msg.type == Gst.MessageType.ERROR:
                     err, dbg = msg.parse_error()
                     logging.error(f"GStreamer pipeline error: {err} ({dbg})")
                     keep_running = False
                     success = False
                 elif msg.type == Gst.MessageType.STATE_CHANGED:
                     # Optional: Log state changes for debugging
                     old_state, new_state, pending_state = msg.parse_state_changed()
                     # logging.debug(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")
                     pass

        else:
            # No message, check pipeline status briefly
            current_state = pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
            if current_state not in (Gst.State.PLAYING, Gst.State.PAUSED):
                 logging.warning(f"Pipeline unexpectedly left PLAYING/PAUSED state. Current: {current_state.value_nick}")
                 # This might indicate an issue, decide if we should stop
                 # keep_running = False # Optional: Stop if state changes unexpectedly
                 pass


    logging.info("Cleaning up GStreamer pipeline...")
    pipeline.set_state(Gst.State.NULL)
    return success # Return True if EOS, False if ERROR 