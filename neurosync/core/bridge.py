import os
import threading
import time
# Corrected config import
from neurosync.core.config import ENABLE_SYSTEM2_BRIDGE, BRIDGE_FILE_PATH, MOCK_SYSTEM2, BRIDGE_DEBUG
# Import the new SCB store
from neurosync.core.scb_store import scb_store, DEFAULT_USE_REDIS # Import singleton and flag
from neurosync.core.color_text import ColorText

class BridgeCache:
    """Lightweight reader for System-2 bridge info.

    Reads either from the new SCBStore (if Redis is enabled)
    or falls back to the legacy file-based bridge.
    Caches the result in-process.
    """
    _text: str = ""
    _mtime: float = 0.0
    _lock = threading.Lock()

    @classmethod
    def read(cls) -> str:
        # If the entire bridge system is disabled, return empty
        if not ENABLE_SYSTEM2_BRIDGE:
            return ""

        # Determine if we should use SCB (Redis) or legacy file
        # Use the default from scb_store logic which checks env var
        use_scb = DEFAULT_USE_REDIS

        current_content = ""
        needs_update = False

        if use_scb:
            # --- Read from SCBStore (Summary) ---
            try:
                summary = scb_store.get_summary()
                # Use a simple timestamp check for changes (or check against cached text)
                # For simplicity, let's just compare text content
                if summary != cls._text:
                    current_content = summary
                    needs_update = True
                    # Update pseudo mtime to reflect change
                    cls._mtime = time.time()
                else:
                    current_content = cls._text # Use cached value
            except Exception as e:
                print(f"{ColorText.RED}[BridgeCache SCB Error] Failed to read summary: {e}{ColorText.END}")
                # Fallback to legacy file read? Or return cached? For now, return cached.
                current_content = cls._text
        else:
            # --- Read from Legacy File ---
            try:
                mtime = os.path.getmtime(BRIDGE_FILE_PATH)
                if mtime != cls._mtime:
                    with open(BRIDGE_FILE_PATH, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    current_content = content
                    needs_update = True
                    cls._mtime = mtime # Update mtime from file
                else:
                    current_content = cls._text # Use cached value
            except FileNotFoundError:
                # bridge file absent – treat as empty, clear cache if needed
                if cls._text != "":
                    needs_update = True
                current_content = ""
            except Exception as e:
                 print(f"{ColorText.RED}[BridgeCache File Error] Failed to read {BRIDGE_FILE_PATH}: {e}{ColorText.END}")
                 current_content = cls._text # Return cached on error

        # Update cache if needed
        if needs_update:
            with cls._lock:
                cls._text = current_content
                if BRIDGE_DEBUG:
                    source = "SCB" if use_scb else "File"
                    print(f"{ColorText.YELLOW}[BridgeCache Updated ({source})]{ColorText.END} Text (truncated): {cls._text[:120]}")

        return cls._text

# ---------------- Mock writer (for dev) -----------------
# This mock writer only makes sense for the legacy file system now.
# If using Redis, System-2 should write via POST /directive or directly to Redis.

def _mock_writer():
    counter = 0
    max_lines = 50
    while True:
        # Check if we are in file mode before writing
        if not DEFAULT_USE_REDIS:
            line = f"[MOCK SYSTEM2 FILE] insight #{counter}"
            try:
                existing = []
                if os.path.exists(BRIDGE_FILE_PATH):
                    with open(BRIDGE_FILE_PATH, "r", encoding="utf-8") as rf:
                        existing = rf.read().splitlines()
                existing.append(line)
                if len(existing) > max_lines:
                    existing = existing[-max_lines:]
                with open(BRIDGE_FILE_PATH, "w", encoding="utf-8") as wf:
                    wf.write("\n".join(existing) + "\n")
                if BRIDGE_DEBUG:
                     print(f"{ColorText.BLUE}[Bridge Mock Writer] Updated {BRIDGE_FILE_PATH}{ColorText.END}")
            except Exception as e:
                print(f"{ColorText.RED}[Bridge Mock Writer Error] {e}{ColorText.END}")
            counter += 1
        else:
            # If using Redis, the mock writer does nothing
             if BRIDGE_DEBUG:
                 # Print occasionally to show it's alive but inactive
                 if counter % 6 == 0: # Print every minute
                    print(f"{ColorText.YELLOW}[Bridge Mock Writer] Inactive (USE_REDIS_SCB=true){ColorText.END}")
             counter += 1 # Increment counter even when inactive

        time.sleep(10)  # update check every 10s

# Start mock writer only if bridge enabled AND mock enabled
if ENABLE_SYSTEM2_BRIDGE and MOCK_SYSTEM2:
    print(f"{ColorText.YELLOW}[Bridge Mock] Initializing mock writer thread... (Mode: {'File' if not DEFAULT_USE_REDIS else 'Inactive/Redis'}){ColorText.END}")
    t = threading.Thread(target=_mock_writer, daemon=True)
    t.start() 