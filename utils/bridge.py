import os
import threading
import time
from utils.config import ENABLE_SYSTEM2_BRIDGE, BRIDGE_FILE_PATH, MOCK_SYSTEM2, BRIDGE_DEBUG

class BridgeCache:
    """Lightweight reader with in-process cache of the bridge file."""
    _text: str = ""
    _mtime: float = 0.0
    _lock = threading.Lock()

    @classmethod
    def read(cls) -> str:
        if not ENABLE_SYSTEM2_BRIDGE:
            return ""
        try:
            mtime = os.path.getmtime(BRIDGE_FILE_PATH)
            if mtime != cls._mtime:
                with open(BRIDGE_FILE_PATH, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                with cls._lock:
                    cls._text = content
                    cls._mtime = mtime
                    if BRIDGE_DEBUG:
                        print("[BridgeCache] Updated text (truncated):", cls._text[:120])
        except FileNotFoundError:
            # bridge file absent – treat as empty
            with cls._lock:
                cls._text = ""
        return cls._text

# ---------------- Mock writer (for dev) -----------------

def _mock_writer():
    counter = 0
    max_lines = 50
    while True:
        line = f"[MOCK SYSTEM2] insight #{counter}"
        # Append line, then trim file to last max_lines lines
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
        except Exception as e:
            print("[Bridge Mock] Error writing bridge file:", e)
        counter += 1
        time.sleep(10)  # update every 10s

if ENABLE_SYSTEM2_BRIDGE and MOCK_SYSTEM2:
    t = threading.Thread(target=_mock_writer, daemon=True)
    t.start() 