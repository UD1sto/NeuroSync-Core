import os
import json
import time
import threading
from collections import deque
import redis # type: ignore
from typing import List, Dict

# Assuming config is in neurosync.core.config
from neurosync.core.config import config as app_config # Rename to avoid conflict
from neurosync.core.color_text import ColorText

# Default configuration (can be overridden by env vars via app_config)
DEFAULT_USE_REDIS = os.getenv("USE_REDIS_SCB", "False").lower() == "true"
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_SCB_MAX_LINES = int(os.getenv("SCB_MAX_LINES", "1000"))
DEFAULT_SCB_SUMMARY_KEY = os.getenv("SCB_SUMMARY_KEY", "scs:summary")
DEFAULT_SCB_LOG_KEY = os.getenv("SCB_LOG_KEY", "scs:log")
DEFAULT_SCB_DEBUG = os.getenv("SCB_DEBUG", "False").lower() == "true"

class SCBStore:
    """
    Manages the Shared Cognitive Blackboard (SCB) storage.

    Provides a unified API for interacting with the SCB, supporting both
    Redis (for production/multi-container setups) and an in-memory deque
    (for simple local development without Redis).
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, use_redis=DEFAULT_USE_REDIS, redis_url=DEFAULT_REDIS_URL,
                 max_lines=DEFAULT_SCB_MAX_LINES, log_key=DEFAULT_SCB_LOG_KEY,
                 summary_key=DEFAULT_SCB_SUMMARY_KEY, debug=DEFAULT_SCB_DEBUG):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.use_redis = use_redis
            self.redis_url = redis_url
            self.max_lines = max_lines
            self.log_key = log_key
            self.summary_key = summary_key
            self.debug = debug
            self._redis_client = None
            self._memory_log = deque(maxlen=max_lines)
            self._memory_summary = ""
            self._init_lock = threading.Lock()

            if self.use_redis:
                self._initialize_redis()
            else:
                if self.debug:
                    print(f"{ColorText.YELLOW}[SCBStore] Using in-memory deque (Redis disabled){ColorText.END}")

            self._initialized = True

    def _initialize_redis(self):
        """Initialize the Redis client connection."""
        if self._redis_client is None:
             with self._init_lock: # Ensure thread-safe initialization
                 if self._redis_client is None:
                    try:
                        self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                        self._redis_client.ping()
                        if self.debug:
                            print(f"{ColorText.GREEN}[SCBStore] Connected to Redis at {self.redis_url}{ColorText.END}")
                    except redis.exceptions.ConnectionError as e:
                        print(f"{ColorText.RED}[SCBStore] Failed to connect to Redis: {e}{ColorText.END}")
                        print(f"{ColorText.YELLOW}[SCBStore] Falling back to in-memory deque.{ColorText.END}")
                        self.use_redis = False # Fallback to memory
                        self._redis_client = None # Ensure it's None
                    except Exception as e:
                         print(f"{ColorText.RED}[SCBStore] An unexpected error occurred during Redis initialization: {e}{ColorText.END}")
                         print(f"{ColorText.YELLOW}[SCBStore] Falling back to in-memory deque.{ColorText.END}")
                         self.use_redis = False # Fallback
                         self._redis_client = None

    def _get_redis_client(self):
         """Get the Redis client, initializing if needed."""
         if self.use_redis and self._redis_client is None:
              self._initialize_redis()
         return self._redis_client


    def append(self, entry: dict):
        """
        Appends a new entry (dict) to the SCB log.
        Ensures required fields ('t', 'type', 'actor', 'text') are present.
        Automatically trims the log if it exceeds max_lines.
        """
        if not all(k in entry for k in ['type', 'actor', 'text']):
            print(f"{ColorText.RED}[SCBStore] Error: Entry missing required fields (type, actor, text): {entry}{ColorText.END}")
            return

        # Add timestamp if missing
        if 't' not in entry:
            entry['t'] = int(time.time())

        entry_json = json.dumps(entry)
        if self.debug:
             # Truncate long text for logging
             log_entry = entry.copy()
             if len(log_entry.get('text', '')) > 100:
                 log_entry['text'] = log_entry['text'][:100] + '...'
             print(f"{ColorText.CYAN}[SCBStore Append] {log_entry}{ColorText.END}")


        client = self._get_redis_client()
        if self.use_redis and client:
            try:
                # LPUSH adds to the left (head), then LTRIM keeps the N newest entries
                # Note: This assumes newest entries are added to the left.
                # If using RPUSH, trim should be LTRIM 0 -(max_lines)
                pipe = client.pipeline()
                pipe.lpush(self.log_key, entry_json)
                pipe.ltrim(self.log_key, 0, self.max_lines - 1)
                pipe.execute()
            except redis.exceptions.ConnectionError as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Connection failed: {e}. Falling back to memory.{ColorText.END}")
                self.use_redis = False
                self._redis_client = None # Ensure client is cleared
                self._memory_log.appendleft(entry) # Add to memory log on fallback
            except Exception as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Failed to append/trim: {e}{ColorText.END}")
                # Optionally fallback here too, depending on desired robustness
        else:
            # In-memory log automatically handles maxlen when appending to the left
            self._memory_log.appendleft(entry)

    def append_chat(self, text: str, actor: str = "user", salience: float = 0.3):
        """Helper to append a chat message event."""
        entry = {
            "type": "event",
            "actor": actor,
            "text": text,
            "salience": salience
        }
        self.append(entry)

    def append_directive(self, text: str, actor: str = "planner", ttl: int = 15):
        """Helper to append a directive."""
        entry = {
            "type": "directive",
            "actor": actor,
            "text": text,
            "ttl": ttl
            # 'salience' could be added, maybe high by default for directives
        }
        self.append(entry)

    def get_log_entries(self, count: int) -> List[Dict]:
        """Retrieves the most recent 'count' entries from the SCB log."""
        if count <= 0:
            return []

        client = self._get_redis_client()
        if self.use_redis and client:
            try:
                # LRANGE 0 (count-1) gets the first 'count' elements (newest if LPUSHed)
                entries_json = client.lrange(self.log_key, 0, count - 1)
                # Parse JSON strings back to dictionaries
                entries = [json.loads(entry) for entry in entries_json]
                return entries
            except redis.exceptions.ConnectionError as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Connection failed: {e}. Falling back to memory.{ColorText.END}")
                self.use_redis = False
                self._redis_client = None
                # Fallback to memory below
            except Exception as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Failed to get log entries: {e}{ColorText.END}")
                return [] # Return empty on error

        # In-memory fallback or if Redis disabled
        # Deque stores newest on the left, so slice directly
        return list(self._memory_log)[:count]

    def get_recent_chat(self, count: int = 3) -> str:
        """Gets the last 'count' chat messages as a formatted string."""
        entries = self.get_log_entries(self.max_lines) # Get more entries to filter
        chat_messages = []
        for entry in reversed(entries): # Iterate oldest to newest from the retrieved slice
            if entry.get('type') == 'event' and entry.get('actor') == 'user':
                 chat_messages.append(f"User: {entry.get('text', '')}")
                 if len(chat_messages) >= count:
                    break
            elif entry.get('type') == 'speech' and entry.get('actor') == 'vtuber': # Assuming vtuber speech is logged
                 chat_messages.append(f"AI: {entry.get('text', '')}")
                 if len(chat_messages) >= count:
                     break

        return "\n".join(chat_messages)


    def get_summary(self) -> str:
        """Retrieves the current SCB summary."""
        client = self._get_redis_client()
        if self.use_redis and client:
            try:
                summary = client.get(self.summary_key)
                return summary if summary else ""
            except redis.exceptions.ConnectionError as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Connection failed: {e}. Falling back to memory.{ColorText.END}")
                self.use_redis = False
                self._redis_client = None
                # Fallback to memory below
            except Exception as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Failed to get summary: {e}{ColorText.END}")
                return self._memory_summary # Return memory summary on Redis error

        # In-memory fallback or if Redis disabled
        return self._memory_summary

    def set_summary(self, summary_text: str):
        """Updates the SCB summary."""
        if self.debug:
             print(f"{ColorText.MAGENTA}[SCBStore Set Summary] Length={len(summary_text)}, Text='{summary_text[:100]}...'{ColorText.END}")

        client = self._get_redis_client()
        if self.use_redis and client:
            try:
                client.set(self.summary_key, summary_text)
            except redis.exceptions.ConnectionError as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Connection failed: {e}. Falling back to memory.{ColorText.END}")
                self.use_redis = False
                self._redis_client = None
                self._memory_summary = summary_text # Set memory summary on fallback
            except Exception as e:
                print(f"{ColorText.RED}[SCBStore Redis Error] Failed to set summary: {e}{ColorText.END}")
                # Optionally fallback here too
        else:
            # In-memory fallback or if Redis disabled
            self._memory_summary = summary_text

    def get_slice(self, token_budget: int = 600) -> dict:
        """
        Retrieves a slice of the SCB containing the summary and recent log entries,
        respecting a token budget (approximated by word count for now).
        """
        summary = self.get_summary()
        # Simple word count approximation for token budget
        summary_tokens = len(summary.split())
        remaining_budget = token_budget - summary_tokens

        window = []
        if remaining_budget > 0:
            # Fetch more entries than needed initially, then trim
            potential_entries = self.get_log_entries(self.max_lines) # Get all for accurate budget slicing
            current_tokens = 0
            # Iterate from newest to oldest
            for entry in potential_entries:
                entry_text = entry.get('text', '')
                entry_tokens = len(entry_text.split())
                if current_tokens + entry_tokens <= remaining_budget:
                    window.append(entry)
                    current_tokens += entry_tokens
                else:
                    break # Stop adding entries once budget exceeded
            # Reverse window to maintain chronological order (oldest first) in the slice
            window.reverse()


        slice_data = {
            "summary": summary,
            "window": window
        }

        if self.debug:
            actual_tokens = summary_tokens + sum(len(e.get('text','').split()) for e in window)
            print(f"{ColorText.BLUE}[SCBStore Get Slice] Budget={token_budget}, Summary Tokens={summary_tokens}, Window Tokens={actual_tokens-summary_tokens}, Total={actual_tokens}, Window Entries={len(window)}{ColorText.END}")


        return slice_data

    def get_full(self) -> dict:
        """Returns the entire SCB whiteboard (summary + full log window)."""
        summary = self.get_summary()
        # Retrieve up to max_lines entries (newest first)
        entries = self.get_log_entries(self.max_lines)
        # Reverse to chronological order (oldest first)
        window = list(reversed(entries))
        return {"summary": summary, "window": window}


# --- Singleton Instance ---
# Instantiate the store so it can be imported and used directly
scb_store = SCBStore()

# Example Usage (can be run directly for testing)
if __name__ == "__main__":
    print("--- SCBStore Test --- ")
    # Ensure config is loaded if run directly
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    # Re-initialize with potential env vars from dotenv
    store = SCBStore(
        use_redis=os.getenv("USE_REDIS_SCB", "False").lower() == "true",
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        debug=True
    )

    print(f"Using Redis: {store.use_redis}")

    # Clear existing test data (optional)
    if store.use_redis and store._get_redis_client():
        try:
            print("Clearing existing Redis test keys...")
            store._get_redis_client().delete(store.log_key, store.summary_key)
        except Exception as e:
             print(f"Error clearing Redis keys: {e}")
    else:
        print("Clearing in-memory store...")
        store._memory_log.clear()
        store._memory_summary = ""


    store.append_chat("Hello there!", actor="user1")
    time.sleep(0.1)
    store.append_directive("Think about cats.", actor="planner")
    time.sleep(0.1)
    store.append_chat("General Kenobi!", actor="user2", salience=0.7)
    time.sleep(0.1)
    store.append({"type": "speech", "actor": "vtuber", "text": "Ah, a bold one!"})

    print("\nRecent Log Entries (3):")
    entries = store.get_log_entries(3)
    for e in entries: print(e)

    print("\nRecent Chat (2):")
    chat = store.get_recent_chat(2)
    print(chat)

    print("\nSetting Summary...")
    store.set_summary("Conversation started with Star Wars references.")

    print("\nGetting Summary:")
    summary = store.get_summary()
    print(summary)

    print("\nGetting Slice (budget 50 words):")
    slice_data = store.get_slice(token_budget=50)
    print(json.dumps(slice_data, indent=2))

    print("\nGetting Slice (budget 10 words):")
    slice_data_small = store.get_slice(token_budget=10)
    print(json.dumps(slice_data_small, indent=2))

    print("\nGetting Full SCB:")
    full_scb = store.get_full()
    print(json.dumps(full_scb, indent=2))

    print("\n--- Test Complete --- ") 