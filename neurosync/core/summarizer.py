import threading
import time
import os

from neurosync.core.scb_store import scb_store # Use the singleton instance
from neurosync.core.color_text import ColorText

# Configuration (can be overridden by environment variables)
DEFAULT_SUMMARIZER_INTERVAL = float(os.getenv("SUMMARIZER_INTERVAL_SECONDS", "3.0"))
DEFAULT_SUMMARIZER_TOKEN_BUDGET = int(os.getenv("SUMMARIZER_TOKEN_BUDGET", "200")) # Max tokens for the summary itself
DEFAULT_SUMMARY_SOURCE_LINES = int(os.getenv("SUMMARY_SOURCE_LINES", "50")) # How many recent lines to consider
DEFAULT_SUMMARY_MIN_SALIENCE = float(os.getenv("SUMMARY_MIN_SALIENCE", "0.4"))
DEFAULT_SUMMARY_KEEP_LAST_N = int(os.getenv("SUMMARY_KEEP_LAST_N", "15"))
DEFAULT_SUMMARIZER_DEBUG = os.getenv("SUMMARIZER_DEBUG", "False").lower() == "true"

class SummarizerThread(threading.Thread):
    """A background thread that periodically summarizes the SCB log."""
    def __init__(self, interval=DEFAULT_SUMMARIZER_INTERVAL,
                 token_budget=DEFAULT_SUMMARIZER_TOKEN_BUDGET,
                 source_lines=DEFAULT_SUMMARY_SOURCE_LINES,
                 min_salience=DEFAULT_SUMMARY_MIN_SALIENCE,
                 keep_last_n=DEFAULT_SUMMARY_KEEP_LAST_N,
                 debug=DEFAULT_SUMMARIZER_DEBUG,
                 stop_event=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.token_budget = token_budget
        self.source_lines = source_lines
        self.min_salience = min_salience
        self.keep_last_n = keep_last_n
        self.debug = debug
        self._stop_event = stop_event or threading.Event()
        self.name = "SummarizerThread"

    def run(self):
        if self.debug:
            print(f"{ColorText.BLUE}[{self.name}] Started. Interval={self.interval}s, Budget={self.token_budget} tokens.{ColorText.END}")
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                self.summarize()
            except Exception as e:
                print(f"{ColorText.RED}[{self.name}] Error during summarization: {e}{ColorText.END}")
                import traceback
                print(f"{ColorText.RED}Traceback:\n{traceback.format_exc()}{ColorText.END}")

            elapsed = time.monotonic() - start_time
            sleep_time = max(0, self.interval - elapsed)
            if self.debug and elapsed > self.interval:
                print(f"{ColorText.YELLOW}[{self.name}] Warning: Summarization took longer ({elapsed:.2f}s) than interval ({self.interval}s).{ColorText.END}")

            # Wait for the interval or until stop event is set
            self._stop_event.wait(sleep_time)

        if self.debug:
             print(f"{ColorText.BLUE}[{self.name}] Stopped.{ColorText.END}")

    def summarize(self):
        """Performs one summarization cycle."""
        if self.debug:
            print(f"{ColorText.BLUE}[{self.name}] Running summarization cycle...{ColorText.END}")

        # 1. Get recent entries from SCB
        entries = scb_store.get_log_entries(self.source_lines)
        if not entries:
            if self.debug:
                print(f"{ColorText.BLUE}[{self.name}] No entries to summarize.{ColorText.END}")
            return

        # 2. Filter entries for summary (MVP Logic)
        # Keep entries with high salience OR the most recent N entries
        salient_entries = []
        recent_entries_indices = set(range(min(len(entries), self.keep_last_n))) # Indices of the last N

        for i, entry in enumerate(entries):
             # Check salience (handle missing key gracefully)
             salience = entry.get('salience', 0.0)
             if salience >= self.min_salience:
                 salient_entries.append(entry)
             # Also include if it's one of the most recent N, even if low salience
             elif i in recent_entries_indices and entry not in salient_entries:
                 salient_entries.append(entry)

        if not salient_entries:
            if self.debug:
                print(f"{ColorText.BLUE}[{self.name}] No salient or recent entries found for summary.{ColorText.END}")
            scb_store.set_summary("") # Clear summary if nothing qualifies
            return

        # 3. Combine and truncate to token budget (using simple word count)
        summary_lines = []
        current_tokens = 0

        # Add entries chronologically (oldest first) from the filtered list
        # Since get_log_entries returns newest first, we reverse the filtered list
        for entry in reversed(salient_entries):
            text = entry.get('text', '')
            entry_tokens = len(text.split())

            # Simple prefix based on type/actor for context
            prefix = f"{entry.get('actor', 'Unknown')}: "
            if entry.get('type') == 'directive':
                prefix = f"[Directive from {entry.get('actor', 'Unknown')}] "
            elif entry.get('type') == 'speech':
                 prefix = f"AI: " # Simpler prefix for AI speech

            line = prefix + text
            line_tokens = len(line.split())

            if current_tokens + line_tokens <= self.token_budget:
                summary_lines.append(line)
                current_tokens += line_tokens
            else:
                # If adding this line exceeds budget, stop or truncate?
                # For MVP, just stop adding more lines.
                if self.debug:
                     print(f"{ColorText.YELLOW}[{self.name}] Token budget ({self.token_budget}) reached. Truncating summary.{ColorText.END}")
                break

        final_summary = "\n".join(summary_lines)

        # 4. Update SCB summary
        scb_store.set_summary(final_summary)

        if self.debug:
             print(f"{ColorText.BLUE}[{self.name}] Summarization complete. Tokens={current_tokens}, Lines={len(summary_lines)}.{ColorText.END}")

    def stop(self):
        """Signals the thread to stop."""
        self._stop_event.set()

# Example Usage (can be run directly for testing)
if __name__ == "__main__":
    print("--- Summarizer Test --- ")
    # Ensure config and SCBStore are initialized (loads .env)
    from neurosync.core.scb_store import scb_store # Re-import to ensure init
    import dotenv
    dotenv.load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

    # Re-initialize SCB with debug explicitly for testing
    store = scb_store # Use the singleton
    store.debug = True
    store.__init__(use_redis=os.getenv("USE_REDIS_SCB", "False").lower() == "true", redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"), debug=True) # Force re-init for debug

    print(f"SCB Using Redis: {store.use_redis}")

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

    # Add some dummy data
    print("\nAdding dummy data...")
    store.append_chat("Low salience message 1", salience=0.1)
    time.sleep(0.01)
    store.append_chat("Medium salience message 2", salience=0.4)
    time.sleep(0.01)
    store.append_directive("High salience directive 3", salience=0.9)
    time.sleep(0.01)
    store.append_chat("Low salience message 4", salience=0.1)
    time.sleep(0.01)
    store.append_chat("Medium salience message 5", salience=0.5)
    time.sleep(0.01)
    store.append_chat("Recent low salience message 6", salience=0.1)
    time.sleep(0.01)
    store.append_chat("Very recent low salience 7", salience=0.1)

    print("\nInitial SCB Log (last 10):")
    entries = store.get_log_entries(10)
    for e in reversed(entries): print(e)

    print("\nRunning Summarizer Thread...")
    stop_event = threading.Event()
    summarizer = SummarizerThread(interval=1, token_budget=50, debug=True, stop_event=stop_event)
    summarizer.start()

    # Let it run a few times
    time.sleep(3.5)

    print("\nStopping Summarizer Thread...")
    summarizer.stop()
    summarizer.join()

    print("\nFinal Summary from SCB:")
    final_summary = store.get_summary()
    print(final_summary)
    print(f"(Word count: {len(final_summary.split())})")

    print("\n--- Test Complete --- ") 