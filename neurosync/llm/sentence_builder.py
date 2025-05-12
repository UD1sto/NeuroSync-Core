# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import re
import string
from queue import Queue
import os

class SentenceBuilder:
    """
    Accumulates tokens into sentences (or partial chunks) and flushes
    complete chunks to a provided chunk_queue for further processing.
    """
    SENTENCE_ENDINGS = {'.', '!', '?'}
    ABBREVIATIONS = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.",
        "vs.", "e.g.", "i.e.", "etc.", "p.s."
    }

    def __init__(self, chunk_queue, max_chunk_length=500, flush_token_count=300, respect_sentence_endings: bool = True):
        """
        Args:
            chunk_queue (Queue): Queue where ready-to-speak text chunks will be published.
            max_chunk_length (int): Maximum character length of the buffered text before an automatic flush.
            flush_token_count (int): Maximum number of tokens to buffer before an automatic flush.
            respect_sentence_endings (bool): When *True* (default) the builder will flush as soon as a sentence-
                ending punctuation (., !, ?) is encountered. When *False* it will ignore punctuation boundaries
                and rely only on *max_chunk_length* / *flush_token_count* / newlines.  Disabling this can be
                helpful for truly continuous audio streaming because many TTS engines introduce long pauses at
                sentence boundaries which makes the VTuber speech sound staccato.
        """
        self.chunk_queue = chunk_queue
        self.max_chunk_length = max_chunk_length
        self.flush_token_count = flush_token_count
        self.respect_sentence_endings = respect_sentence_endings

        # Internal buffer to accumulate tokens
        self.buffer = []
        self.token_count = 0

    def add_token(self, token: str):
        """
        Add a token to the internal buffer.
        Flush the buffer if:
          - The token contains a newline (considered a sentence break).
          - The combined length exceeds max_chunk_length.
          - The token count exceeds flush_token_count.
          - A sentence-ending punctuation is encountered (unless it's an abbreviation).
        """
        self.buffer.append(token)
        self.token_count += 1

        # Flush immediately if the token contains a newline.
        if '\n' in token:
            self._flush_buffer()
            return

        # Flush if raw character length is exceeded
        if self._current_length() >= self.max_chunk_length:
            self._flush_buffer()
            return

        # Flush if token count is exceeded
        if self.token_count >= self.flush_token_count:
            self._flush_buffer()
            return

        # Flush if we detect a sentence end (unless it's an abbreviation) *and* this behaviour is enabled
        if self.respect_sentence_endings and self._ends_sentence(token):
            if not self._is_abbreviation():
                self._flush_buffer()

    def flush_remaining(self):
        """
        Flush any remaining tokens in the buffer.
        """
        if self.buffer:
            self._flush_buffer(force=True)

    def _current_length(self) -> int:
        """
        Return the combined length of the tokens in the buffer.
        """
        return sum(len(t) for t in self.buffer)

    def _ends_sentence(self, token: str) -> bool:
        """
        Return True if the token ends with punctuation that typically ends a sentence.
        """
        token = token.strip()
        if not token:
            return False
        return token[-1] in self.SENTENCE_ENDINGS

    def _is_abbreviation(self) -> bool:
        """
        Check if the last word in the buffer is an abbreviation.
        For example, "Dr." should not trigger a flush.
        """
        combined = ''.join(self.buffer).strip()
        if not combined:
            return False
        words = combined.split()
        if not words:
            return False
        last_word = words[-1].lower()
        return last_word in self.ABBREVIATIONS

    def _flush_buffer(self, force=False):
        chunk_text_val = ''.join(self.buffer).strip()
        # Debug: emit log about flushing behaviour (can be toggled via env)
        if os.getenv("SB_DEBUG", "false").lower() == "true":
            flush_reason = "forced" if force else "auto"
            print(f"[SentenceBuilder] Flushing buffer ({flush_reason}): '{chunk_text_val[:80]}{'...' if len(chunk_text_val) > 80 else ''}'")
        # Clean the chunk text using the helper function.
        clean_chunk = clean_text_for_tts(chunk_text_val)
        if clean_chunk:  # Only enqueue if there's something meaningful.
            self.chunk_queue.put(clean_chunk)
        self.buffer = []
        self.token_count = 0

    def run(self, token_queue: Queue):
        """
        Continuously read tokens from the provided token_queue,
        process them, and flush when appropriate.
        """
        while True:
            token = token_queue.get()  # Wait until a token is available.
            if token is None:  # Sentinel value indicates no more tokens.
                break
            self.add_token(token)
            token_queue.task_done()
        # Flush any remaining tokens after exiting loop.
        self.flush_remaining()


def clean_text_for_tts(text: str) -> str:
    """
    Remove unwanted patterns from text:
      - Anything between asterisks, e.g. *some words*
      - Anything between parentheses, e.g. (some words)
    Then trim whitespace. If the result is empty or only punctuation,
    return an empty string.
    """
    # Remove text enclosed in asterisks (e.g., *example*)
    text = re.sub(r'\*[^*]+\*', '', text)
    # Remove text enclosed in parentheses (e.g., (example))
    text = re.sub(r'\([^)]*\)', '', text)
    # Trim whitespace
    clean_text = text.strip()
    # If the cleaned text is empty, exactly '...', or only punctuation/spaces, return empty.
    if (not clean_text or clean_text == "..." or
        all(char in string.punctuation or char.isspace() for char in clean_text)):
        return ""
    return clean_text 