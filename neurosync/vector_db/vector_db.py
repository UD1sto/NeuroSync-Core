# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import os
import json
import numpy as np

# Define the path for the persistent vector DB JSON file.
# Consider making this path configurable via environment variable or config file
VECTOR_DB_FILE = os.path.join("chat_logs", "vector_db.json")

class VectorDB:
    def __init__(self, db_file: str = VECTOR_DB_FILE):
        self.db_file = db_file
        self.entries = []
        self.load()

    def load(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading vector DB from {self.db_file}: {e}")
                self.entries = []
        else:
            print(f"Vector DB file not found at {self.db_file}, initializing empty DB.")
            self.entries = []

    def save(self):
        try:
            # Ensure directory exists before saving
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=4)
        except Exception as e:
            print(f"Error saving vector DB to {self.db_file}: {e}")

    def add_entry(self, embedding: list, text: str, metadata: dict = None):
        # Basic validation
        if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
            print("Warning: Invalid embedding format. Expected list of numbers.")
            return
        if not isinstance(text, str):
             print("Warning: Invalid text format. Expected string.")
             return

        # Example embedding length check (adjust if needed)
        # if len(embedding) != 768:
        #     print(f"Warning: Embedding length is {len(embedding)}, expected 768.")

        entry = {
            "embedding": embedding,
            "text": text
        }
        if metadata:
            if not isinstance(metadata, dict):
                 print("Warning: Invalid metadata format. Expected dict.")
            else:
                 entry["metadata"] = metadata

        self.entries.append(entry)
        self.save()

    def cosine_similarity(self, vec1: list, vec2: list) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            # print("Warning: Cannot compute similarity for invalid or mismatched vectors.")
            return 0.0

        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # Ensure dot product result is float before returning
        similarity = float(np.dot(arr1, arr2) / (norm1 * norm2))
        # Clip similarity to avoid potential floating point issues outside [-1, 1]
        return max(-1.0, min(1.0, similarity))

    def search(self, query_embedding: list, top_n: int = 4) -> list:
        if not isinstance(query_embedding, list) or not query_embedding:
             print("Warning: Invalid query embedding provided for search.")
             return []

        results = []
        for entry in self.entries:
            if "embedding" not in entry or not isinstance(entry["embedding"], list):
                 print(f"Warning: Skipping entry with invalid embedding: {entry.get('text', '?')}")
                 continue
            sim = self.cosine_similarity(query_embedding, entry["embedding"])
            results.append({"entry": entry, "similarity": sim})

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_n]

    def get_context_string(self, query_embedding: list, top_n: int = 4, max_length: int = 4000) -> str:
        results = self.search(query_embedding, top_n)
        if not results:
            return ""

        lines = ["\n\nRelated Memory:"]
        current_length = len(lines[0])

        for result in results:
            text = result["entry"].get("text", "")
            similarity = result["similarity"]
            line = f"{text} (similarity: {similarity:.3f})"
            line_length = len(line) + 1 # +1 for newline

            if current_length + line_length <= max_length:
                lines.append(line)
                current_length += line_length
            else:
                break # Stop adding lines if max length is exceeded

        return "\n".join(lines)

# Instantiate a default DB instance (optional, depends on usage)
# vector_db = VectorDB() 