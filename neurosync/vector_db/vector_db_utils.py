# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

# neurosync/vector_db/vector_db_utils.py

from datetime import datetime, timezone
# Corrected relative imports
from .get_embedding import get_embedding
from .vector_db import VectorDB

def update_system_message_with_context(user_input: str, base_system_message: str, vector_db: VectorDB, top_n: int = 4) -> str:
    if not vector_db:
         print("Warning: VectorDB not provided to update_system_message_with_context.")
         return base_system_message

    retrieval_embedding = get_embedding(user_input, use_openai=False) # Assuming local embedding is desired
    if retrieval_embedding is None:
         print("Warning: Failed to get embedding for context retrieval.")
         context_string = ""
    else:
         context_string = vector_db.get_context_string(retrieval_embedding, top_n=top_n)

    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")
    return f"{base_system_message}{context_string}\nThe current time and date is: {current_time}"

def add_exchange_to_vector_db(user_input: str, response: str, vector_db: VectorDB):
    if not vector_db:
         print("Warning: VectorDB not provided to add_exchange_to_vector_db.")
         return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")
    combined_text = f"User: {user_input}\nYou: {response}\nTimestamp: {timestamp}\n"
    combined_embedding = get_embedding(combined_text, use_openai=False) # Assuming local embedding

    if combined_embedding is not None:
        vector_db.add_entry(combined_embedding, combined_text)
    else:
         print("Warning: Failed to get embedding for adding exchange to vector DB.") 