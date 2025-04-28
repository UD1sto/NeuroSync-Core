"""
NeuroSync Client

This module provides the command-line client for NeuroSync that:
1. Takes user input
2. Generates text via LLM
3. Converts to speech via TTS
4. Forwards to the local API for animation

It will be a proper migration of neurosync_client.py with adjusted imports.
"""

# This will be filled with the actual migrated code from neurosync_client.py
# with imports adjusted to use the new package structure.

# For now, provide a thin compatibility layer  
import os
import sys

# Add the root directory to sys.path to ensure imports work correctly
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Re-export the main function from neurosync_client.py
from neurosync_client import main

if __name__ == "__main__":
    main() 