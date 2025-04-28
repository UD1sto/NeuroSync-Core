# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import socket
import os # Added

# Use environment variables for configuration
EMOTE_SERVER_ADDRESS = os.getenv("EMOTE_SERVER_ADDRESS", "127.0.0.1")
EMOTE_SERVER_PORT = int(os.getenv("EMOTE_SERVER_PORT", 12345)) # Default port example

class EmoteConnect:

    @classmethod
    def send_emote(cls, emote_name: str):
        """
        Sends the provided emote name to the configured server address and port.
        Creates a new connection for each call.
        """
        # Validate the emote name
        if not emote_name or not emote_name.strip():
            print("Error: Emote name cannot be empty or just whitespace.")
            return

        server_address = EMOTE_SERVER_ADDRESS
        server_port = EMOTE_SERVER_PORT

        try:
            # Create a new socket for each connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                # Set a timeout for connection attempts
                client.settimeout(2.0)
                client.connect((server_address, server_port))
                # Convert the emote name to bytes (UTF-8 encoded), trim whitespace
                message_bytes = emote_name.strip().encode('utf-8')
                # Send the emote using sendall
                client.sendall(message_bytes)
                print(f"Sent emote '{emote_name.strip()}' to {server_address}:{server_port}")
        except socket.timeout:
             print(f"Error: Connection to emote server {server_address}:{server_port} timed out.")
        except ConnectionRefusedError:
             print(f"Error: Connection refused by emote server at {server_address}:{server_port}. Is it running?")
        except Exception as ex:
            print(f"Error sending emote '{emote_name}': {ex}")

# Example usage (can be removed if this is purely a library)
# if __name__ == "__main__":
#     import threading
#     EmoteConnect.send_emote("Wave")
#     # To send an emote on its own thread (non-blocking):
#     thread = threading.Thread(target=EmoteConnect.send_emote, args=("Nod",))
#     thread.start()
#     thread.join() 