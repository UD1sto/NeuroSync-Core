# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain a to use this software commercially.

# # livelink_init.py

import socket
from livelink.connect.pylivelinkface import PyLiveLinkFace, FaceBlendShape
import time
import sys

# Define potential IP addresses to try (in order of preference)
POTENTIAL_UDP_IPS = ["127.0.0.1", "10.5.0.2", "0.0.0.0"]
UDP_PORT = 11111

def create_socket_connection():
    """Create a socket connection trying multiple IP addresses"""
    socket_error = None
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Try each IP address in order
    for ip in POTENTIAL_UDP_IPS:
        try:
            print(f"Attempting to connect to LiveLink at {ip}:{UDP_PORT}...")
            s.connect((ip, UDP_PORT))
            print(f"✅ LiveLink socket connection established successfully to {ip}:{UDP_PORT}")
            return s  # Return the socket if successful
        except socket.error as e:
            socket_error = e
            print(f"⚠️ Warning: Could not connect to LiveLink at {ip}:{UDP_PORT}: {e}")
            # Continue to the next IP address
    
    print("⚠️ Could not connect to any LiveLink endpoints - animations may not work")
    
    # Return socket even if all connection attempts failed - UDP can still send packets
    # We'll use the last attempted IP for sending
    if s:
        return s
    
    # Create dummy socket as last resort
    print(f"⚠️ Warning: Failed to create LiveLink socket: {socket_error}")
    print("⚠️ Creating dummy socket - animations will not be sent to Unreal Engine")
    
    # Create dummy socket object with sendall method that does nothing
    class DummySocket:
        def sendall(self, *args, **kwargs):
            pass
        def close(self):
            pass
            
    return DummySocket()

def initialize_py_face():
    py_face = PyLiveLinkFace()
    initial_blendshapes = [0.0] * 61
    for i, value in enumerate(initial_blendshapes):
        py_face.set_blendshape(FaceBlendShape(i), float(value))
    return py_face
