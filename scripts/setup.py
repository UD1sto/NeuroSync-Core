import os
import subprocess
import sys
import platform
from setuptools import setup, find_packages

def get_cuda_version():
    """Get the CUDA version from nvidia-smi or return None if not available"""
    try:
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                return line.split('CUDA Version:')[1].strip()
        return None
    except:
        return None

def install_pytorch():
    """Install PyTorch with appropriate CUDA support"""
    cuda_version = get_cuda_version()
    
    # If CUDA is not available, install CPU version
    if cuda_version is None:
        print("No CUDA detected, installing CPU version of PyTorch")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision'])
        return
    
    # Convert CUDA version string to float for comparison
    try:
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        cuda_float = float(cuda_major_minor)
    except:
        print(f"Could not parse CUDA version: {cuda_version}, installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision'])
        return
    
    print(f"Detected CUDA version: {cuda_version}")
    
    # Install PyTorch based on CUDA version
    if cuda_float >= 12.8:
        print("Installing PyTorch with CUDA 12.8 support (nightly)")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torchvision', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
        except:
            print("Failed to install nightly build, falling back to stable CUDA 12.1")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                                  '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 12.1:
        print("Installing PyTorch with CUDA 12.1 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 11.8:
        print("Installing PyTorch with CUDA 11.8 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu118'])
    elif cuda_float >= 11.7:
        print("Installing PyTorch with CUDA 11.7 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu117'])
    else:
        print(f"CUDA {cuda_version} is older than recommended. Installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])

def main():
    # First install PyTorch with appropriate CUDA support
    install_pytorch()
    
    # Now continue with normal setup
    setup(
        name="neurosync-core",
        version="0.1.0",
        packages=find_packages(),
        install_requires=[
            # List other dependencies here, excluding PyTorch which is installed separately
            "numpy",
            "scipy",
            "flask",
            "pydub",
            "sounddevice",
            "transformers",
            "librosa",
            # Add other dependencies as needed
        ],
        description="NeuroSync-Core: A neural interface for facial animation",
        author="Georg",
        author_email="your.email@example.com",
    )

if __name__ == "__main__":
    main() 