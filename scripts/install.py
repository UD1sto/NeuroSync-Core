#!/usr/bin/env python
"""
NeuroSync-Core Installation Script
This script automatically detects your CUDA capabilities and installs the 
appropriate version of PyTorch and other dependencies.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {platform.python_version()}")

def check_cuda():
    """Check for CUDA availability and version"""
    try:
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        for line in output.split('\n'):
            if 'CUDA Version' in line:
                cuda_version = line.split('CUDA Version:')[1].strip()
                print(f"✅ CUDA detected: {cuda_version}")
                return cuda_version
        print("❌ NVIDIA GPU detected but couldn't determine CUDA version")
        return None
    except:
        print("❌ No NVIDIA GPU detected or drivers not installed")
        return None

def install_dependencies(cuda_version):
    """Install dependencies based on CUDA availability"""
    print("\n📦 Installing dependencies...")
    
    # Install base requirements
    requirements = [
        "numpy",
        "scipy",
        "flask",
        "pydub",
        "sounddevice",
        "transformers",
        "librosa",
    ]
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + requirements)
    print("✅ Installed base dependencies")
    
    # Install PyTorch based on CUDA version
    if cuda_version is None:
        print("⚠️ Installing CPU-only PyTorch version (slower)")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision'])
        return
    
    # Convert CUDA version string to float for comparison
    try:
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        cuda_float = float(cuda_major_minor)
    except:
        print(f"⚠️ Could not parse CUDA version: {cuda_version}, installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                              'torch', 'torchvision'])
        return
    
    # Install PyTorch based on CUDA version
    if cuda_float >= 12.8:
        print("🚀 Installing PyTorch with CUDA 12.8 support (nightly build)")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torch', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--pre', 'torchvision', 
                                  '--index-url', 'https://download.pytorch.org/whl/nightly/cu128'])
        except Exception as e:
            print(f"⚠️ Failed to install nightly build: {e}")
            print("⚠️ Falling back to stable CUDA 12.1")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                                  '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 12.1:
        print("🚀 Installing PyTorch with CUDA 12.1 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu121'])
    elif cuda_float >= 11.8:
        print("🚀 Installing PyTorch with CUDA 11.8 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu118'])
    elif cuda_float >= 11.7:
        print("🚀 Installing PyTorch with CUDA 11.7 support")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision',
                              '--index-url', 'https://download.pytorch.org/whl/cu117'])
    else:
        print(f"⚠️ CUDA {cuda_version} is older than recommended. Installing CPU version")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'])

def verify_installation():
    """Verify PyTorch is installed correctly with CUDA if available"""
    print("\n🔍 Verifying installation...")
    try:
        verification_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    # Test CUDA tensor operation
    x = torch.rand(5, 3, device='cuda')
    y = torch.rand(3, 5, device='cuda')
    z = x @ y
    print("✅ CUDA tensor operation successful!")
else:
    print("⚠️ CUDA not available, using CPU only")
"""
        subprocess.run([sys.executable, '-c', verification_script], check=True)
    except Exception as e:
        print(f"❌ Installation verification failed: {e}")
        return False
    
    return True

def main():
    print("=" * 70)
    print("🧠 NeuroSync-Core Installation Script")
    print("=" * 70)
    
    check_python_version()
    cuda_version = check_cuda()
    install_dependencies(cuda_version)
    
    success = verify_installation()
    
    if success:
        print("\n✅ Installation completed successfully!")
        print("\n📝 Next steps:")
        print("1. Run 'python neurosync_local_api.py' to start the server")
        print("2. Check out the documentation for more information")
    else:
        print("\n❌ Installation may not be complete. Please check the errors above.")
        print("If you're having issues with CUDA, try installing manually:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main() 