#!/usr/bin/env python3
"""
Installation script for NeuroSync-Core

This script:
1. Checks for Python version
2. Detects CUDA capabilities
3. Installs the appropriate PyTorch version
4. Installs other dependencies from requirements.txt
5. Verifies the installation
"""

import os
import sys
import subprocess
import platform

def print_step(message):
    """Print a step message with formatting"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80)

def run_command(command):
    """Run a command and return the output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print_step("Checking Python version")
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    major, minor, _ = map(int, python_version.split('.'))
    if major < 3 or (major == 3 and minor < 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    
    print("Python version is compatible.")
    return True

def detect_cuda():
    """Detect CUDA capabilities"""
    print_step("Detecting CUDA capabilities")
    
    # Check if nvidia-smi is available
    has_nvidia_smi = run_command("nvidia-smi" if platform.system() != "Windows" else "where nvidia-smi")
    
    if has_nvidia_smi:
        print("NVIDIA GPU detected.")
        # Try to extract CUDA version
        try:
            result = subprocess.run("nvidia-smi", shell=True, check=True, text=True, capture_output=True)
            output = result.stdout
            
            # Extract CUDA version from nvidia-smi output
            import re
            cuda_version_match = re.search(r"CUDA Version: (\d+\.\d+)", output)
            
            if cuda_version_match:
                cuda_version = cuda_version_match.group(1)
                print(f"CUDA version: {cuda_version}")
                major, minor = map(int, cuda_version.split('.'))
                
                if major >= 12:
                    return "cu121"  # Use CUDA 12.1
                elif major >= 11 and minor >= 8:
                    return "cu118"  # Use CUDA 11.8
                else:
                    print("Warning: CUDA version is older than 11.8, defaulting to CPU version.")
                    return "cpu"
            else:
                print("Could not determine CUDA version, defaulting to CUDA 11.8")
                return "cu118"
        except:
            print("Error determining CUDA version, defaulting to CUDA 11.8")
            return "cu118"
    else:
        print("No NVIDIA GPU detected or nvidia-smi not available. Using CPU version.")
        return "cpu"

def install_pytorch(cuda_version):
    """Install PyTorch with appropriate CUDA support"""
    print_step(f"Installing PyTorch with {cuda_version} support")
    
    if cuda_version == "cu128":
        cmd = "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
    elif cuda_version == "cu121":
        cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_version == "cu118":
        cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:  # CPU
        cmd = "pip install torch torchvision"
    
    success = run_command(cmd)
    
    if not success:
        print("Failed to install PyTorch. Please install it manually.")
        return False
    
    # Verify PyTorch installation
    print("Verifying PyTorch installation...")
    verify_cmd = 'python -c "import torch; print(f\'PyTorch version: {torch.__version__}\'); print(f\'CUDA available: {torch.cuda.is_available()}\')"'
    run_command(verify_cmd)
    
    return True

def install_dependencies():
    """Install other dependencies from requirements.txt"""
    print_step("Installing dependencies from requirements.txt")
    
    # Exclude torch, torchvision from requirements.txt since we installed it separately
    cmd = "pip install --upgrade pip"
    run_command(cmd)
    
    cmd = "pip install -r requirements.txt"
    success = run_command(cmd)
    
    if not success:
        print("Failed to install all dependencies. Some packages may be missing.")
        return False
    
    return True

def verify_installation():
    """Verify the key components are installed correctly"""
    print_step("Verifying installation")
    
    # Check if essential packages can be imported
    checks = [
        ("openai", "OpenAI API"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("flask", "Flask"),
        ("transformers", "Transformers"),
        ("dotenv", "python-dotenv"),
        ("sounddevice", "SoundDevice"),
    ]
    
    all_passed = True
    for package, name in checks:
        try:
            __import__(package)
            print(f"✅ {name} installed successfully")
        except ImportError:
            print(f"❌ {name} not installed correctly")
            all_passed = False
    
    if all_passed:
        print("\n✅ Installation verified successfully!")
    else:
        print("\n⚠️ Some components could not be verified. Please check the error messages.")
    
    return all_passed

def main():
    """Main installation process"""
    print("""
    ===============================================================================
                            NeuroSync-Core Installation
    ===============================================================================
    This script will install all required dependencies for NeuroSync-Core.
    """)
    
    # Check Python version
    check_python_version()
    
    # Detect CUDA capabilities
    cuda_version = detect_cuda()
    
    # Install PyTorch
    install_pytorch(cuda_version)
    
    # Install other dependencies
    install_dependencies()
    
    # Verify installation
    verify_installation()
    
    print("""
    ===============================================================================
                            Installation Complete
    ===============================================================================
    
    You can now run NeuroSync-Core with:
    
        python neurosync_client.py --llm openai --tts elevenlabs
    
    Make sure to set up your .env file with API keys first!
    """)

if __name__ == "__main__":
    main() 