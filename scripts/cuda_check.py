import torch
import subprocess
import sys

print("===== CUDA Availability Check =====")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print("CUDA working properly!")
else:
    print("\n===== Diagnostic Information =====")
    print("CUDA is not available. Possible reasons:")
    print(" 1. PyTorch installed without CUDA support")
    print(" 2. NVIDIA drivers are outdated or not properly installed")
    print(" 3. CUDA toolkit not installed correctly")
    
    print("\n===== Installation Help =====")
    print("To install PyTorch with CUDA, run the following command:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("(Change cu121 to match your CUDA version if needed)")
    
    print("\n===== NVIDIA Driver Info =====")
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            print(result.stdout)
        else:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            print(result.stdout)
    except:
        print("Couldn't execute nvidia-smi. NVIDIA drivers might not be installed correctly.") 