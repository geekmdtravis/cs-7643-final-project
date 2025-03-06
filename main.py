import torch
import platform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpu_info = 'Not Detected'
if torch.cuda.is_available():
    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
    gpu_memory = f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

system_info = {
    'OS': platform.system() + ' ' + platform.release(),
    'CPU': platform.processor(),
    'PyTorch Version': torch.__version__,
    'Device': str(device),
    'GPU Info': gpu_info
}

# Print system information
print("\n=== System Information ===")
for key, value in system_info.items():
    print(f"{key}: {value}")
