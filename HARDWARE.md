# Training Hardware

Specs of the machine used to train the language model.

## Summary

| Component | Value |
|-----------|-------|
| GPU | AMD Radeon RX 7800 XT (RDNA3, gfx1101) — **AMD, not NVIDIA: no CUDA** |
| VRAM | 16 GB GDDR6 (WMI misreports as 4 GB; it caps at 32-bit) |
| iGPU | AMD Radeon Graphics (Ryzen 7700X integrated, not used for training) |
| CPU | AMD Ryzen 7 7700X, 8 cores / 16 threads, 4.5 GHz |
| RAM | 32 GB DDR5-4800 (2 × 16 GB) |
| OS | Windows 11 Home, 64-bit (10.0.26300) |
| GPU backend | TBD — ROCm (PyTorch for Windows, or WSL2) instead of CUDA |
| Python | 3.14.3 installed (may need 3.12 for ROCm PyTorch wheels) |
| PyTorch version | not installed yet |

---

## Raw command output

### GPU
```
> Get-CimInstance Win32_VideoController
Name          : AMD  Radeon RX 7800 XT
DriverVersion : 32.0.32015.7029

Name          : AMD Radeon(TM) Graphics
DriverVersion : 32.0.21045.16010

> nvidia-smi
not found (no NVIDIA GPU)
```

### CPU
```
Name                      : AMD Ryzen 7 7700X 8-Core Processor
NumberOfCores             : 8
NumberOfLogicalProcessors : 16
MaxClockSpeed             : 4501
```

### RAM
```
   Capacity Speed
17179869184  4800
17179869184  4800
Total GB: 31.1
```

### OS / system
```
Caption        : Microsoft Windows 11 Home
Version        : 10.0.26300
OSArchitecture : 64-bit
```

### GPU + PyTorch (run after installing PyTorch)
```
```
