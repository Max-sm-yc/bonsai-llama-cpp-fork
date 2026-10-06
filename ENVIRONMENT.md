# Environment

Captured 2026-10-06 during baseline setup.

## Operating system and host

- Fedora Linux 43 Workstation, x86_64
- Kernel: Linux 7.1.8-100.fc43.x86_64
- CPU: Intel Core i7-10700K, 8 cores / 16 threads
- RAM: 31 GiB; swap: 8 GiB
- Initial free disk on the project volume: 541 GiB (929 GiB volume)

## GPU and CUDA

- GPU: NVIDIA GeForce RTX 3080, GA102, PCI 01:00.0
- Compute capability: 8.6 (sm_86)
- VRAM: 10,240 MiB
- Driver: NVIDIA 580.178.04; `nvidia-smi` reports CUDA compatibility 13.0
- Toolkit: CUDA 13.2.86 (`nvcc`)
- At idle capture: 173 MiB used, 0% utilization, P8; GNOME and Xwayland use the display GPU.
- The current command environment exposes `/dev/nvidia*`; `nvidia-smi`, CUDA model loading, benchmarking, and Nsight Systems tracing were verified on the RTX 3080. Keep desktop VRAM use in mind when calculating model memory growth.

## Build and profiling tools

- GCC 15.3.1
- CMake 3.31.11
- Ninja 1.13.1
- Python 3.14.7
- Nsight Systems 2025.6.3.541-256337736014v0
- Nsight Compute 2026.1.1.0
- `cuda-gdb` and `compute-sanitizer` are installed
- Git 2.55.0 and Git LFS 3.7.1
- `huggingface-cli` is available from `huggingface_hub` 0.30.2; `hf_xet` is not installed at initial capture

## Notes

CUDA compilation can target sm_86 without probing a GPU by setting `CMAKE_CUDA_ARCHITECTURES=86`. Runtime benchmarks need the RTX 3080 device nodes and should record the concurrent desktop VRAM use.
