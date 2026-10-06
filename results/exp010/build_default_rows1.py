#!/usr/bin/env python3
"""Build the source-default ROWS=1 library with the cached sm_86 compile recipe."""
import subprocess

commands = subprocess.run(["ninja", "-t", "commands"], cwd="build", text=True, capture_output=True, check=True).stdout.splitlines()
compile_cmd = next(line for line in commands if "nvcc" in line and "-MT ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o" in line)
link_cmd = next(line for line in commands if "-o bin/libggml-cuda.so.0.21.0" in line and "/usr/bin/g++" in line)
if "PTQ1_0_PT_ROWS_1=" in compile_cmd:
    raise SystemExit("expected CMAKE_CUDA_FLAGS to be empty")
print("Compiling mmvq.cu with source-default PTQ1_0_PT_ROWS_1=1", flush=True)
subprocess.run(compile_cmd, shell=True, cwd="build", check=True)
print("Relinking libggml-cuda.so", flush=True)
subprocess.run(link_cmd, shell=True, cwd="build", check=True)
