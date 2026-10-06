#!/usr/bin/env python3
"""Recompile only mmvq.cu with a PTQ1 planar ROWS override, then relink CUDA."""
import subprocess
import sys

rows = int(sys.argv[1])
if rows not in (1, 2, 4, 8):
    raise SystemExit("rows must be one of 1, 2, 4, 8")
commands = subprocess.run(["ninja", "-t", "commands"], cwd="build", text=True, capture_output=True, check=True).stdout.splitlines()
compile_cmd = next((line for line in commands if "nvcc" in line and "-MT ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o" in line), None)
link_cmd = next((line for line in commands if "-o bin/libggml-cuda.so.0.21.0" in line and "/usr/bin/g++" in line), None)
if not compile_cmd or not link_cmd:
    raise SystemExit("could not find mmvq compile and ggml-cuda link commands")
old = "-DPTQ1_0_PT_ROWS_1=1"
if old not in compile_cmd:
    raise SystemExit("configured compile command does not contain expected ROWS=1 macro")
compile_cmd = compile_cmd.replace(old, f"-DPTQ1_0_PT_ROWS_1={rows}", 1)
print(f"Compiling mmvq.cu with PTQ1_0_PT_ROWS_1={rows}", flush=True)
subprocess.run(compile_cmd, shell=True, cwd="build", check=True)
print("Relinking libggml-cuda.so", flush=True)
subprocess.run(link_cmd, shell=True, cwd="build", check=True)
