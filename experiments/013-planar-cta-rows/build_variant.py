#!/usr/bin/env python3
"""Compile one rows/CTA cap variant from the configured CUDA build."""
import pathlib
import shlex
import subprocess
import sys

cap = int(sys.argv[1])
target = int(sys.argv[2])
label = sys.argv[3]
root = pathlib.Path(__file__).resolve().parents[2]
build = root / 'build'
out = root / 'results/exp013/builds' / label
commands = subprocess.run(['ninja', '-t', 'commands'], cwd=build, text=True,
                          capture_output=True, check=True).stdout.splitlines()
compile_cmd = next((line for line in commands if 'nvcc' in line and
                    '-MT ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o' in line), None)
link_cmd = next((line for line in commands if '-o bin/libggml-cuda.so.0.21.0' in line and
                 '/usr/bin/g++' in line), None)
if not compile_cmd or not link_cmd:
    raise SystemExit('could not find mmvq compile and ggml-cuda link commands')
args = shlex.split(compile_cmd)
args.insert(1, f'-DPTQ1_0_PT_MAX_ROWS={cap}')
args.insert(2, f'-DPTQ1_0_PT_SMEM_FLOATS={target}')
print(f'Compiling cap={cap}, target={target} with configured mmvq command', flush=True)
subprocess.run(args, cwd=build, check=True)
subprocess.run(link_cmd, shell=True, cwd=build, check=True)
out.mkdir(parents=True, exist_ok=True)
subprocess.run(['cp', '-a', str(build / 'bin/.') , str(out)], check=True)
print(f'Archived variant at {out}', flush=True)
