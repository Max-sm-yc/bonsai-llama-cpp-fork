#!/usr/bin/env python3
import pathlib, shlex, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[2]; build=root/'build'; n=int(sys.argv[1]); label=f'items{n}'
out=root/'results/exp025/libs'/label; out.mkdir(parents=True,exist_ok=True)
cmds=subprocess.run(['ninja','-t','commands'],cwd=build,text=True,capture_output=True,check=True).stdout.splitlines()
cc=next(x for x in cmds if 'nvcc' in x and '-MT ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o' in x)
link=next(x for x in cmds if '/usr/bin/g++' in x and 'libggml-cuda.so.0.21.0' in x)
a=shlex.split(cc); a.insert(1,f'-DPTQ1_0_PT_ITEMS_PER_THREAD={n}')
with open(root/f'results/exp025/{label}_build.log','w') as f:
 p=subprocess.run(a,cwd=build,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode: raise SystemExit(p.returncode)
with open(root/f'results/exp025/{label}_build.log','a') as f:
 p=subprocess.run(link,shell=True,cwd=build,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode: raise SystemExit(p.returncode)
subprocess.run(['cp','-a',str(build/'bin/libggml-cuda.so.0.21.0'),str(out/'libggml-cuda.so.0.21.0')],check=True)
subprocess.run(['ln','-sf','libggml-cuda.so.0.21.0',str(out/'libggml-cuda.so.0')],check=True)
subprocess.run(['ln','-sf','libggml-cuda.so.0',str(out/'libggml-cuda.so')],check=True)
print(out)
