#!/usr/bin/env python3
import json, os, pathlib, subprocess, time
work=pathlib.Path(__file__).resolve().parents[2]
model=pathlib.Path('/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
bin=work/'build-fast/bin/llama-bench'
out=work/'results/exp064/raw'; out.mkdir(parents=True,exist_ok=True)
baseenv=os.environ.copy(); baseenv['LD_LIBRARY_PATH']=str(work/'build-fast/bin')
# Load paths are recorded and verified before every arm.
resolved=subprocess.run(['ldd',str(bin)],env=baseenv,text=True,capture_output=True,check=True).stdout
(out/'ldd_candidate.txt').write_text(resolved)
assert str(work/'build-fast/bin/libggml-cuda.so.0') in resolved

def gate():
    while True:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
        t,u,m=[int(x.strip()) for x in line.split(',')]
        if t<=60 and u<=5:
            return {'temperature_c':t,'utilization_pct':u,'memory_mib':m,'utc_epoch':time.time()}
        time.sleep(5)

for ctx,order in [(512,['generic','candidate']), (512,['candidate','generic']), (4096,['generic','candidate']), (4096,['candidate','generic'])]:
    pair=1 if order[0]=='generic' else 2
    for arm in order:
        pre=gate()
        env=baseenv.copy()
        if arm=='generic': env['GGML_CUDA_DISABLE_SSM_L2_ALIAS_FUSION']='1'
        else: env.pop('GGML_CUDA_DISABLE_SSM_L2_ALIAS_FUSION',None)
        command=[str(bin),'-m',str(model),'-ngl','99','-fa','on','-b','2048','-ub','512','-ctk','f16','-ctv','f16','-t','8','-p','0','-n','128','-d',str(ctx),'-r','7','-o','json']
        label=f'pair{pair}_{arm}_ctx{ctx}'
        samples=[]
        proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        while proc.poll() is None:
            try:
                line=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
                t,u,m=[int(x.strip()) for x in line.split(',')]
                samples.append({'temperature_c':t,'utilization_pct':u,'memory_mib':m,'utc_epoch':time.time()})
            except Exception: pass
            time.sleep(.5)
        stdout,stderr=proc.communicate()
        (out/f'{label}.stdout.json').write_text(stdout)
        (out/f'{label}.stderr.log').write_text(stderr)
        meta={'label':label,'arm':arm,'context':ctx,'pair':pair,'command':command,'pre_run_gate':pre,
              'peak_vram_mib':max([s['memory_mib'] for s in samples],default=pre['memory_mib']),
              'gpu_samples':samples,'exit_code':proc.returncode,'library_path':str(work/'build-fast/bin/libggml-cuda.so.0.21.0')}
        (out/f'{label}.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        if proc.returncode: raise SystemExit(f'{label} failed; see stderr log')
        try:
            result=json.loads(stdout)
        except Exception as e: raise SystemExit(f'{label} output was not JSON: {e}')
        (out/f'{label}.json').write_text(json.dumps({'meta':meta,'rows':result},indent=2)+'\n')
        print(label, pre, flush=True)
