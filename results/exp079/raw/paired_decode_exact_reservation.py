import datetime as dt, json, os, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[3]
binary=root/'results/exp079/build/bin/llama-bench'
model=Path('/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
outdir=root/'results/exp079/raw'; out=[]
def sample():
 p=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return [float(x.strip()) for x in p.stdout.strip().split(',')]
def waitgate():
 t0=time.monotonic()
 while True:
  x=sample()
  if x[0]<=60 and x[1]<=5:return {'temp_c':x[0],'util_pct':x[1],'mem_mib':x[2],'wait_s':time.monotonic()-t0}
  print('cooldown',x,file=sys.stderr,flush=True);time.sleep(5)
for ctx in (512,4096):
 for pair in (1,2):
  order=('baseline','candidate') if pair%2 else ('candidate','baseline')
  for arm in order:
   gate=waitgate(); env=dict(os.environ); env['LD_LIBRARY_PATH']=str(binary.parent)+':'+env.get('LD_LIBRARY_PATH','')
   if arm=='candidate':env['GGML_CUDA_EXP079_PERSIST_FIRST_KV']='1'
   else:env.pop('GGML_CUDA_EXP079_PERSIST_FIRST_KV',None)
   cmd=[str(binary),'-m',str(model),'-ngl','99','-fa','on','-b','2048','-ub','512','-ctk','f16','-ctv','f16','-t','8','-r','7','-o','json','-p','0','-n','128','-d',str(ctx)]
   print(f'ctx={ctx} pair={pair} arm={arm} gate={gate}',file=sys.stderr,flush=True)
   started=time.monotonic(); p=subprocess.Popen(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True); tele=[]
   while p.poll() is None:
    try: tele.append({'elapsed_s':time.monotonic()-started,'gpu':sample()})
    except Exception:pass
    time.sleep(.25)
   stdout,stderr=p.communicate();
   stem=f'ctx{ctx}_pair{pair}_{arm}_exact'
   (outdir/f'{stem}.stdout.json').write_text(stdout);(outdir/f'{stem}.stderr.log').write_text(stderr)
   if p.returncode: raise RuntimeError(f'{stem} failed rc={p.returncode}: {stderr[-4000:]}')
   if (arm=='candidate') != ('Exp079 persisting policy attached' in stderr): raise RuntimeError(f'{stem} policy attach marker mismatch')
   rows=json.loads(stdout)
   out.append({'context':ctx,'pair':pair,'arm':arm,'order':order,'gate':gate,'command':cmd,'exit_code':p.returncode,'elapsed_s':time.monotonic()-started,'gpu_samples':tele,'result':rows[0],'stdout':stem+'.stdout.json','stderr':stem+'.stderr.log'})
   print(stem,rows[0].get('avg_ts'),file=sys.stderr,flush=True)
path=root/'results/exp079/ptq1-decode-exact-reservation-paired.json';path.write_text(json.dumps({'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'pairs':2,'contexts':[512,4096],'repetitions':7,'decode_tokens':128,'runs':out},indent=2)+'\n');print(path)
