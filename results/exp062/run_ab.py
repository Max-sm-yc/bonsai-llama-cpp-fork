import json, pathlib, subprocess, time, statistics
root=pathlib.Path('/home/maxsun/autonomous_projects/.worktrees/exp062-repeated-smallop-fusion')
model='/home/maxsun/autonomous_projects/bonsai2-rtx3080/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'
bins={
 'base':'/home/maxsun/autonomous_projects/.worktrees/exp060-concat-cache-fusion/build/bin/llama-bench',
 'candidate':str(root/'build/bin/llama-bench'),
}
out=root/'results/exp062/raw'; out.mkdir(parents=True,exist_ok=True)
plan=[('pair1','base'),('pair1','candidate'),('pair2','candidate'),('pair2','base')]
for ctx in [512,4096]:
 for pair,arm in plan:
  for attempt in range(60):
   q=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
   temp,util,mem=map(int,[v.strip() for v in q.split(',')])
   gate={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'temperature_c':temp,'utilization_pct':util,'memory_mib':mem,'gate_attempt':attempt+1}
   if temp<=60 and util<=5: break
   time.sleep(5)
  else: raise SystemExit(f'gate timed out for {pair}/{arm}/ctx{ctx}: {gate}')
  (out/f'{pair}_{arm}_ctx{ctx}_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
  cmd=[bins[arm],'-m',model,'-ngl','99','-fa','on','-b','2048','-ub','512','-ctk','f16','-ctv','f16','-t','8','-p','0','-n','128','-d',str(ctx),'-r','7','-o','json']
  print(f'START {pair} {arm} ctx{ctx} gate={gate}',flush=True)
  p=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
  (out/f'{pair}_{arm}_ctx{ctx}.stdout.json').write_text(p.stdout)
  (out/f'{pair}_{arm}_ctx{ctx}.stderr.log').write_text(p.stderr)
  if p.returncode:
   raise SystemExit(f'failed {pair}/{arm}/ctx{ctx}, rc={p.returncode}')
  try: data=json.loads(p.stdout)
  except Exception as e: raise SystemExit(f'bad JSON {pair}/{arm}/ctx{ctx}: {e}; stdout={p.stdout[-1000:]}')
  (out/f'{pair}_{arm}_ctx{ctx}.json').write_text(json.dumps({'gate':gate,'command':cmd,'rows':data},indent=2)+'\n')
  print(f'DONE {pair} {arm} ctx{ctx}',flush=True)
