#!/usr/bin/env python3
"""Paired PTQ1_0 batch-1 GEMV prefetch benchmark."""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import benchmark.run as harness
MODEL = ROOT / 'models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'

def wait_cool(limit):
    started = time.monotonic()
    while True:
        sample = harness.gpu_sample()
        if sample is None: raise RuntimeError('GPU telemetry unavailable')
        if sample['temperature_c'] is not None and sample['utilization_percent'] is not None and sample['temperature_c'] <= limit and sample['utilization_percent'] <= 5:
            return {'sample': sample, 'wait_seconds': time.monotonic()-started}
        print(f"cooldown: {sample}", file=sys.stderr, flush=True)
        time.sleep(5)

def command(binary, mode, context, args):
    c = [str(binary), '-m', str(MODEL), '-ngl', '99', '-fa', 'on', '-b', '2048', '-ub', '512', '-ctk', 'f16', '-ctv', 'f16', '-t', '8', '-r', str(args.repetitions), '-o', 'json']
    if mode == 'decode': c += ['-p', '0', '-n', str(args.decode_tokens), '-d', str(context)]
    else: c += ['-p', '0', '-n', '0', '-pg', f'{context},{args.decode_tokens}']
    return c

def run_one(variant, binary, mode, ctx, pair, args):
    gate = wait_cool(args.cooldown_temp_c)
    cmd = command(binary, mode, ctx, args)
    before = harness.gpu_sample(); samples=[]
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(binary.parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    started = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while proc.poll() is None:
        sample = harness.gpu_sample()
        if sample is not None:
            samples.append({'elapsed_seconds': time.monotonic() - started, **sample})
        time.sleep(.25)
    out, err = proc.communicate(); after = harness.gpu_sample()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw=ROOT/'results/raw'; raw.mkdir(parents=True,exist_ok=True)
    stem=f'{stamp}_exp002_{variant}_{mode}_ctx{ctx}_pair{pair}'
    (raw/f'{stem}.stdout.json').write_text(out); (raw/f'{stem}.stderr.log').write_text(err)
    if proc.returncode: raise RuntimeError(f'failed: {cmd}; see {stem}')
    rows=json.loads(out)
    return {'variant':variant,'mode':mode,'context':ctx,'pair':pair,'gate':gate,'command':cmd,'exit_code':proc.returncode,'before':before,'after':after,'telemetry':samples,'results':[harness.summarize_row(r) for r in rows],'stdout_raw':f'results/raw/{stem}.stdout.json','stderr_raw':f'results/raw/{stem}.stderr.log'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--on-binary',default='results/exp002/on/bin/llama-bench'); p.add_argument('--off-binary',default='results/exp002/off/bin/llama-bench')
    p.add_argument('--modes',nargs='+',choices=('decode','combined'),default=['decode','combined'])
    p.add_argument('--contexts',nargs='+',type=int,default=[512,4096])
    p.add_argument('--pair-start',type=int,default=1)
    p.add_argument('--pairs',type=int,default=5); p.add_argument('--repetitions',type=int,default=3); p.add_argument('--decode-tokens',type=int,default=128); p.add_argument('--cooldown-temp-c',type=float,default=60); p.add_argument('--output',default='results/exp002/paired.json')
    a=p.parse_args(); on=Path(a.on_binary); off=Path(a.off_binary)
    for b in (on,off):
        if not b.is_absolute(): b=ROOT/b
        if not b.is_file(): raise RuntimeError(f'missing benchmark: {b}')
    runs=[]
    for mode in a.modes:
      for ctx in a.contexts:
       for pair in range(a.pair_start,a.pair_start+a.pairs):
        order=('on','off') if pair%2 else ('off','on')
        print(f'{mode} ctx={ctx} pair={pair} order={order}',file=sys.stderr,flush=True)
        for variant in order:
         binary=on if variant=='on' else off
         runs.append(run_one(variant,binary,mode,ctx,pair,a))
    out={'schema_version':1,'timestamp_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'hardware':{'gpu':'NVIDIA GeForce RTX 3080','compute_capability':'8.6'},'configuration':{'contexts':a.contexts,'modes':a.modes,'pair_ids':list(range(a.pair_start,a.pair_start+a.pairs)),'repetitions_per_process':a.repetitions,'decode_tokens':a.decode_tokens,'n_gpu_layers':99,'flash_attention':'on','batch_size':2048,'ubatch_size':512,'kv_type':'f16','cpu_threads':8,'cooldown_temperature_c':a.cooldown_temp_c},'runs':runs}
    dest=Path(a.output); dest=dest if dest.is_absolute() else ROOT/dest; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(json.dumps(out,indent=2)+'\n'); print(dest)
if __name__=='__main__': main()
