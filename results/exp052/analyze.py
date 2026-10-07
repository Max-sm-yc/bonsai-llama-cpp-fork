#!/usr/bin/env python3
"""Export one-token CUDA graph replay and kernel-signature profile for Exp052."""
import csv, json, sqlite3, statistics, sys
from pathlib import Path

def family(name):
    if 'mul_mat_vec_ptq1_0_pt' in name: return 'GEMV'
    if 'mul_mat_vec_q<(ggml_type)142' in name: return 'GEMV'
    if 'fwht_quantize_q8_1' in name or 'fwht_rms_quantize_q8_1' in name or 'fwht_cuda_block' in name or 'quantize_q8_1' in name: return 'Activation prep'
    if 'rms_norm_f32' in name: return 'RMSNorm'
    if any(x in name for x in ('gated_delta_net_cuda','ssm_conv_f32','l2_norm_f32')): return 'GDN'
    if 'flash_attn' in name: return 'Attention'
    if 'mul_mat_q<' in name: return 'Quantized GEMM'
    return 'Other'

for input_path in map(Path,sys.argv[1:]):
    con=sqlite3.connect(input_path)
    strings=dict(con.execute('select id,value from StringIds'))
    launches={corr for (corr,) in con.execute("select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%'")}
    records={}
    for corr,start,end,name_id,node in con.execute('select correlationId,start,end,demangledName,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null'):
        if corr not in launches: continue
        r=records.setdefault(corr,{'start':start,'end':end,'nodes':set(),'families':{},'signatures':{}})
        r['start']=min(r['start'],start); r['end']=max(r['end'],end); r['nodes'].add(node)
        name=strings.get(name_id,'')
        fam=family(name); dur=end-start
        fr=r['families'].setdefault(fam,{'count':0,'ns':0}); fr['count']+=1; fr['ns']+=dur
        sr=r['signatures'].setdefault(name,{'count':0,'ns':0}); sr['count']+=1; sr['ns']+=dur
    con.close()
    rows=sorted(records.values(),key=lambda r:r['start'])
    assert len(rows)==len(launches)
    node_counts={len(r['nodes']) for r in rows}; kernels={sum(x['count'] for x in r['families'].values()) for r in rows}
    total_ns=[sum(v['ns'] for v in r['families'].values()) for r in rows]
    summary={'file':input_path.name,'graph_replays':len(rows),'nodes_per_replay':sorted(node_counts),'kernel_instances_per_replay':sorted(kernels),'gpu_span_ms':{'mean':statistics.mean((r['end']-r['start'])/1e6 for r in rows),'median':statistics.median((r['end']-r['start'])/1e6 for r in rows),'min':min((r['end']-r['start'])/1e6 for r in rows),'max':max((r['end']-r['start'])/1e6 for r in rows)},'summed_kernel_ms_per_replay':statistics.mean(total_ns)/1e6,'families':{},'signatures':{}}
    cats=sorted({f for r in rows for f in r['families']})
    for cat in cats:
        vals=[r['families'].get(cat,{}).get('ns',0)/1e6 for r in rows]
        counts={r['families'].get(cat,{}).get('count',0) for r in rows}
        summary['families'][cat]={'count_per_replay':sorted(counts),'mean_ms_per_replay':statistics.mean(vals),'stdev_ms_per_replay':statistics.stdev(vals),'min_ms_per_replay':min(vals),'max_ms_per_replay':max(vals),'share_pct':100*sum(r['families'].get(cat,{}).get('ns',0) for r in rows)/sum(total_ns)}
    sigs=sorted({s for r in rows for s in r['signatures']})
    for sig in sigs:
        vals=[r['signatures'].get(sig,{}).get('ns',0)/1e6 for r in rows]
        counts={r['signatures'].get(sig,{}).get('count',0) for r in rows}
        summary['signatures'][sig]={'count_per_replay':sorted(counts),'mean_ms_per_replay':statistics.mean(vals),'stdev_ms_per_replay':statistics.stdev(vals),'min_ms_per_replay':min(vals),'max_ms_per_replay':max(vals),'share_pct':100*sum(r['signatures'].get(sig,{}).get('ns',0) for r in rows)/sum(total_ns)}
    out=input_path.with_suffix('.profile.json')
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(out)
