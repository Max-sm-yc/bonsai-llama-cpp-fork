#!/usr/bin/env python3
"""Summarize one-token CUDA graph replays from an Nsight Systems SQLite export."""
import argparse
import csv
import json
import re
import sqlite3
import statistics
from pathlib import Path


def family(name):
    if "mul_mat_vec_ptq1_0_pt" in name:
        return "PTQ1_0 GEMV"
    if "fwht_quantize_q8_1" in name or "fwht_rms_quantize_q8_1" in name:
        return "QKV activation prep"
    if "rms_norm_f32" in name:
        return "RMSNorm"
    if any(x in name for x in ("gated_delta_net_cuda", "ssm_conv_f32", "l2_norm_f32")):
        return "GDN"
    if "flash_attn" in name:
        return "Attention"
    if "mul_mat_q<" in name:
        return "Quantized GEMM"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", type=Path)
    ap.add_argument("--prefix", required=True, help="output prefix, e.g. results/profile/exp047_ctx512")
    ap.add_argument("--context", type=int, required=True)
    args = ap.parse_args()
    con = sqlite3.connect(args.sqlite)
    names = dict(con.execute("select id,value from StringIds"))
    replays = {}
    query = """select correlationId, start, end, demangledName, graphNodeId
               from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null
               order by start"""
    for corr, start, end, name_id, node_id in con.execute(query):
        if corr is None:
            continue
        rec = replays.setdefault(corr, {"start": start, "end": end, "nodes": set(), "families": {}, "kernels": 0})
        rec["start"] = min(rec["start"], start)
        rec["end"] = max(rec["end"], end)
        rec["nodes"].add(node_id)
        rec["kernels"] += 1
        fam = family(names.get(name_id, ""))
        rec["families"][fam] = rec["families"].get(fam, 0) + (end - start)
    graph_calls = con.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
        join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%'""").fetchone()[0]
    con.close()
    rows = sorted(replays.items(), key=lambda x: x[1]["start"])
    assert len(rows) == graph_calls, (len(rows), graph_calls)
    assert all(r["kernels"] == 1432 and len(r["nodes"]) == 1432 for _, r in rows)

    categories = ["PTQ1_0 GEMV", "Quantized GEMM", "QKV activation prep", "GDN", "RMSNorm", "Attention", "Other"]
    csv_path = args.prefix + ".replay.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["replay", "graph_launch_correlation_id", "gpu_span_ms", "kernel_instances", "unique_nodes"] + [x + "_ms" for x in categories])
        for index, (corr, rec) in enumerate(rows, 1):
            w.writerow([index, corr, (rec["end"]-rec["start"])/1e6, rec["kernels"], len(rec["nodes"])] + [rec["families"].get(x, 0)/1e6 for x in categories])
    summaries = {}
    for category in categories:
        vals = [r["families"].get(category, 0)/1e6 for _, r in rows]
        summaries[category] = {
            "mean_ms_per_token": statistics.mean(vals),
            "median_ms_per_token": statistics.median(vals),
            "stdev_ms_per_token": statistics.stdev(vals),
            "min_ms_per_token": min(vals),
            "max_ms_per_token": max(vals),
            "share_of_kernel_time_pct": 100 * sum(vals) / sum(sum(r["families"].values()) for _, r in rows) * 1e6,
        }
    total_ms = [(r["end"]-r["start"])/1e6 for _, r in rows]
    result = {
        "context": args.context,
        "graph_replays": len(rows),
        "graph_launch_api_calls": graph_calls,
        "kernel_instances_per_replay": sorted({r["kernels"] for _, r in rows}),
        "unique_graph_nodes_per_replay": sorted({len(r["nodes"]) for _, r in rows}),
        "replay_gpu_span_ms": {"mean": statistics.mean(total_ms), "median": statistics.median(total_ms), "stdev": statistics.stdev(total_ms), "min": min(total_ms), "max": max(total_ms)},
        "families": summaries,
        "definitions": {
            "PTQ1_0 GEMV": "All mul_mat_vec_ptq1_0_pt specializations.",
            "Quantized GEMM": "mul_mat_q kernels.",
            "QKV activation prep": "fwht_quantize_q8_1 and fwht_rms_quantize_q8_1 kernels.",
            "GDN": "gated_delta_net_cuda plus ssm_conv_f32 and l2_norm_f32 kernels.",
            "RMSNorm": "rms_norm_f32 kernels; RMS fused inside fwht_rms_quantize_q8_1 is counted as activation prep.",
            "Attention": "flash_attn kernels including stream-fixup names.",
            "Other": "All remaining kernels, including non-PTQ1 GEMV and graph pointwise/copy operations.",
        },
    }
    with open(args.prefix + ".replay.json", "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
