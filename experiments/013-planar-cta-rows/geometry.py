#!/usr/bin/env python3
"""Enumerate PTQ1_0 planar CTA geometry using the production selection rule."""
import argparse
import csv
import math

THREADS = 128


def choose(bpr, cols, rows, target, cap, largest_tile=True):
    if cols != 1 or not largest_tile:
        # Exact production control logic for all multi-column dispatches.
        target, cap = 4096, 16
    rmax = target // (cols * (bpr + 1))
    rmax = max(rows, min(cap, rmax))
    rmax -= rmax % rows
    best, best_util = rows, 0.0
    for r in range(rows, rmax + 1, rows):
        items = (r // rows) * bpr
        iters = (items + THREADS - 1) // THREADS
        util = items / (iters * THREADS)
        if util > best_util + 1e-9 or (cols == 1 and largest_tile and util >= best_util - 1e-9 and r > best):
            best, best_util = r, util
        if cols != 1 and util > .999:
            break
    return best, best_util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true')
    ap.add_argument('--shapes', default='')
    args = ap.parse_args()
    # Exact PTQ1_0 tensor shapes read from the experiment model's GGUF header;
    # gated paths double the partial storage used by the fused launch.
    shapes = [(label, k, m, gate) for label, k, m, gate in [
        ('attn_qkv_K5120_M10240', 5120, 10240, False),
        ('attn_gate_K5120_M6144', 5120, 6144, True),
        ('ffn_fused_gate_K5120_M17408', 5120, 17408, True),
        ('ssm_out_K6144_M5120', 6144, 5120, False),
        ('ffn_down_K17408_M5120', 17408, 5120, False),
        ('output_K5120_M248320', 5120, 248320, False),
    ]]
    if args.shapes:
        shapes = []
        for entry in args.shapes.split(','):
            name, k, m, gate = entry.split(':')
            shapes.append((name, int(k), int(m), gate.lower() in ('1','true','gate')))
    variants = [('cap4_t4096', 4, 4096), ('cap8_t4096', 8, 4096),
                ('cap12_t4096', 12, 4096), ('control_cap16_t4096', 16, 4096),
                ('cap24_t4096', 24, 4096), ('cap32_t4096', 32, 4096),
                ('cap32_t8192', 32, 8192)]
    rows = []
    for v, cap, target in variants:
        for name, k, m, gate in shapes:
            bpr = k // 128
            rpc, util = choose(bpr, 1, 1, target, cap, largest_tile=not v.startswith('control_'))
            smem = rpc * (bpr + 1) * 4 * (2 if gate else 1)
            ctas = math.ceil(m / rpc)
            rows.append((v, name, k, m, gate, rpc, smem, ctas, util, smem <= 98304))
    if args.csv:
        w = csv.writer(__import__('sys').stdout)
        w.writerow(('variant','shape','K','M','gate','rows_per_cta','dynamic_smem_bytes','ctas','item_utilization','under_96KiB'))
        w.writerows(rows)
        # Assert candidate schedules exactly match the control for ncols 2..8.
        bprs = {k // 128 for _, k, _, _ in shapes}
        for bpr in bprs:
            for cols in range(2, 9):
                assert all(choose(bpr, cols, 2, target, cap)[0] ==
                           choose(bpr, cols, 2, 4096, 16)[0]
                           for _, cap, target in variants)
    else:
        print('variant shape K M gate rows/CTA smem/CTA CTAs item-util <=96KiB')
        for row in rows:
            print('%s %s %d %d %s %d %d %d %.3f %s' % (*row[:5], row[5], row[6], row[7], row[8], row[9]))


if __name__ == '__main__':
    main()
