#!/usr/bin/env python3
"""Summarize Exp084 and draw an SVG chart of fork/original throughput."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/exp084"
COLORS = {"PTQ1_0": "#286090", "PQ2_0": "#e07a24"}
MODES = ("prefill", "decode", "combined")
FORMATS = ("PTQ1_0", "PQ2_0")


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, int], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[(record["mode"], record["format"], record["context"])][record["arm"]].append(record)

    output = []
    for (mode, format_name, context), arms in sorted(groups.items()):
        original = sorted(arms["original"], key=lambda row: row["pair"])
        fork = sorted(arms["fork"], key=lambda row: row["pair"])
        if len(original) != 2 or len(fork) != 2:
            raise ValueError(f"expected two runs per arm for {mode}/{format_name}/{context}")
        original_tps = statistics.median(row["median_tokens_per_second"] for row in original)
        fork_tps = statistics.median(row["median_tokens_per_second"] for row in fork)
        pair_deltas = [
            100 * (fork_row["median_tokens_per_second"] / original_row["median_tokens_per_second"] - 1)
            for original_row, fork_row in zip(original, fork)
        ]
        output.append({
            "mode": mode,
            "format": format_name,
            "context": context,
            "original_tps": original_tps,
            "fork_tps": fork_tps,
            "change_percent": 100 * (fork_tps / original_tps - 1),
            "pair1_change_percent": pair_deltas[0],
            "pair2_change_percent": pair_deltas[1],
            "original_peak_gpu_mib": statistics.median(row["peak_gpu_memory_mib"] for row in original),
            "fork_peak_gpu_mib": statistics.median(row["peak_gpu_memory_mib"] for row in fork),
            "run_pairs": 2,
            "repetitions_per_run": 7,
        })
    return output


def write_csv(rows: list[dict]) -> None:
    path = RESULTS / "summary.csv"
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for field in (
                "original_tps", "fork_tps", "change_percent",
                "pair1_change_percent", "pair2_change_percent",
            ):
                csv_row[field] = f"{row[field]:.6f}"
            for field in ("original_peak_gpu_mib", "fork_peak_gpu_mib", "run_pairs", "repetitions_per_run"):
                csv_row[field] = int(row[field])
            writer.writerow(csv_row)


def write_svg(rows: list[dict], gates_by_mode: dict) -> None:
    width, height = 1200, 800
    left, right = 110, 38
    panel_top, panel_gap, panel_height = 160, 60, 130
    plot_width = width - left - right
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#17212b}.title{font-size:27px;font-weight:700}.subtitle{font-size:14px;fill:#52616d}.panel{font-size:19px;font-weight:700}.axis{font-size:12px;fill:#52616d}.label{font-size:12px}.value{font-size:12px;font-weight:700}.grid{stroke:#d8dee4;stroke-width:1}.zero{stroke:#394957;stroke-width:1.5}.bar{opacity:.9}</style>',
        '<text x="110" y="42" class="title">RTX 3080: research fork vs frozen original</text>',
        '<text x="110" y="68" class="subtitle">Fork throughput change vs original (%) | median of two reversed-order run medians</text>',
        '<text x="110" y="90" class="subtitle">Ternary Bonsai 2 27B | whiskers show the two paired deltas | 7 repetitions per run</text>',
    ]
    gate_text = "; ".join(
        f"{mode.title()} {'/'.join(f'{gate:g}°C' for gate in gates)}"
        for mode, gates in sorted(gates_by_mode.items())
    )
    parts.append(f'<text x="730" y="111" class="axis">Start gates: {html.escape(gate_text)}</text>')
    legend_y = 104
    for i, format_name in enumerate(FORMATS):
        x = 110 + i * 125
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{COLORS[format_name]}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y + 12}" class="axis">{format_name}</text>')

    for panel_index, mode in enumerate(MODES):
        panel_rows = [row for row in rows if row["mode"] == mode]
        y_top = panel_top + panel_index * (panel_height + panel_gap)
        max_abs = max((abs(value) for row in panel_rows for value in (
            row["change_percent"], row["pair1_change_percent"], row["pair2_change_percent"]
        )), default=1)
        max_abs = max(2, math.ceil(max_abs / 2) * 2)
        zero_y = y_top + panel_height / 2
        scale = (panel_height / 2 - 23) / max_abs
        parts.append(f'<text x="{left}" y="{y_top - 8}" class="panel">{html.escape(mode.title())}</text>')
        for tick in (-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs):
            y = zero_y - tick * scale
            line_class = "zero" if tick == 0 else "grid"
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" class="{line_class}"/>')
            parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:+g}%</text>')

        panel_rows.sort(key=lambda row: (FORMATS.index(row["format"]), row["context"]))
        slot = plot_width / max(1, len(panel_rows))
        bar_width = min(72, slot * 0.52)
        for index, row in enumerate(panel_rows):
            x_mid = left + slot * (index + 0.5)
            change = row["change_percent"]
            y_value = zero_y - change * scale
            bar_y = min(zero_y, y_value)
            bar_h = max(1.5, abs(y_value - zero_y))
            color = COLORS[row["format"]]
            parts.append(f'<rect x="{x_mid - bar_width / 2:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}" class="bar"/>')
            low = min(row["pair1_change_percent"], row["pair2_change_percent"])
            high = max(row["pair1_change_percent"], row["pair2_change_percent"])
            whisker_top = zero_y - high * scale
            whisker_bottom = zero_y - low * scale
            parts.append(f'<line x1="{x_mid:.1f}" y1="{whisker_top:.1f}" x2="{x_mid:.1f}" y2="{whisker_bottom:.1f}" stroke="#17212b" stroke-width="2"/>')
            parts.append(f'<line x1="{x_mid - 5:.1f}" y1="{whisker_top:.1f}" x2="{x_mid + 5:.1f}" y2="{whisker_top:.1f}" stroke="#17212b" stroke-width="2"/>')
            parts.append(f'<line x1="{x_mid - 5:.1f}" y1="{whisker_bottom:.1f}" x2="{x_mid + 5:.1f}" y2="{whisker_bottom:.1f}" stroke="#17212b" stroke-width="2"/>')
            value_y = min(bar_y, whisker_top) - 8 if change >= 0 else max(bar_y + bar_h, whisker_bottom) + 17
            parts.append(f'<text x="{x_mid:.1f}" y="{value_y:.1f}" text-anchor="middle" class="value">{change:+.2f}%</text>')
            label_y = y_top + panel_height + 17
            parts.append(f'<text x="{x_mid:.1f}" y="{label_y:.1f}" text-anchor="middle" class="label">{row["format"]}</text>')
            parts.append(f'<text x="{x_mid:.1f}" y="{label_y + 15:.1f}" text-anchor="middle" class="axis">{row["context"]} tokens</text>')

    parts.append('<text x="110" y="780" class="subtitle">Positive values are faster in the fork. Combined mode includes prompt evaluation and 128 generated tokens.</text>')
    parts.append('</svg>')
    (RESULTS / "fork-vs-original.svg").write_text("\n".join(parts) + "\n")


def main() -> int:
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete":
        raise RuntimeError("benchmark summary is not marked complete")
    rows = summarize(summary["records"])
    if len(rows) != 14:
        raise RuntimeError(f"expected 14 workload/format rows, found {len(rows)}")
    write_csv(rows)
    write_svg(rows, summary["configuration"].get("cooldown_temperature_c_by_mode", {}))
    summary["comparisons"] = rows
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {RESULTS / 'summary.csv'} and {RESULTS / 'fork-vs-original.svg'}")
    for row in rows:
        print(f"{row['mode']:8s} {row['format']:7s} {row['context']:4d}: {row['original_tps']:9.2f} -> {row['fork_tps']:9.2f} tok/s ({row['change_percent']:+.2f}%)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"plot failed: {error}", file=sys.stderr)
        raise SystemExit(1)
