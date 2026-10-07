#!/usr/bin/env python3
"""Validate the recorded Exp072 sampler, verifier, and emission trace."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/exp072/raw"
TARGET_LOG = RAW / "natural_r5_bundle_target_ctx512_trace.server.log"
MTP_LOG = RAW / "natural_r5_pq2_0_mtp_ctx512_trace.server.log"
ALIGNED = RAW / "aligned_logits_ctx512_qwen.csv"


def parse(path):
    text = path.read_text()
    samples = re.findall(r"MTPTRACE sample ctx=([^ ]+) idx=\d+ chosen=(-?\d+)", text)
    contexts = Counter(ctx for ctx, _ in samples)
    target_ctx = contexts.most_common(1)[0][0]
    target_samples = [int(token) for ctx, token in samples if ctx == target_ctx]
    emitted = [
        (int(position), int(token))
        for position, token in re.findall(
            r"MTPTRACE server_emit_(?:direct|spec) n_gen=(\d+)(?: row=\d+)? token=(\d+)", text
        )
    ]
    emitted.sort()
    return text, contexts, target_samples, emitted


def post_bias_scores(text, emitted_position):
    lines = text.splitlines()
    emit_index = next(
        i for i, line in enumerate(lines)
        if re.search(rf"MTPTRACE server_emit_(?:direct|spec) n_gen={emitted_position}(?: |$)", line)
    )
    for line in reversed(lines[:emit_index]):
        match = re.search(r"MTPTRACE stage .* name=logit-bias .* top=(.*)$", line)
        if match:
            return {
                int(token): float(score)
                for token, score in re.findall(r"(-?\d+):(-?inf|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", match.group(1))
            }
    raise AssertionError(f"no post-bias stage found before emission {emitted_position}")


def main():
    target_text, target_contexts, target_samples, target_emitted = parse(TARGET_LOG)
    mtp_text, mtp_contexts, mtp_samples, mtp_emitted = parse(MTP_LOG)
    assert len(target_contexts) == 1
    assert len(target_samples) == len(target_emitted) == 128
    assert len(mtp_contexts) == 2
    assert len(mtp_samples) == len(mtp_emitted) == 128
    assert [position for position, _ in target_emitted] == list(range(1, 129))
    assert [position for position, _ in mtp_emitted] == list(range(1, 129))
    assert target_samples == [token for _, token in target_emitted]
    assert mtp_samples == [token for _, token in mtp_emitted]

    with ALIGNED.open(newline="") as source:
        row = next(row for row in csv.DictReader(source) if row["generated_position"] == "66")
    assert int(row["target_token"]) == 1167
    assert int(row["mtp_token"]) == 6195
    assert re.search(r"draft=18912 target=6195 verdict=reject", mtp_text)
    target_scores = post_bias_scores(target_text, 67)
    mtp_scores = post_bias_scores(mtp_text, 67)
    target_gap = target_scores[1167] - target_scores[6195]
    mtp_gap = mtp_scores[6195] - mtp_scores[1167]
    assert int(row["target_token"]) == 1167 and int(row["mtp_token"]) == 6195
    result = {
        "target_only_samples_match_emissions": len(target_samples),
        "mtp_target_samples_match_emissions": len(mtp_samples),
        "first_divergence_zero_based_position": 66,
        "target_only_token": 1167,
        "mtp_draft": 18912,
        "mtp_verdict": "reject",
        "mtp_target_emission": 6195,
        "post_bias_scores": {
            "target_only": {"token_1167": target_scores[1167], "token_6195": target_scores[6195]},
            "mtp_batched_target": {"token_6195": mtp_scores[6195], "token_1167": mtp_scores[1167]},
        },
        "target_post_bias_margin_1167_over_6195": round(target_gap, 8),
        "mtp_post_bias_margin_6195_over_1167": round(mtp_gap, 8),
        "relative_gap_shift": round(target_gap + mtp_gap, 8),
    }
    assert abs(result["target_post_bias_margin_1167_over_6195"] - 0.2885065) < 1e-6
    assert abs(result["mtp_post_bias_margin_6195_over_1167"] - 0.00583839) < 1e-6
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
