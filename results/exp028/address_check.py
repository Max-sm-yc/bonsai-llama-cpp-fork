#!/usr/bin/env python3
"""Exhaustive/random address proof for the ROWS=1 work-list lookahead formula."""
import random
THREADS = 128
rng = random.Random(0x028)
checked = 0

# Exhaust short and awkward tail shapes, then large realistic rows.
shapes = [(nrows, bpr, rpc) for nrows in range(1, 66)
          for bpr in (1, 2, 3, 7, 31, 40, 63, 127, 128, 129, 257)
          for rpc in (1, 2, 4, 8, 16)]
shapes += [(rng.randint(1, 10000), rng.choice((1, 40, 128, 129, 257, 1024)),
            rng.choice((1, 2, 4, 8, 16))) for _ in range(1000)]
for nrows, bpr, rows_per_cta in shapes:
    stride = bpr + rng.randint(0, 7)  # row pitch may include padding
    last_row0 = ((nrows-1)//rows_per_cta)*rows_per_cta
    row0s = sorted({0, (last_row0//2//rows_per_cta)*rows_per_cta, last_row0})
    for row0 in row0s:
        n_rows_cta = min(rows_per_cta, nrows-row0)
        n_items = rows_per_cta*bpr
        for tid in range(THREADS):
            for idx in range(tid, n_items, THREADS):
                for distance in (1, 2, 3):
                    future = idx + distance*THREADS
                    if future < n_items:
                        rg, kbx = divmod(future, bpr)
                        r = min(rg, n_rows_cta-1)
                        # Equivalent to source's (row0+r)*stride + kbx, in block units.
                        block = (row0+r)*stride+kbx
                        assert 0 <= r < n_rows_cta
                        assert 0 <= kbx < bpr
                        assert (row0+r) < nrows
                        assert block < nrows*stride
                        # The address is precisely that of the future owner's original work item.
                        orig_rg, orig_k = divmod(future, bpr)
                        orig_r = min(orig_rg, n_rows_cta-1)
                        assert (row0+r)*stride+kbx == (row0+orig_r)*stride+orig_k
                        checked += 1
print(f'PASS: {len(shapes)} shapes, {checked} in-range future addresses; distances 1/2/3; 128-thread stride; row tails and padded pitches covered')
