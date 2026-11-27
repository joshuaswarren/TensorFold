# #5430 arm data: Mac Studio M1 Ultra (Metal, M1 Ultra 128 GB)

Arm data for the exact-grouped-path discussion in #5430, measured on a Mac Studio M1 Ultra
(Mac13,2, 128 GiB unified memory, macOS 26.6.2), stock v1.3.6 (`f56b1420`) built from source
with `zig build native -Dcpu=apple_m1 -Dversion=1.3.6` — this Mac has the Xcode Metal
toolchain, so all metallibs compiled from source and no carve was needed. Candidate: the
same build with #5430's 2-line gate drop (`sorted()` no longer requires tensor units);
`_ = k;` added for the now-unused parameter. Production oMLX legs ran co-tenant throughout
(this host serves them); no other load was added.

Checkpoint `mlx-community/Qwen3.6-35B-A3B-4bit` @ `38740b847e4cb78f352aba30aa41c76e08e6eb46`
(pulled into the engine store with `tensorfold pull`), CHANGELOG.md inputs verified against
the issue's sha256 (`40c98361…`), the issue's serve flags, and its run order: 1 uncounted
warm-up, then 5 x 4k, 5 x 16k prefill, then the 3 equality prompts x 2 rounds. The leading
`Run K.` nonce differs per prefill request so only the chat-template header tokens hit the
prompt cache, as the issue specifies. Server timings only; each binary reproduced its own
equality replies across rounds (stock: `9aa4c5593e07`/`b105daac502d`/`560391cb6977`;
patched: `45fbdfcedd39`/`c8cc596b20df`/`e6ef9e4acd40` — stock and patched differ from each
other, as the issue reports for the M4 Pro).

## Measured results (n = 5 per prefill cell, n = 6 decode samples)

| metric | stock v1.3.6 | 2-line candidate | change |
| --- | ---: | ---: | ---: |
| prefill 4k, tok/s (median) | 274.7 (249.0-383.3) | 804.0 (777.2-806.0) | +192.6% |
| prefill 16k, tok/s (median) | 295.2 (285.6-315.2) | 377.8 (276.4-500.5) | +28.0% |
| TTFT 4k, s (median) | 14.99 | 5.18 | -65.4% |
| TTFT 16k, s (median) | 55.71 | 43.46 | -22.0% |
| decode, tok/s (median) | 70.1 | 46.3 | -33.9% |

Raw runs: `stock.json` and `patched.json` in this directory (per-run numbers, token_sha).

## What the M1 Ultra arm adds to #5430

1. The prefill win from routing M1 (no tensor units) down the expert-sorted path is larger
   here than on the M4 Pro at 4k (2.93x vs 1.94x), consistent with the per-slot path's
   weight reads being more DRAM-bound on this part.
2. **Decode regressed -33.9%** on the candidate (M4 Pro reported -0.1%, neutral). Decode
   runs per-slot in both builds (the gate is prompt-only), so the regression is not the
   sorted decode path; the likely cause is memory-pressure residue: the sorted path's
   per-chunk buffers (order, offsets, sorted rows, expert gate/up/act/out at 16k =
   ~48k routed rows) stay resident after prefill and the prompt cache sheds under the
   pressure the server logged ("critical: new requests are refused with 503"). The exact
   grouped path should size or release those buffers so decode keeps its weights resident.
3. The 16k prefill candidate cell is unstable under production co-tenancy
   (276-500 tok/s spread; one earlier patched attempt hit the server's critical-pressure
   503 refusals at 16k and had to be re-run after the previous server's memory drained).
   The M1 Ultra runs production serving co-tenant; cross-arm comparisons on this host
   should state that.

Both diffs change output bits, so neither is a merge candidate by the exactness rule —
this receipt exists to size the win and name the two constraints (decode residency,
16k buffer pressure) the exact grouped path for M1-M4 must respect.
