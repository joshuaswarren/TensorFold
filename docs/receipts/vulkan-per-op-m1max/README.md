# Fence-timed per-op profile receipts, 16-inch M1 Max (CLX-7)

`TF_Q35_VK_PROFILE=1` on the pinned document-1024 case: Qwen3.5-2B 4-bit,
64 greedy steps, `tf-qwen35-model` built at `b437529f8` plus the
`e.prof = .{}` initialization in this PR (see "Finding" below). Two fresh
processes under the GPU turn lock, `--vendor 0x10005`, `--context 7168`.
Linux 7.3.0-2-20261110.1-lab-ARCH, Honeykrisp (Omarchy Mesa build).

## Gate

`generated_ids_hash a472a62c7e22…` matches the oracle on both in-process
runs and both processes (`ids_match_a/b`), and `logits_bf16_sha256
82c353787e89` equals the published pinned digest. Greedy tokens are
identical across the two runs.

## Per-op table (process 1, 64 rounds, per-dispatch flush; shares, not absolute)

| class | calls | ms | ms/round | share |
|---|---:|---:|---:|---:|
| dense | 35530 | 32887.18 | 513.862 | 72.6% |
| norm | 11590 | 2688.54 | 42.008 | 5.9% |
| activation | 4560 | 673.07 | 10.517 | 1.5% |
| gdn_prepare | 3420 | 2182.21 | 34.097 | 4.8% |
| gdn_scan | 3420 | 3007.99 | 47.000 | 6.6% |
| gdn_gate | 3420 | 516.17 | 8.065 | 1.1% |
| attention_prepare | 3420 | 833.77 | 13.028 | 1.8% |
| attention | 1140 | 2533.72 | 39.589 | 5.6% |

Process 2 agrees within run noise (dense 33324.76 ms / 72.3%).

## Finding

At `b437529f8` the run itself completed and the RESULT line matched, but
`report()` then panic-overflowed in `Engine.deinit`: `Engine.init` builds
the engine with `gpa.create` plus per-field assignment, so `prof` kept
uninitialized memory (`0xAA` fill on this allocator). `prof.on` read
truthy by accident, spans accumulated onto garbage `wall_ns`, and the
`u64` sum wrapped at teardown. `e.prof = .{}` restores the intended
zeroed counters — and also stops the flag-off path from reading a
garbage `prof.on`. `run1.log` holds the crash output.

## Files

- `pr1.log`, `pr2.log`: raw logs of the two profiled processes.
- `run1.log`: the panic at the pinned commit on the same fixture.
- `result-pr1.json`, `result-pr2.json`: probe result records (ids,
  hashes, oracle comparison).
- `summary.json`: setup and hashes in one place.

Not claimed: other chips, other prompts, absolute per-op times (the
per-dispatch flush changes them; the unprofiled leg gives the real
total).
