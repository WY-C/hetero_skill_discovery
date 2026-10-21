# HiSSD V3 no-observation speed optimization

> Current final configuration removes the experimental policy-KL objective and
> its teacher decoder entirely. The training timings below describe the older
> KL-enabled benchmark and are therefore conservative for the current code.

Measured on `noward` (2 x RTX 5090), 2026-10-19 KST.  The baseline is an
exact code copy of the job running on `noward2`; only results, logs, datasets,
and the intended no-observation agent/config delta were excluded from the
cross-host source comparison.

## Training benchmark

Both variants used seed 2, batch size 32, 300 optimizer steps, no evaluation,
no checkpoints, and no progress display.  Baseline and optimized variants
were run once on each GPU to cancel device variation.

| GPU | Running-code baseline | First optimized | Final teacher-batched |
| --- | ---: | ---: | ---: |
| RTX 5090 GPU 0 | 342.31 s | 338.00 s | 319.95 s |
| RTX 5090 GPU 1 | 345.26 s | 332.09 s | 323.24 s |
| Mean | 343.79 s | 335.05 s | 321.60 s |

The final mean wall time is 6.5% lower than the running-code baseline
(throughput is 6.9% higher).  In a separate simultaneous, GPU-swapped A/B,
teacher batching reduced the current optimized mean from 333.67 s to 321.60 s,
an additional 3.6% wall-time reduction (3.8% throughput increase).
Parent-process peak RSS remained approximately 5.89 GB.  Observed GPU memory
rose from about 12.5 GB to 18.9 GB, which remains below the RTX 5090's 32 GB.

Historical KL-enabled benchmark changes:

- evaluate all timestep KL terms in one batched masked-KL call;
- compute detached RMS monitoring tensors only on logged steps;
- transfer all logged scalar metrics to CPU with one synchronization;
- use fused CUDA AdamW;
- allow TF32 matmul through `torch.set_float32_matmul_precision("high")`;
- concatenate the no-grad KL teacher decoder inputs across timesteps and run
  one larger decoder call, preserving the per-timestep hidden-state inputs;
- use a 100-step metric interval and disable the per-step progress bar in the
  final launcher.

The batched KL value and gradient match the former weighted per-timestep
calculation to `atol=1e-7` in `tests/test_v3_smoke.py`.  TF32 is not bitwise
identical: the first decoder CE was 10.116764 at highest precision and
10.116753 in fast mode.  Use `--fused_adamw=False
--cuda_matmul_precision=highest` when strict floating-point reproducibility is
more important than speed.

Before KL was removed, the teacher-batched and serial variants produced
identical logged decoder CE, KL, and adapter metrics at steps 1, 101, and 201
in both GPU-swapped 300-step runs. The final code contains neither path.

An attempted static capability/team encoding deduplication did not improve the
end-to-end benchmark and was removed.

Full BF16 autocast was also rejected: it took 361.21 s versus 337.07 s for
FP32 (7.2% slower), and decoder CE at step 201 diverged to 7.598615 versus
0.664993.  It reduced GPU memory, but was neither faster nor numerically safe
for this workload.

## Evaluation benchmark

Episode count was unchanged at 100 per map.  Only the number of temporary
parallel SC2 workers changed.

| Evaluation | Workers | Evaluation time | Relative to 1 worker |
| --- | ---: | ---: | ---: |
| `2s3z`, 100 episodes | 1 | 36.931 s | baseline |
| `2s3z`, 100 episodes | 4 | 20.770 s | 43.8% less |
| `2s3z`, 100 episodes | 10 | 13.289 s | 64.0% less |
| All 13 maps, 100 episodes/map | 1 | 631.405 s | baseline |
| All 13 maps, 100 episodes/map | 10 | 192.554 s | 69.5% less (3.28x) |

The all-map optimized run took 202.81 seconds including process setup and
teardown.  All 13 maps completed and zero SC2 child processes remained.

## Final launch

The launcher inherits the official HiSSD optimizer hyperparameters instead of
overriding them: AdamW, `lr=0.0003`, `critic_lr=0.0001`, and
`weight_decay=0.0001`.  The timing benchmarks above used the earlier run's
`lr=0.0001` and `weight_decay=0.001`; these scalar values do not change the
executed model graph or optimizer kernel count.

The final objective now matches the official HiSSD objective; the experimental
difference is only the no-observation team/capability residual adapter. Policy-KL
capture, teacher decoding, hyperparameters, and metric logging are not part of
the final run.

```bash
ssh noward '~/Desktop/HiSSD_v3_noobs_speed_optimized_20261019/scripts/run_v3_noobs_optimized.sh 0 2 30000'
```

The launcher validates the code, refuses a busy GPU or duplicate PID, uses 10
evaluation workers, and writes the log to
`run_logs/noobs_optimized_seed2_30000.log`.
