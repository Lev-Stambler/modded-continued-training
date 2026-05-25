# nanoFineTune

Competitive single-H100 fine-tuning speedrun on Modal.

Inspired by [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

## What

nanoFineTune measures how far a pretrained language model's heldout eval loss
can drop during a fixed wall-clock fine-tuning window on a single H100 GPU.

The current default track starts from `Qwen/Qwen3.5-4B-Base`, uses a
continued-pretraining objective on `HuggingFaceTB/finemath`
(`finemath-4plus`), and is scored by:

```
score = baseline_eval_loss − final_eval_loss
```

Higher is better. The timer starts after model load, eval-cache prep,
optimizer setup, and baseline eval. Compilation, graph capture, autotuning, and
train-shaped warmup all consume the selected track budget. The final eval runs
after the timed train loop.

The baseline iteration uses PEFT LoRA by default: all linear language-model
layers get rank-32 adapters, the base checkpoint stays frozen, and full
fine-tuning remains available with `--tuning-mode full`.

## Tracks

| Track | Budget | Default command |
|-------|--------|-----------------|
| 1     | 30 min | `./run.sh` |
| 2     | 5 min  | `./run.sh track2` |
| 3     | 2 hr   | `./run.sh track3` |

Override the budget explicitly with `--minutes` (default 0 = use track
default).

## Rules

New records must:

1. **Not modify the model or data source.** The model checkpoint, dataset,
   dataset config, and their pinned revisions are fixed (see Fixed Inputs
   below). You may not change these to different models or datasets.
2. **Not modify the eval data pipeline.** The eval set construction (first N
   non-empty documents from the unshuffled train stream, packed into
   `eval_blocks × seq_len` blocks) must remain identical. You may change
   `eval_blocks`, `seq_len`, or the eval batch size, but not the underlying
   stream of tokens.
3. **Not reward hack against the fixed dataset or eval construction.** Changes
   designed to exploit dataset quirks, memorized ordering, eval-set leakage, or
   other source-specific shortcuts are not valid records. Future validation may
   run submissions across multiple heldout datasets or dataset configs to catch
   overfitting to a single source.
4. **Attain a positive eval loss drop.** `baseline_eval_loss - final_eval_loss`
   must be > 0. Due to inter-run variance, submissions targeting a new record
   should provide enough run logs to attain statistical significance at p < 0.01
   that the mean eval loss drop is positive.
5. **Run on a single H100 via Modal.** The hardware is fixed. The run must use
   the Modal image defined in `main.py`.
6. **Count compilation work against the track budget.** `torch.compile`,
   graph capture, autotuning, recompilation, and train-shaped compile warmup
   are allowed, but they consume the same timed budget as training.
7. **Beat the prior record.** When baselined on the same hardware, the new run
   must achieve a higher eval loss drop than the previous record.

Other than that, anything and everything is fair game:

- Optimizer choice, learning rate schedules, weight decay
- Batch size, gradient accumulation, sequence length
- Attention implementation (FA2, FlexAttention, SDPA, etc.)
- Model-aware optimizations that use the underlying architecture, layer layout,
  parameter shapes, attention/MLP structure, or other implementation details
- Architecture and trainable-structure changes (freeze layers, add adapters,
  replace modules, add auxiliary parameters, alter which weights are updated,
  etc.)
- Training data ordering, shuffling, document packing strategies
- Mixed precision, compilation, kernel optimizations
- Novel training techniques (Muon, value embeddings, etc.)

In other words, the starting checkpoint is fixed, but the trainer does not need
to treat the model as a black box. Submissions may incorporate knowledge of the
Qwen3.5 architecture directly into their optimization strategy, provided they do
not change the fixed input model to a different checkpoint or exploit eval/data
leakage.

### Discretionary

A PR may not be accepted if it:

- Disproportionately degrades code readability for a marginal gain.
- Substantially narrows the loss-drop buffer without outperforming simpler
  alternatives at equivalent loss.

## Fixed Inputs

| Input | Value | Revision |
|-------|-------|----------|
| Model | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` |
| Dataset | `HuggingFaceTB/finemath` | `e92b25a616738fe95dc186b64dfb19f9c8525594` |
| Dataset config | `finemath-4plus` | — |

All are public and ungated.

## Quick start

Install local launcher dependencies and authenticate Modal:

```bash
uv sync
uv run modal setup
```

If Modal is already configured, verify the active profile:

```bash
uv run modal profile list
```

Short smoke test:

```bash
./run.sh smoke
```

This uses the fastest path: short budget, two eval blocks, SDPA attention, and
no model compile or compile warmup.

Full fine-tune compatibility smoke test:

```bash
./run.sh full-smoke
```

Compiled full fine-tune smoke test:

```bash
./run.sh full-compile-smoke
```

Launch a 30-minute run (Track 1):

```bash
./run.sh
```

5-minute sprint (Track 2):

```bash
./run.sh track2
```

2-hour endurance (Track 3):

```bash
./run.sh track3
```

## Submitting a record

Run with `--record-description` and `--record-contributors`:

```bash
./run.sh \
  --record-description "MuonOptimizer" \
  --record-contributors "@yourhandle"
```

This saves a local record folder under `records/track_N_<budget>/` after the
Modal run returns. The folder contains:

- `main.py` — full source code snapshot (like modded-nanogpt)
- `config.json` — all hyperparameters
- `summary.json` — full run metrics
- `record.txt` — human-readable summary
- `metrics.jsonl` — event log from the run

Open a PR with the new record folder. The PR should:

1. Include at least 3 runs for statistical significance.
2. Clearly describe what changed vs. the previous record.
3. List all contributors.
4. Update the record history table in this README.

## Record history

Current baseline status: the v1 LoRA run keeps
`compile_mode=max-autotune-no-cudagraphs`. CUDA graphs were tried with
`max-autotune` and failed during the PEFT LoRA path with a CUDAGraph overwritten
tensor error, so they are not part of the baseline. The v1 Track 1 run compiled
and completed, but its eval loss drop was negative; treat it as a logged
baseline iteration, not a valid competition record under the positive-loss-drop
rule. This saved v1 snapshot predates the scoring cleanup that moved compile
and warmup inside the timed budget; future comparable records should run on the
current trainer. The first positive Track 1 record is the Hadamard lowpass saved
activation run below.

### Track 1 — 30 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| v5 | +0.000477 | Full-stack doc-safe Hadamard lowpass (hidden scope, chunk 64, keep 16, int8 retained coeffs, suffixes 9216/2560 only, mbs=3, loss_chunk=128, manual CE, bf16 train logits, no model/loss checkpointing, AdamW fused, lr=1e-8, flex_attention, torch.compile max-autotune-no-cudagraphs); 1331 steps, 16.36M tokens, 9081.4 tok/s | 2026-05-25 | [summary](records/track_1_30min/2026-05-25_Doc-safe_span-local_Hadamard_lowpass_int8_chunk64_keep16_mbs3_flex_compile_manual_CE_loss_chunk128_no_checkpoint_chunked_eval/summary.json) | codex |
| exp | +0.000461 | Tail-tier doc-safe Hadamard lowpass (hidden scope, primary chunk 128, keep 32, tail tiers 64/32 with ratio-scaled keeps, int8 retained coeffs, suffixes 9216/2560 only, mbs=3, loss_chunk=128, manual CE, bf16 train logits, no model/loss checkpointing, AdamW fused, lr=1e-8, flex_attention, torch.compile max-autotune-no-cudagraphs); 1326 steps, 16.29M tokens, 9047.4 tok/s; reduced peak allocation by 1.67% and exact-tail bytes by 50.5% vs v5 but did not beat v5 score | 2026-05-25 | [summary](records/track_1_30min/2026-05-25_Doc-safe_tail-tier_Hadamard_lowpass_int8_chunk128_keep32_tail32_mbs3_flex_compile_manual_CE_loss_chunk128_no_checkpoint_chunked_eval/summary.json) | codex |
| v4 | +0.000089 | Comparable full-stack doc-safe Hadamard lowpass (hidden scope, chunk 32, keep 30, int8 retained coeffs, suffixes 9216/2560 only, mbs=3, loss_chunk=256, manual CE checkpoint, bf16 logits, AdamW fused, lr=1e-8, flex_attention, torch.compile max-autotune-no-cudagraphs); 1122 steps, 13.79M tokens, 7657.1 tok/s | 2026-05-24 | [summary](records/track_1_30min/2026-05-24_Doc-safe_span-local_Hadamard_lowpass_int8_chunk_32_keep_30_mbs_3_flex_compile_manual_CE_checkpoint/summary.json) | codex |
| v3 | +0.000858 | Doc-safe span-local Hadamard lowpass activations (hidden scope, chunk 32, keep 16, suffixes 9216/2560 only, mbs=2, no checkpointing, AdamW fused, lr=1e-8, SDPA, no torch.compile); 2312 steps, 18.94M tokens, 10520.3 tok/s | 2026-05-24 | [summary](records/track_1_30min/2026-05-24_Doc-safe_span-local_Hadamard_lowpass_activations_chunk_32_keep_16_suffixes_9216_2560_mbs_2_AdamW_fused_lr_1e-8_SDPA_no_torch.compile/summary.json) | codex |
| v2 | +0.000109 | Hadamard lowpass saved activations (hidden scope, chunk 256, keep 128, mbs=2, no checkpointing, AdamW fused, lr=1e-8, SDPA, no compile); 1739 steps, 14.25M tokens, 7912.0 tok/s | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_hadamard-lowpass-chunk256-keep128-mbs2-lr1e-8/summary.json) | codex |
| exp | -0.000030 | Peak-memory Hadamard lowpass saved activations (hidden scope, chunk 128, keep 80, mbs=2, lr=1e-8, SDPA, no compile); 2401 steps, 19.67M tokens, 10926.1 tok/s | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_hadamard-lowpass-chunk128-keep80-mbs2-lr1e-8/summary.json) | codex |
| exp | -0.000135 | More aggressive Hadamard lowpass plus chunked train loss (hidden scope, chunk 64, keep 16, mbs=3, loss_chunk=1024, lr=1e-8, SDPA, no compile); 1644 steps, 20.20M tokens, 11221.3 tok/s | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_hadamard-lowpass-chunk64-keep16-mbs3-losschunk1024-lr1e-8/summary.json) | codex |
| exp | -0.000267 | Hadamard lowpass saved activations (same as v2, lr=2e-8); 1898 steps, 15.55M tokens, 8637.7 tok/s | 2026-05-22 | [summary](records/track_1_30min/2026-05-22_hadamard-lowpass-chunk256-keep128-mbs2-lr2e-8/summary.json) | codex |
| v1 | -0.050743 | LoRA baseline (all-linear r32, AdamW fused, lr=2e-4, seq=4096, max-autotune-no-cudagraphs); 488 steps, 15.99M tokens, 8872.6 tok/s | 2026-05-21 | [summary](records/track_1_30min/2026-05-21_v1_LoRA_Track_1_30min_compiled_baseline/summary.json) | — |

### Track 2 — 5 minutes

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run | — | — | — |

### Track 3 — 2 hours

| # | Loss drop | Description | Date | Log | Contributors |
|---|-----------|-------------|------|-----|--------------|
| 1 | — | Baseline run | — | — | — |

## Useful flags

```bash
./run.sh --minutes 30 --seq-len 4096 --micro-batch-size 1 --grad-accum 8
```

Tuning modes:

```bash
./run.sh --tuning-mode lora
./run.sh --tuning-mode full
```

LoRA baseline knobs:

```bash
./run.sh --lora-r 32 --lora-alpha 64 --lora-target-modules all-linear
./run.sh --gradient-checkpointing true
./run.sh --gradient-checkpointing false
```

Learning-rate schedule knobs:

```bash
./run.sh --lr-schedule constant --lr 1e-8
./run.sh --lr-schedule cosine --lr 1e-7 --lr-final 1e-8 --warmup-steps 50
./run.sh --lr-schedule linear --lr 1e-7 --lr-final 1e-8 --warmup-steps 50
```

`constant` preserves the previous behavior. `linear` and `cosine` treat `--lr`
as the post-warmup peak and decay toward `--lr-final` over the remaining timed
training loop.

Activation save compression experiment:

```bash
./run.sh --activation-filter hadamard-lowpass --activation-filter-chunk-size 128 --activation-filter-keep 64
./run.sh smoke --activation-filter hadamard-lowpass --gradient-checkpointing false
./run.sh smoke --activation-filter hadamard-lowpass --gradient-checkpointing true
uv run modal run main.py --minutes 2 --eval-blocks 16 --grad-accum 1 \
  --micro-batch-size 2 --lr 5e-7 --warmup-steps 0 \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-chunk-size 256 --activation-filter-keep 128 \
  --gradient-checkpointing false --no-compile-model --no-compile-warmup \
  --attn-implementation sdpa
uv run modal run main.py --minutes 5 --eval-blocks 16 --grad-accum 1 \
  --micro-batch-size 3 --lr 1e-8 --warmup-steps 0 --loss-chunk-size 1024 \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-kernel dense --activation-filter-chunk-size 64 \
  --activation-filter-keep 16 --gradient-checkpointing false \
  --no-compile-model --no-compile-warmup --attn-implementation sdpa
uv run modal run main.py --minutes 5 --eval-blocks 64 --grad-accum 1 \
  --micro-batch-size 2 --lr-schedule cosine --lr 1e-7 --lr-final 1e-8 \
  --warmup-steps 50 --activation-filter hadamard-lowpass \
  --activation-filter-scope hidden --activation-filter-kernel dense \
  --activation-filter-chunk-size 128 --activation-filter-keep 80 \
  --gradient-checkpointing false --no-compile-model --no-compile-warmup \
  --attn-implementation sdpa
uv run modal run main.py --minutes 30 --eval-blocks 64 --grad-accum 1 \
  --micro-batch-size 2 --lr 1e-8 --warmup-steps 0 \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-kernel dense --activation-filter-chunk-size 32 \
  --activation-filter-keep 16 --activation-filter-suffix-numel 9216,2560 \
  --gradient-checkpointing false --no-compile-model --no-compile-warmup \
  --attn-implementation sdpa
uv run modal run main.py --minutes 30 --eval-blocks 64 --grad-accum 1 \
  --micro-batch-size 3 --lr 1e-8 --warmup-steps 0 \
  --loss-chunk-size 256 --no-loss-chunk-logits-fp32 \
  --loss-chunk-manual-ce --loss-chunk-checkpoint \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-kernel dense --activation-filter-quantization int8 \
  --activation-filter-chunk-size 32 --activation-filter-keep 30 \
  --activation-filter-suffix-numel 9216,2560 \
  --gradient-checkpointing false --attn-implementation flex_attention \
  --compile-model --compile-warmup \
  --compile-mode max-autotune-no-cudagraphs
uv run modal run main.py --minutes 30 --eval-blocks 64 --grad-accum 1 \
  --micro-batch-size 3 --lr 1e-8 --warmup-steps 0 \
  --loss-chunk-size 128 --no-loss-chunk-logits-fp32 \
  --loss-chunk-manual-ce \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-kernel dense --activation-filter-quantization int8 \
  --activation-filter-chunk-size 128 --activation-filter-keep 32 \
  --activation-filter-tail-min-chunk-size 32 \
  --activation-filter-suffix-numel 9216,2560 \
  --gradient-checkpointing false --attn-implementation flex_attention \
  --compile-model --compile-warmup \
  --compile-mode max-autotune-no-cudagraphs
```

Use `train_peak_cuda_allocated_bytes` as the memory comparison metric; the
packed/original byte counters are diagnostics for how much saved activation data
the hook compressed.
`activation_filter_scope="hidden"` compresses `[batch, tokens, coordinates]`
hidden activations only. `activation_filter_scope="token-shaped"` also includes
2D flattened `[batch*tokens, coordinates]` tensors and is more aggressive.
Training packing carries document IDs alongside each token. Hadamard lowpass
uses span-local chunks: within each contiguous source-document span in the
packed batch, full chunks are lowpass-compressed and the per-document remainder
tokens are saved exactly. It never compresses across an EOS/document boundary.
`activation_filter_tail_min_chunk_size` enables no-cross-document tail tiers.
For example, chunk 128 / keep 32 with tail min 32 greedily compresses each
document span as 128-token, then 64-token, then 32-token blocks; keep counts are
ratio-scaled to 32, 16, and 8 respectively, and only the final sub-32-token tail
is saved exactly. The default `0` preserves the single-primary-chunk behavior.
`activation_filter_kernel="dense"` uses a small dense Hadamard projection and is
currently faster on H100 for these shapes. `activation_filter_kernel="fwht"`
uses an add/sub fast Walsh-Hadamard transform path, but the current PyTorch
implementation is slower despite the lower algorithmic FLOP count.
`activation_filter_quantization` can be `none`, `int8`, or `int4`. Quantization
is applied only to the retained lowpass coefficients; exact per-document
remainders are still saved in the original dtype.
`activation_filter_max_suffix_numel` and `activation_filter_suffix_numel` can
cap or allow-list coordinate widths for shape-selective probes.
`activation_filter_suffix_keep` accepts comma-separated `suffix_numel:keep`
pairs for per-width keep schedules.
`loss_chunk_size` computes the train lm-head and cross-entropy in token chunks
to avoid allocating the full `[batch, tokens, vocab]` loss workspace. This was
needed to make `micro_batch_size=3` fit after activation compression.
`loss_chunk_manual_ce` computes the CE as `logsumexp - target_logit` inside each
chunk, and `loss_chunk_checkpoint` activation-checkpoints that chunked loss
path. `loss_chunk_logits_fp32` defaults to true; `--no-loss-chunk-logits-fp32`
keeps logits in bf16 to reduce memory.
When `loss_chunk_size > 0`, eval also uses the chunked lm-head/CE path so large
packed micro-batches do not materialize the full fp32 `[batch, tokens, vocab]`
logit tensor.
`empty_cache_every` is available for allocator-fragmentation experiments, but it
hurts throughput and should stay `0` for the current best path.
Boundary-safety diagnostics are logged as `activation_filter_doc_spans`,
`activation_filter_lowpass_chunks`, `activation_filter_lowpass_tokens`,
`activation_filter_exact_remainder_tokens`, and
`activation_filter_exact_remainder_saved_bytes`.

Smoke peak-memory notes on `seq_len=4096`, `grad_accum=1`, SDPA, no compile:

- `micro_batch_size=1`, no checkpointing: filter reduced train peak from ~49.2 GB to ~41.8 GB.
- `micro_batch_size=1`, checkpointing: filter increased train peak from ~24.4 GB to ~26.2 GB.
- `micro_batch_size=2`, no checkpointing: unfiltered OOMed at ~80.7 GB allocated.
- `micro_batch_size=2`, no checkpointing, token-shaped filtered: `keep=64` completed at ~73.4 GB, but 2-minute learning diverged even after large LR reductions.
- `micro_batch_size=2`, no checkpointing, hidden filtered, chunk 128 / keep 64: fits one step at ~78.4 GB allocated but was too tight before fixing the CUDA allocator env.
- `micro_batch_size=2`, no checkpointing, hidden filtered, chunk 128 / keep 48: repeated steps fit at ~74.5 GB allocated, but default LR produced NaNs.
- `micro_batch_size=2`, no checkpointing, hidden filtered, chunk 256 / keep 128: repeated steps fit at ~78.8 GB allocated and ~82.5 GB reserved with the corrected `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Best short validation so far: chunk 256 / keep 128, hidden scope, `lr=5e-7`, `warmup_steps=0`, `eval_blocks=16`, `minutes=2`, `micro_batch_size=2`, no checkpointing. It ran 29 steps, saved activation bytes at 50%, and improved eval loss by `+0.000122`.
- Longer 5-minute / 16-block validation is near neutral: `lr=1e-7` gave `eval_loss_drop=-0.000007`, while `lr=2e-7` gave `-0.001135`.
- Full 30-minute / 64-block Track 1 record attempts use the same chunk 256 / keep 128, hidden-scope, no-checkpointing profile. `lr=1e-8` produced `eval_loss_drop=+0.000109` with 14.25M tokens and 78.8 GB peak allocated; `lr=2e-8` produced `eval_loss_drop=-0.000267` with 15.55M tokens and the same peak allocation.
- `micro_batch_size=3`, hidden filtered, chunk 64 / keep 16, no checkpointing, chunked train loss at 1024 tokens: fits at ~75.6 GB allocated / ~79.3 GB reserved. A 5-minute / 16-block validation was slightly positive (`+0.000041`) at 2.85M tokens and 9497.0 tok/s, but the full 30-minute / 64-block attempt regressed (`-0.000135`) despite 20.20M tokens and 11221.3 tok/s.
- High constant LR does not help this compressed path. On `micro_batch_size=3`, chunk 64 / keep 16, loss_chunk 1024, 5-minute / 64-block probes gave `lr=1e-7 -> -0.000300` and `lr=1e-6 -> -0.121335`; a 2-minute / 2-block `lr=1e-5` probe collapsed by `-1.080272`. Treat `1e-4` as ruled out without a different warmup/decay schedule.
- Additional peak-memory probes at `micro_batch_size=2`, hidden scope, `lr=1e-8`: chunk 64 / keep 32 ran 5 minutes / 64 eval blocks at ~74.8 GB allocated and regressed by `-0.000016`; chunk 64 / keep 40 used ~79.2 GB and regressed by `-0.000074`; chunk 128 / keep 64 regressed by `-0.000024`; chunk 128 / keep 80 was slightly positive in a 5-minute probe (`+0.000033`) but the full 30-minute run regressed by `-0.000030` while using ~79.2 GB allocated and 10926.1 tok/s.
- Per-width schedules that compressed MLP suffix `9216` harder while keeping hidden/attention widths `2560/4096/2048/1024` at higher keep rates fit, but both tested 2-minute / 16-block profiles regressed by about `-0.000175`.
- First doc-safe boundary implementation saved whole cross-document global
  chunks exactly; chunk 256 / keep 128 OOMed at step 6 because boundary chunks
  consumed too much memory. The current span-local implementation avoids that by
  compressing full chunks within each document span and saving only per-document
  leftovers exactly.
- Span-local chunk 32 / keep 8 across all hidden-scope shapes was very memory
  efficient but produced NaN after one update; keep 16 across all shapes also
  NaNed. Restricting compression to suffix widths `9216,2560` was finite.
- With suffixes `9216,2560`, chunk 32 / keep 8, `micro_batch_size=2`, no
  torch.compile, 5 minutes / 64 eval blocks ran 319 steps, 2.61M tokens,
  8709.6 tok/s, ~72.3 GB peak allocated, and regressed by `-0.000044`.
- The same selective chunk 32 path with keep 16 used ~78.3 GB peak allocated and
  was near-neutral in 5 minutes (`+0.000001`). The full 30-minute Track 1 record
  improved by `+0.000858` with 18.94M tokens and 10520.3 tok/s.
- Cold `torch.compile(max-autotune-no-cudagraphs)` is expensive under the timed
  budget: a 5-minute chunk 32 / keep 8 selective probe spent about 269 seconds
  compiling/warming up, completed only 41 steps, and regressed by `-0.001257`
  despite a steady train-loop rate around 10.45k tok/s.
- Earlier attempts to spend the freed memory on `micro_batch_size=3` under SDPA
  were not competitive: loss chunks of 1024 and 512 OOMed; loss chunk 256 plus
  `empty_cache_every=1` completed but only reached 4539.2 tok/s and regressed by
  `-0.000080`.
- Full comparable flex_attention + `torch.compile(max-autotune-no-cudagraphs)`
  runs need the chunked manual CE path. Chunk 32 / keep 16, no quantization,
  `micro_batch_size=3`, loss chunk 256, manual CE checkpoint used ~79 GB peak
  allocated but regressed by about `-0.00031`; fp32 loss logits did not fix it.
- Quantized retained coefficients trade speed for fidelity. Chunk 32 / keep 24
  with int4 used ~67.7 GB and regressed by `-0.000071`; keep 28 with int8 used
  ~74.9 GB and was essentially neutral (`+0.000002`); keep 30 with int8 used
  ~75.8 GB and improved by `+0.000089` under the full flex/compile stack.
- Under the no-checkpoint full flex/compile path, chunk 32 / keep 8 was fast
  but too lossy in a full record (`-0.000652`), chunk 32 / keep 10 fit at
  ~82.0 GB and was near-neutral in a 5-minute probe (`-0.000163`), and chunk 32
  / keep 11 or 12 OOMed during FLA/Triton backward autotuning. Switching to
  chunk 64 / keep 16 preserved the aggressive 25% retained-coefficient ratio,
  stayed doc-safe, fit at ~82.3 GB allocated, and produced the v5 positive full
  record (`+0.000477`) at 16.36M tokens.
- Tail-tier compression with chunk 128 / keep 32 / tail min 32 fit at
  ~80.9 GB allocated and cut exact-tail saved bytes by 50.5% versus v5, but the
  full 30-minute score was slightly lower (`+0.000461`, 16.29M tokens). The
  comparable 5-minute probe improved tokens by 3.6% over the chunk 64 / keep 16
  probe and flipped the short-run eval drop positive, but this did not carry to
  a new full-run record.
- `micro_batch_size=4` with int4 keep 16 saved activation memory aggressively
  but OOMed during FLA/Triton backward autotuning. Tail-tier chunk 128 / keep 32
  also OOMed at `micro_batch_size=4` under the full flex/compile path with int8
  and int4 retained coefficients, even with smaller CE chunks, so the current
  full-stack path stays at `micro_batch_size=3`.

For lowest peak memory, use checkpointing without the filter. For the Hadamard
experiment's GPU-utilization path, use no-checkpoint larger-batch runs such as:

```bash
./run.sh smoke --micro-batch-size 2 --activation-filter hadamard-lowpass \
  --activation-filter-scope hidden --activation-filter-chunk-size 256 \
  --activation-filter-keep 128 --gradient-checkpointing false
./run.sh smoke --micro-batch-size 3 --loss-chunk-size 1024 \
  --activation-filter hadamard-lowpass --activation-filter-scope hidden \
  --activation-filter-kernel dense --activation-filter-chunk-size 64 \
  --activation-filter-keep 16 --gradient-checkpointing false
```

Optimizer choices:

```bash
./run.sh --optimizer-name auto
./run.sh --optimizer-name adamw8bit
./run.sh --optimizer-name adamw_fused
./run.sh --optimizer-name muon
```

Attention backends or disable compile:

```bash
./run.sh --attn-implementation flash_attention_2
./run.sh --attn-implementation sdpa --no-compile-model
```

Save final weights:

```bash
./run.sh --save-final
```

In LoRA mode, `--save-final` writes adapter weights. In full mode, it writes
the full model.

## Architecture

`main.py` is the canonical training source file. Like modded-nanogpt, new
optimization attempts should directly edit the current trainer. Accepted
records preserve source snapshots under `records/` so old runs remain
reproducible after the trainer evolves.

The local launcher uses `uv` (`pyproject.toml`, `uv.lock`) and `run.sh`. The
remote training environment is still defined inside `main.py`, which builds a
Modal image with:

- Current Hugging Face Transformers for Qwen3.5 support
- NVIDIA CUDA devel base image so source-built CUDA extensions have `nvcc`
- H100 CUDA build env defaults, including `TORCH_CUDA_ARCH_LIST=9.0`
- `attn_implementation="flex_attention"` by default, with `flash-attn` for FA2 fallback
- `flash-linear-attention`, `causal-conv1d`, and `tilelang` for Qwen3.5 Gated DeltaNet layers
- `peft` LoRA support; default mode applies all-linear rank-32 adapters before compile
- Sequence packing from streamed FineMath documents into fixed `seq_len` blocks
- `torch.compile(..., dynamic=False)` plus a train-shaped compile warmup inside the track budget
- `optimizer_name="auto"` defaults to fused AdamW for LoRA and `AdamW8bit` for full fine-tuning
- Optional `lr_schedule="constant|linear|cosine"` applies step warmup and timed decay to `lr_final`
- LoRA `gradient_checkpointing="auto"` starts without checkpointing and retries the timed warmup with checkpointing if CUDA OOMs
- Optional `activation_filter="hadamard-lowpass"` compresses autograd-saved CUDA activations across configurable token chunks before backward reconstruction
- Optional `loss_chunk_size` chunks train-time lm-head and cross-entropy memory so larger micro-batches can use the memory freed by activation compression
- Optional Muon, with 2D matrix weights on Muon and embeddings/norms/biases/head on `AdamW8bit`

Artifacts are written to the `nanofinetune-cache` Modal volume:

- `/cache/runs/<run_id>/config.json`
- `/cache/runs/<run_id>/metrics.jsonl`
- `/cache/runs/<run_id>/summary.json`
- `/cache/eval/<hash>.pt` for the deterministic fixed eval blocks

When `--record-description` is provided, `main.py` also returns the source,
config, summary, record text, and metrics log to the local entrypoint, which
writes the canonical `records/` snapshot in this repository.

## Scoring detail

```
score = baseline_eval_loss - final_eval_loss
```

The timer starts after model load, eval-cache prep, optimizer setup, and
baseline eval. Compilation, graph capture, autotuning, recompilation, and
train-shaped compile warmup all consume the selected track budget. The final
eval runs after the timed train loop.
