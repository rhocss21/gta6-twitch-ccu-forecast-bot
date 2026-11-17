# Evidence — gta6-twitch-ccu-forecast-01

## Competition (verified from the kit + make.fun API, 2026-11-17)

- Slug `gta6-twitch-ccu-forecast-01` — forecast GTA VI's launch-day Twitch
  viewers hour by hour (hours 2–24, five quantiles each), from hour 1's print.
- Pot $250 USDC, top 20 paid (1st $50, 2nd $35, 3rd $25, 4th–10th $10,
  11th–20th $7). Deadline Thu 2026-11-19 08:00 WIB (01:00 UTC).
- Score: mean pinball loss on the log scale. A prize needs a loss strictly
  below the best house entry (persistence baseline ~0.387 on practice, or
  the house AI model made 2026-11-18 23:00 UTC).
- Practice: 4 past launches (Wolverine, RDR2, GoW Ragnarök, Spider-Man 2).
  Practice answers are SEALED on the server; `grade-local` does not work
  for this competition. Dev leaderboard at brief time: #1 0.169, #10 0.225.
- Program: `main.py`, Python 3.13, stdlib only, ≤8 s / 1 CPU / 1024 MB per
  case, deterministic (two runs byte-identical), output ≤65,536 bytes.

## Verified facts about the data (recomputed from the kit inputs)

- All 4 practice cases carry the SAME 37 usable comparables (38 listed;
  the 38th is the GTA VI pre-release category history, which has no
  release day and drops out of every fit). Every usable launch has a
  complete 24-hour release day and a positive hour-1 print.
- Median ratios of hour-h viewers to the hour-1 print, h2–24:
  [0.87, 0.75, 0.73, 0.75, 0.79, 0.80, 0.90, 0.88, 0.91, 0.93, 0.98,
  1.06, 1.18, 1.52, 1.65, 1.77, 1.95, 2.40, 2.37, 2.33, 2.30, 2.18, 2.04].
- IQR of the log-ratio: 0.25 (h2), 0.74 (h6), 1.21 (h12), 2.09–2.16
  (h19–20), 1.84 (h24) — dispersion explodes late; quantiles must widen
  a lot late-day.
- Hour-1 prints across comparables: min 164, median 53,526, max 875,155.
- Practice hour-1 prints: p00 Wolverine 23,357; p01 RDR2 116,550;
  p02 GoW 269,517; p03 Spider-Man 2 119,087.
- The naive median-ratio curve's late quantiles explode (q95 ratio
  300–400x at h19–24) because a few launches have tiny hour-1 prints
  (e.g. Hollow Knight: Silksong 164) and huge evenings — a pre-unlock
  hour 1, not a small launch.

## The key structural finding (drove the whole design)

The log-ratio curve is strongly NONLINEAR in log hour-1. Per-hour OLS of
ln(Y_h/V1) on ln V1 gives a ratio slope of ~0.0 early, falling to −0.93
by h19 (R² 0.87) — but that slope is mechanical: evening absolute levels
sit in a narrow band (ln Y19 10.5–13.6) while ln V1 spans 5.1–13.7.
Among the biggest launches the evening level DOES scale with hour 1
(Elden Ring V1 875k → h19 ratio 0.95; Cyberpunk 852k → 0.94), while
tiny-V1 launches still reach 100k–560k in the evening. Consequences,
all confirmed by the evaluator:

- A global median-ratio curve (v1) over-predicts a big launch's evening
  by ~2x (big-slice loss 0.371 vs 0.202 for the champion).
- A single global OLS is misspecified (big-slice 0.296) — it is dragged
  by the tiny-V1 launches and under-predicts big launches' evenings.
- Similarity weighting on ln V1 is the big win (v1 0.714 → kernel b=1
  0.397). Local-linear OLS is better still (0.377); the 3-member
  ensemble is best (0.368).
- Channels add nothing once ln V1 is in (corr lnV–lnCh 0.946).
  Pre-release 7-day mean helps ONLY the tiny-V1 slice (OLS+pre7 small
  slice 0.642 vs 0.765 for OLS) and HURTS the big slice (0.335 vs
  0.296), so it is not used — GTA VI's hour 1 will be large, and the
  final case's history only reaches back to 2026-11-17 anyway.
- Ridge shrinkage toward a zero ratio-slope hurts (κ=0.75: 0.473,
  κ=0.50: 0.548 vs κ=1.0: 0.437 for global OLS) — the slope is real,
  do not shrink it.
- Training OLS only on big launches (V1 ≥ 30k) is best-median (0.243)
  but catastrophic on small launches (mean 0.658, coverage 0.712) —
  rejected as too fragile if GTA's hour 1 surprises low.
- Tail widening (spread ×1.15) lifts coverage to ~0.89–0.91 at no mean
  cost for single models, but the chosen ensemble already covers 0.884
  and widening it costs mean loss (0.375 vs 0.373), so no extra spread.

## Champion design (v7-ens3, exactly what main.py implements)

Geometric mean (average in log space) of three per-hour members, all
fit from the case's own comparables (41 usable launches in the final
case: 37 + the 4 practice launches; the GTA VI pre-release comparable
is excluded by the completeness check):

1. Local-linear OLS, Gaussian kernel b=1.0 on ln V1 (ESS < 8 falls back
   to b=2.0): per-hour OLS of ln Y on ln V1, centre at the target,
   spread = kernel-weighted residual quantiles, λ=0.25 normal shrinkage
   (median + IQR-based sigma), 3-hour centred smoothing.
2. Local-linear OLS, kernel b=2.0, λ=0.25, no smoothing.
3. Local empirical quantiles, kernel b=1.0: weighted quantiles of
   log-ratios ln(Y/V1), λ=0.50 normal shrinkage.
4. No channels, no pre-release history — hour 1 alone.

Median log-bias of the champion is negligible (+0.065 early, +0.052
mid, +0.025 late; −0.010 on the big slice), so no bias correction.
Example forecast for a GoW-sized hour 1 (269,517): median ratios
h2 0.87, h6 0.73, h12 0.61, h19 1.21, h24 1.11 — far below the naive
all-launch median curve's evening ratios (~2.4).

## Local evaluation protocol (candidates/evaluate.py)

- Leave-one-launch-out over the 37 usable comparables: hold out one
  entire launch, train on the other 36 (all preprocessing from the 36
  only), reveal only the held-out hour-1 print (+ channels / pre7 where
  the candidate uses them), predict 23×5 quantiles, score with the
  EXACT pinball formula from RULES.txt (verified against its worked
  example: 0.112545 exactly).
- Nested LOO inside each outer training set for hyperparameter
  selection; fixed-candidate results reported separately from the
  nested-selection result.
- Slices reported because the final is ONE launch with a large hour 1:
  big (V1 ≥ 50k, n=20), practice band (V1 10k–300k, n=22), small
  (V1 < 10k, n=9). The LOO mean is dominated by the 9 tiny-V1 launches
  (their hour 1 is pre-unlock noise); the big/band slices are the
  decision-relevant ones for GTA VI.
- Bootstrap over WHOLE launches (2000 resamples) for paired CIs, never
  individual hours.
- Reference targets (NOT calibrated thresholds): dev #1 0.169,
  #10 0.225, house baseline ~0.387 on practice. A local mean below
  0.169 does NOT prove the final will beat the house bar — the final is
  one unseen day with distribution shift (minute-read mean hour 1 vs
  practice hourly averages; unlock/clock mismatch across launches).
  Report local numbers as local estimates only.

## Results (candidates/results.csv — 43 fixed candidates + nested)

LOO mean loss / median launch loss / worst-5 / 90% coverage / big slice
/ band slice; paired diffs vs baseline and vs v1 with 95% bootstrap CI
(whole launches). Headline rows:

| candidate | mean | median | worst-5 | cov | big | band | vs v0 [95% CI] | vs v1 [95% CI] |
|---|---|---|---|---|---|---|---|---|
| v0-persist (house baseline) | 0.930 | 0.325 | 3.529 | 0.624 | 0.251 | 0.385 | — | +0.216 [0.055, 0.386] |
| v1-emp-lam0 (global median curve) | 0.714 | 0.415 | 2.296 | 0.878 | 0.371 | 0.393 | −0.216 [−0.383, −0.054] | — |
| v2-kern-b1-lam05 | 0.396 | 0.254 | 1.149 | 0.866 | 0.204 | 0.296 | −0.534 [−0.851, −0.251] | −0.318 [−0.480, −0.171] |
| v3-ols-k100-lam025 | 0.437 | 0.335 | 1.022 | 0.877 | 0.296 | 0.328 | −0.493 [−0.834, −0.180] | −0.277 [−0.459, −0.107] |
| v5-ols-pre7 | 0.418 | 0.360 | 0.870 | 0.848 | 0.335 | 0.326 | −0.512 [−0.860, −0.185] | −0.296 [−0.488, −0.119] |
| v6-lols-b1-lam025-smooth | 0.376 | 0.261 | 1.091 | 0.852 | 0.203 | 0.291 | −0.554 [−0.891, −0.245] | −0.338 [−0.525, −0.165] |
| v6-lols-b1-lam05-smooth | 0.373 | 0.260 | 1.092 | 0.859 | 0.202 | 0.289 | −0.557 [−0.892, −0.248] | −0.341 [−0.529, −0.167] |
| v7-ens-lols-kern | 0.373 | 0.249 | 1.084 | 0.886 | 0.202 | 0.289 | −0.557 [−0.888, −0.258] | −0.341 [−0.521, −0.180] |
| **v7-ens3 (CHAMPION)** | **0.368** | **0.249** | **1.057** | **0.884** | **0.202** | **0.285** | **−0.562 [−0.896, −0.260]** | **−0.346 [−0.530, −0.180]** |
| nested-selection | 0.368 | 0.249 | 1.057 | 0.884 | 0.202 | 0.285 | −0.562 [−0.896, −0.260] | −0.346 [−0.530, −0.180] |

Champion per-band loss: early (h2–8) 0.334, mid (h9–15) 0.466, late
(h16–24) 0.319. Per-quantile loss: τ05 0.165, τ25 0.439, τ50 0.548,
τ75 0.475, τ95 0.213. Nested selection picks ens3 in 37/37 outer folds
— the tuning agrees with the fixed-candidate table, not against it.
(Note: in an earlier round the nested grid was smaller and picked a
mix, mean 0.385 — selection noise across near-tied candidates; the
full-grid nested run is unanimous.)

Worst launches for the champion are the tiny-V1 regime (Black Myth:
Wukong 1.679, Marathon 1.275, ARC Raiders 0.751) plus Resident Evil
Requiem (0.947); the best are the big launches (Elden Ring 0.089,
Cyberpunk 0.114, Black Ops 7 0.125) — exactly the regime GTA VI is in.

## Verification of main.py

- Reproduces the evaluator's champion EXACTLY on all 4 practice cases
  (max relative difference 0.0 after 2-decimal rounding).
- Determinism: two runs on the practice inputs byte-identical.
- Speed: ~0.2 s for all 4 practice cases (~0.05 s/case), far under 8 s.
- Outputs validate: 23 rows, hours 2–24, non-decreasing quantiles, all
  in (0, 1e9], ~1.5 KB each.
- Edge cases tested under the grader's flags (`python3 -P -s`): no
  comparables → persistence fallback; V1 = 2,000,000 (above training
  max) → valid, h19 median ratio 0.86; V1 = 500 → valid; V1 = 0 →
  floored to 1, valid; broken comparable (truncated hourly) → excluded.
- main.py was written from scratch in the build session (the starter was
  used only as a reference for the I/O contract) and revised after its
  first run; outputs unchanged by the revision.

## Risks / honest caveats

- One unseen day: the final's hour-1 print is a minute-read mean, the
  practice's is a SullyGnome hourly average — a denominator shift the
  model cannot see. Unlock/clock mismatch across launches is the same
  story: hour 1 means different things for different launches.
- GTA VI's hour 1 will likely exceed the training max (875,155). The
  kernel then concentrates on the two biggest launches (ESS fallback to
  b=2.0 below ESS 8) and the local-linear fit extrapolates its slope —
  tested at V1 = 2M, behaves sensibly, but it is extrapolation.
- The local LOO mean (0.368) is far above the dev #1 practice score
  (0.169): the LOO pool includes 9 tiny-V1 launches no practice case
  resembles. The practice-band slice (0.285) is the closer analogue,
  and even that is not the practice score — the 4 practice launches are
  not in the pool.
