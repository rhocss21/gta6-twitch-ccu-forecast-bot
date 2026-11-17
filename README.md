# GTA VI Twitch CCU Forecaster

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rhocss21/gta6-twitch-ccu-forecast-bot/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rhocss21/gta6-twitch-ccu-forecast-bot/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty on make.fun: forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), as five quantiles per hour, from everything recorded up to the end of the first hour.

## How it works

`main.py` (Python 3.13, standard library only) reads the case file, fits a
model fresh on every run from the case's own comparables — past big game
launches' hourly Twitch curves — and writes 23 rows of quantiles
(5/25/50/75/95%) for hours 2–24.

The forecast is the geometric mean of three members, all fit per hour in
log space and all *local* in log hour-1 size (Gaussian kernel over
log(V1)):

1. Local-linear OLS, bandwidth 1.0 — kernel-weighted OLS of ln Y on
   ln V1, centre at the target, spread from weighted residual quantiles,
   25% normal shrinkage, 3-hour smoothing.
2. Local-linear OLS, bandwidth 2.0 — same, wider kernel, no smoothing.
3. Local empirical quantiles, bandwidth 1.0 — kernel-weighted quantiles
   of the log-ratios ln(Y/V1), 50% normal shrinkage.

Local fits matter: the day's shape depends strongly on hour 1. Tiny
hour-1 launches (a pre-unlock first hour) explode 100–1000x by evening,
while the biggest launches barely double — a single global median curve
over-predicts a big launch's evening by ~2x. Channels and pre-release
history were tested as extra features and did not help on the
decision-relevant slices, so the model uses hour 1 alone.

## Local validation

Leave-one-launch-out over the 37 usable comparables in the kit
(all preprocessing from the 36 training launches only; exact pinball
loss on the log scale). Full table in `candidates/results.csv`,
method in `candidates/evaluate.py`, design notes in `evidence.md`.

| model | LOO mean loss | V1 ≥ 50k slice | V1 10k–300k slice | 90% coverage |
|---|---|---|---|---|
| Persistence baseline (house) | 0.930 | 0.251 | 0.385 | 0.624 |
| Global median-ratio curve (v1) | 0.714 | 0.371 | 0.393 | 0.878 |
| **This entry (ens3)** | **0.368** | **0.202** | **0.285** | **0.884** |

Nested LOO (hyperparameters re-picked inside each fold) agrees
unanimously with this configuration (0.368). These are local estimates
on past launches — the final is one unseen day, so they are not a
guarantee; the dev leaderboard's #1 is 0.169 on the four practice cases.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic (two runs byte-identical), ~0.05 s per case, stdlib only,
no network.

## License

MIT — see [LICENSE](LICENSE).
