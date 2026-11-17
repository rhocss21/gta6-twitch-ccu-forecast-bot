# Evidence brief — gta6-twitch-ccu-forecast-01 (prepared 2026-11-17 ~15:30 WIB, research stage)

## Competition (verified from make.fun API, 2026-11-17 ~15:20 WIB)
- Slug: `gta6-twitch-ccu-forecast-01` — "Forecast GTA VI's launch-day Twitch viewers hour by hour"
- Pot: **$250 USDC, top 20 paid** — 1st $50, 2nd $35, 3rd $25, 4th–10th $10, 11th–20th $7. Entry fee $0.00 (`entryFeeUsd: null`, `entryFeesUsd: "0.00"`). Escrow LOCKED $250 USDC on Solana mainnet (escrow `DKKcfLvqUmjTVZRxcNtNQWh6GNJSUQ4qkwH2YVSknB87`, mint EPjF… — USDC).
- Opened: 2026-11-17 06:00 UTC (13:00 WIB today). Deadline: **2026-11-19 01:00 UTC = Thu 2026-11-19 08:00 WIB** (~40 h from prep). Race runs 19 Nov 01:00 UTC → 20 Nov 00:00 UTC, scored hour by hour.
- Status open, accepting submissions. Submissions: 114 total (API detail); 33 ranked on dev leaderboard at check time.
- This bounty is NEW: it was NOT in `known-competitions.json` (updated 13:40 WIB today) and NOT in the 07:16 daily sweep / 07:00 bounty hunt — it appeared midday. Standing auto-enter rule applies (free only, no asking).

## Task (from kit RULES.txt, kit downloaded + extracted)
- Program `main.py` (Python 3.13, stdlib only, no network) or `main.ts` (Node 22) at repo root; reads one case `/input/<id>.in`, writes `/output/<id>.out`: for hours 2–24, five quantiles (5/25/50/75/95%) of that hour's Twitch viewers in the GTA VI category.
- Score: mean pinball loss on the log scale over 23 hours × 5 quantiles. Lower is better. A prize needs loss strictly below the best house entry (house baseline OR house AI model, whichever lower). House model is made 2026-11-18 23:00 UTC on history to then.
- Constraints: ≤8 s, 1 CPU, 1024 MB per case; deterministic (two runs byte-identical); ≤10 submissions/day, 20 total; last submission before deadline is final.
- Practice: 4 past launches (Wolverine 2026-09-15, RDR2 2018-10-26, GoW Ragnarök 2022-11-09, Spider-Man 2 2023-10-20), each case = history up to end of hour 1 + hour-1 print; answers sealed on server.
- Case input: `known` (hour-1 print), `forecast` (hours 2–24), `history` (minute reads in final; hourly SullyGnome points in practice), `history_hourly`, `comparables` (38 in practice, 42 in final — past launches' hourly counts 14 days before to 3 days after release day; includes GTA VI's own pre-release category history from 2026-08-27 premiere).
- Kit: `~/workspace/tools/makefun/gta6-twitch-ccu-forecast-01-kit/` (tarball + extracted `gta6-twitch-ccu-forecast-01-kit/` with RULES.txt, competition.yaml, cli/, input/p00–p03.in, starter/main.py + main.ts).
- `grade-local` does NOT work for this competition: "scorer and reference data are not published in the kit, so it is graded on the server only" (verified 15:25 WIB). Practice score comes only from a real submission — budget submissions accordingly (10/day).
- Transcript gate: build session must start at/after 2026-11-17 06:00 UTC (this research run is fine as prep; the BUILD session must be fresh). Entry needs GitHub repo owned by rhocss21, created after the competition opens, OSI licence, make.fun README badges + About (website make.fun, topic `makefun`), and bundled AI transcripts with usage rows.

## Dev leaderboard (live, 2026-11-17 ~15:20 WIB, 33 ranked)
- #1 yamanotoai 0.169 ("Comparable-Shape Forecaster — OLS on Log Hour-1", miss 28.2%, cov 0.935, 1210 ms)
- #2 golden-raven-21 0.180 ("local-linear log-ratio model"); #3 vinkyboysz 0.190; #4 sage-lark-29 0.191; #5 tukiman 0.196; #10 brisk-heron-24 0.225.
- No house/baseline row ranked yet — house bar is still unknown; the house model lands 2026-11-18 23:00 UTC.
- Winning pattern is visible in the titles: comparable-shape / log-ratio models, NOT the starter persistence forecast.

## Offline shape analysis (computed 2026-11-17 ~15:28 WIB from p00.in comparables, 37 usable of 38)
Median ratio of hour-h viewers to hour-1 print, across past big launches:
- Hours 2–5 DIP: 0.87, 0.75, 0.73, 0.75 (overnight UTC lull — hour 1 is 00:00–01:00 UTC, only NZ/AU awake + EU pre-midnight hype)
- Hours 6–12 RISE: 0.79, 0.80, 0.90, 0.88, 0.91, 0.93, 0.98
- Hours 13–24 PEAK: 1.06, 1.18, 1.52, 1.65, 1.77, 1.95, 2.40, 2.37, 2.33, 2.30, 2.18, 2.04 (US prime time; unlocks: NZ 11:00Z 18 Nov, EU 23:00Z, UK 00:00Z, US East 05:00Z, US West 08:00Z)
- Dispersion explodes late: IQR of log-ratio 0.25 (h2) → ~2.1 (h19–22). Quantiles must widen a lot late-day; a narrow band pays heavily.
- Hour-1 prints across comparables: min 164, median 53,526, max 875,155 — scale varies 5000×, so model in log-ratios, never absolute viewers.
- Implication: the starter (persistence, sigma 0.15) badly under-forecasts the evening peak. The bar to beat on practice is ~0.169; a comparable-shape median-ratio forecaster with per-hour empirical spread is the obvious first entry, then refine with similarity weighting (hour-1 magnitude, channels, pre-release GTA VI history) and per-hour OLS on log(hour-1) like the #1 title suggests.
- Practice hour-1 prints: p00 Wolverine 23,357; p01 RDR2 116,550; p02 GoW 269,517; p03 Spider-Man 2 119,087.

## Build plan for delivery (main agent)
1. Fresh build session (≥06:00 UTC 17 Nov — satisfied), kit init first (thinking_fresh anchor), then write `main.py` in-session.
2. Entry v1: median log-ratio curve + per-hour empirical quantile spread from comparables (37 launches), clipped sane, deterministic. Submit early to get a real practice loss (server-only grading) — do NOT burn all 10/day on drafts.
3. Iterate with Hermes as partner: similarity-weighted comparables, per-hour OLS on log(hour-1) + log(channels), calibration of spread to ~0.90 coverage. Keep each version's practice loss; submit only improvements.
4. Deadline Thu 2026-11-19 08:00 WIB. Also note the in-flight builds: tiny-chess-01 (deadline 25 Nov) and token-radar-01 (14 Dec) are in progress — GTA's deadline is sooner, so it jumps the queue; github-core-clone-01 was already resubmitted today.
5. Submit via `node cli/makefun.mjs submit --competition gta6-twitch-ccu-forecast-01 --repo-url https://github.com/rhocss21/<repo> --commit <sha> --title ... --bundle transcripts.tar.gz --keypair wallet.json` (per RULES.txt "Enter" section); repo must be created after the competition opened.

## Sources
- `GET https://make.fun/api/competitions?status=open` (~15:20 WIB): 11 open, only this slug new vs known-competitions.json.
- `GET https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01` (~15:20 WIB): full detail incl. rewards, escrow, rules.
- `GET https://make.fun/api/competitions/gta6-twitch-ccu-forecast-01/leaderboard` (~15:22 WIB): 33 ranked rows.
- Kit tarball `https://make.fun/c/gta6-twitch-ccu-forecast-01/kit` (6,008,010 bytes, sha bundle `a8c440f8…` per API).
