"""Leave-one-launch-out evaluator for gta6-twitch-ccu-forecast-01 candidates.

Local evaluation only: the practice answers are sealed on the server, so we
evaluate on the 37 usable comparable launches shipped in the kit's practice
cases (all four practice cases carry the same 37). Each launch is held out in
turn; the model trains on the other 36 (all preprocessing from the 36 only),
sees only the held-out launch's hour-1 print (+ hour-1 channels and its
pre-release 7-day mean, where the candidate uses them), and is scored with
the EXACT pinball formula from RULES.txt on log scale.

Run: python3 candidates/evaluate.py  (from the repo root)
Writes: candidates/results.csv
"""
import csv
import json
import math
import os
import random
import statistics
from datetime import datetime

KIT_INPUT = os.path.expanduser(
    "~/workspace/tools/makefun/gta6-twitch-ccu-forecast-01-kit/"
    "gta6-twitch-ccu-forecast-01-kit/input/")
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))


def parse_ts(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def release_hours(comp):
    """Map release-day hour (1..24) -> (viewers, channels) or None if incomplete."""
    ds = parse_ts(comp["day_start"])
    out = {}
    for row in comp["hourly"]:
        t = parse_ts(row["from"])
        k = (t - ds).total_seconds() / 3600.0
        if k == int(k) and 0 <= int(k) < 24:
            out[int(k) + 1] = (row["viewers"], row.get("channels"))
    return out if len(out) == 24 else None


def pre7_mean(comp):
    """Mean viewers over the 168 hours before day_start (log-scale feature base)."""
    ds = parse_ts(comp["day_start"])
    vals = [r["viewers"] for r in comp["hourly"]
            if -168 <= (parse_ts(r["from"]) - ds).total_seconds() / 3600 < 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_pool(case_path):
    """Usable launches from one practice case: complete 24h release day,
    positive hour-1 print. Excludes the GTA VI pre-release comparable
    (no release day) automatically via the completeness check."""
    case = json.load(open(case_path))
    pool = []
    for comp in case["comparables"]:
        rh = release_hours(comp)
        if rh is None:
            continue
        v1, ch1 = rh[1]
        if v1 is None or v1 <= 0:
            continue
        y = {h: rh[h][0] for h in HOURS}
        if any(v is None or v <= 0 for v in y.values()):
            continue
        pool.append({
            "name": comp["name"],
            "v1": float(v1),
            "ch1": float(ch1) if ch1 is not None else None,
            "pre7": pre7_mean(comp),
            "y": y,
        })
    return pool


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(1e-9, q))
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def wquantile(pairs, tau):
    """Weighted empirical quantile, linear interpolation on midpoint
    plotting positions (Hazen): the i-th order statistic sits at
    (cum_weight_before + w_i / 2) / total_weight. Reduces to the
    standard type-7 quantile for uniform weights up to the half-step
    plotting convention, and is unbiased for the median."""
    s = sorted(pairs)
    tot = sum(w for _, w in s)
    if tot <= 0:
        return s[len(s) // 2][0]
    mids = []
    cum = 0.0
    for v, w in s:
        mids.append((cum + w / 2.0) / tot)
        cum += w
    if tau <= mids[0]:
        return s[0][0]
    if tau >= mids[-1]:
        return s[-1][0]
    for i in range(len(s) - 1):
        if mids[i] <= tau <= mids[i + 1]:
            frac = (tau - mids[i]) / (mids[i + 1] - mids[i])
            return s[i][0] + frac * (s[i + 1][0] - s[i][0])
    return s[-1][0]


def norm_ppf(tau):
    """Acklam's approximation to the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def shrink_quantiles(qs, lam):
    """Blend empirical log-quantiles toward a normal with the empirical
    median and IQR-based sigma. qs: list of 5 log values in tau order."""
    if lam <= 0:
        return list(qs)
    med = qs[2]
    s = max(1e-6, (qs[3] - qs[1]) / 1.34898)
    out = []
    for tau, q in zip(TAUS, qs):
        target = med + s * norm_ppf(tau)
        out.append((1 - lam) * q + lam * target)
    # keep monotone
    for i in range(1, 5):
        out[i] = max(out[i], out[i - 1])
    return out


def smooth_hours(logq):
    """Centered 3-hour moving average of per-hour log-quantiles.
    logq: dict h -> [5 log values]."""
    out = {}
    for h in HOURS:
        neigh = [hh for hh in (h - 1, h, h + 1) if hh in logq]
        out[h] = [sum(logq[hh][i] for hh in neigh) / len(neigh) for i in range(5)]
    return out

# ---------------------------------------------------------------------------
# Forecasters. Every forecaster takes (train, target, cfg) and returns
# {h: [q05, q25, q50, q75, q95]} in VIEWER units. train is a list of launches,
# target is a dict with v1 / ch1 / pre7. All fitting uses train only.
# ---------------------------------------------------------------------------

def weights_for(train, target, cfg):
    """Per-launch weights for the quantile method."""
    mode = cfg.get("weight", "uniform")
    lv = math.log(target["v1"])
    ws = []
    for tr in train:
        d = math.log(tr["v1"]) - lv
        if mode == "uniform":
            w = 1.0
        elif mode.startswith("kern"):
            b = float(mode[4:])
            w = math.exp(-(d * d) / (2 * b * b))
        elif mode.startswith("knn"):
            w = 1.0  # selection handled below
        else:
            raise ValueError(mode)
        ws.append(w)
    if mode.startswith("knn"):
        k = int(mode[3:])
        order = sorted(range(len(train)), key=lambda i: abs(math.log(train[i]["v1"]) - lv))
        keep = set(order[:k])
        ws = [1.0 if i in keep else 0.0 for i in range(len(train))]
    return ws


def eff_sample_size(ws):
    s = sum(ws)
    return (s * s) / sum(w * w for w in ws) if s > 0 else 0.0


def forecast_quantile(train, target, cfg):
    """Per-hour (weighted) empirical quantiles of log-ratios ln(Y_h / V1)."""
    ws = weights_for(train, target, cfg)
    lam = cfg.get("lam", 0.0)
    spread = cfg.get("spread", 1.0)
    logq = {}
    for h in HOURS:
        pairs = [(math.log(tr["y"][h] / tr["v1"]), w)
                 for tr, w in zip(train, ws) if w > 0]
        qs = [wquantile(pairs, tau) for tau in TAUS]
        qs = shrink_quantiles(qs, lam)
        if spread != 1.0:
            c = qs[2]
            qs = [c + (q - c) * spread for q in qs]
        logq[h] = qs
    if cfg.get("smooth"):
        logq = smooth_hours(logq)
    lv = math.log(target["v1"])
    return {h: [math.exp(lv + q) for q in logq[h]] for h in HOURS}


def _ols1(xs, ys, ws=None, shrink=1.0):
    """Weighted OLS y ~ a + b*(x - mean_x); b multiplied by `shrink` (ridge
    toward a constant level). Returns (a_at_mean_x, b, residuals)."""
    n = len(xs)
    ws = ws or [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = (sxy / sxx) * shrink if sxx > 0 else 0.0
    a = my  # value at x = mx
    res = [y - (a + b * (x - mx)) for x, y in zip(xs, ys)]
    return a, b, mx, res


def forecast_ols(train, target, cfg):
    """Per-hour OLS of ln Y_h on ln V1 (optionally + residualized ln channels
    and/or ln pre7), centre = fitted value at target, spread = residual
    quantiles (optionally normal-shrunk)."""
    lam = cfg.get("lam", 0.0)
    shrink = cfg.get("shrink", 1.0)
    use_ch = cfg.get("use_ch", False)
    use_pre7 = cfg.get("use_pre7", False)
    big_only = cfg.get("big_only", 0)
    sub = [tr for tr in train if tr["v1"] >= big_only] if big_only else train
    if len(sub) < 12:
        sub = train
    lv = math.log(target["v1"])
    lvs = [math.log(tr["v1"]) for tr in sub]
    # residualized ln channels: regress ln ch on ln v inside train subset
    ch_resid, ch_target = None, None
    if use_ch:
        pairs = [(math.log(tr["v1"]), math.log(tr["ch1"])) for tr in sub if tr["ch1"]]
        if len(pairs) >= 10 and target.get("ch1"):
            a_c, b_c, mx_c, res_c = _ols1([p[0] for p in pairs], [p[1] for p in pairs])
            by_name = {tr["name"]: r for tr, r in zip([t for t in sub if t["ch1"]], res_c)}
            ch_resid = [by_name.get(tr["name"], 0.0) for tr in sub]
            ch_target = math.log(target["ch1"]) - (a_c + b_c * (lv - mx_c))
        else:
            use_ch = False
    pre_target = None
    if use_pre7:
        if target.get("pre7") and all(tr["pre7"] for tr in sub):
            pre_target = math.log(target["pre7"])
        else:
            use_pre7 = False
    logq = {}
    for h in HOURS:
        ys = [math.log(tr["y"][h]) for tr in sub]
        if not use_ch and not use_pre7:
            a, b, mx, res = _ols1(lvs, ys, shrink=shrink)
            centre = a + b * (lv - mx)
        else:
            # two/three-feature OLS via normal equations on centred features
            feats = [[math.log(tr["v1"])] for tr in sub]
            if use_ch:
                for row, r in zip(feats, ch_resid):
                    row.append(r)
            if use_pre7:
                for row, tr in zip(feats, sub):
                    row.append(math.log(tr["pre7"]))
            k = len(feats[0])
            mxv = [sum(row[j] for row in feats) / len(feats) for j in range(k)]
            X = [[row[j] - mxv[j] for j in range(k)] for row in feats]
            my = sum(ys) / len(ys)
            yc = [y - my for y in ys]
            A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)] for a in range(k)]
            v = [sum(X[i][a] * yc[i] for i in range(len(X))) for a in range(k)]
            # ridge: shrink all slopes toward 0 by factor `shrink`
            beta = _solve(A, v)
            beta = [bb * shrink for bb in beta]
            tgt = [lv - mxv[0]]
            if use_ch:
                tgt.append(ch_target)
            if use_pre7:
                tgt.append(pre_target - mxv[-1])
            centre = my + sum(bb * tt for bb, tt in zip(beta, tgt))
            res = [yc[i] - sum(beta[j] * X[i][j] for j in range(k)) for i in range(len(yc))]
        qs = [wquantile([(r, 1.0) for r in res], tau) for tau in TAUS]
        qs = [centre + q for q in qs]
        qs = shrink_quantiles(qs, lam)
        logq[h] = qs
    if cfg.get("smooth"):
        logq = smooth_hours(logq)
    return {h: [math.exp(q) for q in logq[h]] for h in HOURS}


def forecast_lols(train, target, cfg):
    """Local-linear version of the OLS forecaster: per hour, kernel-weighted
    OLS of ln Y_h on ln V1 (bandwidth b), centre = fitted value at the
    target's ln V1, spread = kernel-weighted quantiles of the residuals.
    Adapts the slope to the target's neighbourhood instead of fitting one
    global slope dominated by the tiny-hour-1 launches."""
    lam = cfg.get("lam", 0.0)
    b = cfg.get("bandwidth", 1.0)
    spread = cfg.get("spread", 1.0)
    lv = math.log(target["v1"])
    ws = [math.exp(-((math.log(tr["v1"]) - lv) ** 2) / (2 * b * b)) for tr in train]
    if eff_sample_size(ws) < 8:
        b = 2.0
        ws = [math.exp(-((math.log(tr["v1"]) - lv) ** 2) / (2 * b * b)) for tr in train]
    lvs = [math.log(tr["v1"]) for tr in train]
    logq = {}
    for h in HOURS:
        ys = [math.log(tr["y"][h]) for tr in train]
        a, slope, mx, res = _ols1(lvs, ys, ws=ws, shrink=1.0)
        centre = a + slope * (lv - mx)
        qs = [wquantile(list(zip(res, ws)), tau) for tau in TAUS]
        qs = [centre + q for q in qs]
        qs = shrink_quantiles(qs, lam)
        if spread != 1.0:
            c = qs[2]
            qs = [c + (q - c) * spread for q in qs]
        logq[h] = qs
    if cfg.get("smooth"):
        logq = smooth_hours(logq)
    return {h: [math.exp(q) for q in logq[h]] for h in HOURS}


def forecast_ens(train, target, cfg):
    """Average of member sub-forecasters' log-quantiles (simple ensemble)."""
    import math as _m
    members = cfg.get("members") or [cfg["a"], cfg["b"]]
    preds = [forecast(m, train, target) for m in members]
    spread = cfg.get("spread", 1.0)
    out = {}
    for h in HOURS:
        qs = [_m.exp(sum(_m.log(p[h][i]) for p in preds) / len(preds)) for i in range(5)]
        if spread != 1.0:
            lqs = [_m.log(q) for q in qs]
            c = lqs[2]
            lqs = [c + (q - c) * spread for q in lqs]
            qs = [_m.exp(q) for q in lqs]
        out[h] = qs
    return out


def _solve(A, v):
    k = len(v)
    M = [row[:] + [v[i]] for i, row in enumerate(A)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(k):
            if r != col:
                f = M[r][col] / pv
                for cc in range(col, k + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]


def forecast_persist(train, target, cfg):
    """House baseline reproduction: q = p1 * exp(z * 0.15 * sqrt(h-1))."""
    p1 = max(1.0, target["v1"])
    out = {}
    for h in HOURS:
        out[h] = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
    return out


FORECASTERS = {
    "persist": forecast_persist,
    "quantile": forecast_quantile,
    "ols": forecast_ols,
    "lols": forecast_lols,
    "ens": forecast_ens,
}


def forecast(cfg, train, target):
    return FORECASTERS[cfg["method"]](train, target, cfg)

# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def score_launch(pred, truth_y):
    """pred: {h: [5 qs]}. Returns dict of metrics for one launch."""
    items = []
    per_tau = {tau: [] for tau in TAUS}
    per_band = {"early": [], "mid": [], "late": []}
    covered = 0
    n_hours = 0
    med_abs = []
    for h in HOURS:
        y = truth_y[h]
        n_hours += 1
        if pred[h][0] <= y <= pred[h][4]:
            covered += 1
        med_abs.append(abs(math.log(y) - math.log(pred[h][2])))
        band = "early" if h <= 8 else ("mid" if h <= 15 else "late")
        for tau, q in zip(TAUS, pred[h]):
            loss = pinball(y, q, tau)
            items.append(loss)
            per_tau[tau].append(loss)
            per_band[band].append(loss)
    return {
        "mean": sum(items) / len(items),
        "per_tau": {tau: sum(v) / len(v) for tau, v in per_tau.items()},
        "per_band": {b: sum(v) / len(v) for b, v in per_band.items()},
        "coverage": covered / n_hours,
        "typ_miss": math.exp(sum(med_abs) / len(med_abs)) - 1,
    }


def loo(pool, cfg):
    """Leave-one-launch-out. Returns list of (name, v1, metrics)."""
    out = []
    for i, held in enumerate(pool):
        train = pool[:i] + pool[i + 1:]
        target = {"v1": held["v1"], "ch1": held["ch1"], "pre7": held["pre7"]}
        pred = forecast(cfg, train, target)
        out.append((held["name"], held["v1"], score_launch(pred, held["y"])))
    return out


def summarize(rows):
    losses = [m["mean"] for _, _, m in rows]
    big = [m["mean"] for _, v1, m in rows if v1 >= 50000]
    band = [m["mean"] for _, v1, m in rows if 10000 <= v1 <= 300000]
    small = [m["mean"] for _, v1, m in rows if v1 < 10000]
    return {
        "mean": sum(losses) / len(losses),
        "median": statistics.median(losses),
        "worst5": sum(sorted(losses, reverse=True)[:5]) / 5,
        "coverage": sum(m["coverage"] for _, _, m in rows) / len(rows),
        "typ_miss": sum(m["typ_miss"] for _, _, m in rows) / len(rows),
        "early": sum(m["per_band"]["early"] for _, _, m in rows) / len(rows),
        "mid": sum(m["per_band"]["mid"] for _, _, m in rows) / len(rows),
        "late": sum(m["per_band"]["late"] for _, _, m in rows) / len(rows),
        "tau05": sum(m["per_tau"][0.05] for _, _, m in rows) / len(rows),
        "tau25": sum(m["per_tau"][0.25] for _, _, m in rows) / len(rows),
        "tau50": sum(m["per_tau"][0.5] for _, _, m in rows) / len(rows),
        "tau75": sum(m["per_tau"][0.75] for _, _, m in rows) / len(rows),
        "tau95": sum(m["per_tau"][0.95] for _, _, m in rows) / len(rows),
        "big_mean": sum(big) / len(big) if big else float("nan"),
        "n_big": len(big),
        "band_mean": sum(band) / len(band) if band else float("nan"),
        "n_band": len(band),
        "small_mean": sum(small) / len(small) if small else float("nan"),
        "n_small": len(small),
    }


def bootstrap_paired(rows_a, rows_b, reps=2000, seed=7):
    """Bootstrap WHOLE launches: CI for (mean loss A - mean loss B)."""
    rng = random.Random(seed)
    n = len(rows_a)
    la = [m["mean"] for _, _, m in rows_a]
    lb = [m["mean"] for _, _, m in rows_b]
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(la[i] for i in idx) / n - sum(lb[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def nested_loo(pool, grid):
    """Outer LOO; inside each outer training set, pick the best config by
    inner LOO mean, then score the pick on the outer held-out launch.
    Returns (rows, picks)."""
    rows = []
    picks = {}
    for i, held in enumerate(pool):
        train = pool[:i] + pool[i + 1:]
        best_cfg, best_loss = None, float("inf")
        for cfg in grid:
            inner = loo(train, cfg)
            m = sum(x[2]["mean"] for x in inner) / len(inner)
            if m < best_loss:
                best_loss, best_cfg = m, cfg["name"]
        picks[best_cfg] = picks.get(best_cfg, 0) + 1
        target = {"v1": held["v1"], "ch1": held["ch1"], "pre7": held["pre7"]}
        cfg = next(c for c in grid if c["name"] == best_cfg)
        pred = forecast(cfg, train, target)
        rows.append((held["name"], held["v1"], score_launch(pred, held["y"])))
    return rows, picks


CANDIDATES = [
    {"name": "v0-persist", "method": "persist"},
    {"name": "v1-emp-lam0", "method": "quantile", "weight": "uniform", "lam": 0.0},
    {"name": "v1-emp-lam025", "method": "quantile", "weight": "uniform", "lam": 0.25},
    {"name": "v1-emp-lam05", "method": "quantile", "weight": "uniform", "lam": 0.5},
    {"name": "v1-emp-lam025-smooth", "method": "quantile", "weight": "uniform", "lam": 0.25, "smooth": True},
    {"name": "v2-kern-b05-lam025", "method": "quantile", "weight": "kern0.5", "lam": 0.25},
    {"name": "v2-kern-b07-lam025", "method": "quantile", "weight": "kern0.7", "lam": 0.25},
    {"name": "v2-kern-b1-lam0", "method": "quantile", "weight": "kern1", "lam": 0.0},
    {"name": "v2-kern-b1-lam025", "method": "quantile", "weight": "kern1", "lam": 0.25},
    {"name": "v2-kern-b1-lam05", "method": "quantile", "weight": "kern1", "lam": 0.5},
    {"name": "v2-kern-b15-lam025", "method": "quantile", "weight": "kern1.5", "lam": 0.25},
    {"name": "v2-kern-b2-lam0", "method": "quantile", "weight": "kern2", "lam": 0.0},
    {"name": "v2-kern-b2-lam025", "method": "quantile", "weight": "kern2", "lam": 0.25},
    {"name": "v2-kern-b1-lam025-smooth", "method": "quantile", "weight": "kern1", "lam": 0.25, "smooth": True},
    {"name": "v2-knn08-lam025", "method": "quantile", "weight": "knn8", "lam": 0.25},
    {"name": "v2-knn10-lam025", "method": "quantile", "weight": "knn10", "lam": 0.25},
    {"name": "v2-knn12-lam025", "method": "quantile", "weight": "knn12", "lam": 0.25},
    {"name": "v2-knn18-lam025", "method": "quantile", "weight": "knn18", "lam": 0.25},
    {"name": "v2-knn24-lam025", "method": "quantile", "weight": "knn24", "lam": 0.25},
    {"name": "v3-ols-k100-lam0", "method": "ols", "shrink": 1.0, "lam": 0.0},
    {"name": "v3-ols-k100-lam025", "method": "ols", "shrink": 1.0, "lam": 0.25},
    {"name": "v3-ols-k075-lam025", "method": "ols", "shrink": 0.75, "lam": 0.25},
    {"name": "v3-ols-k050-lam025", "method": "ols", "shrink": 0.5, "lam": 0.25},
    {"name": "v3-ols-big30k", "method": "ols", "shrink": 1.0, "lam": 0.25, "big_only": 30000},
    {"name": "v3-ols-k100-lam025-smooth", "method": "ols", "shrink": 1.0, "lam": 0.25, "smooth": True},
    {"name": "v4-ols-ch", "method": "ols", "shrink": 1.0, "lam": 0.25, "use_ch": True},
    {"name": "v5-ols-pre7", "method": "ols", "shrink": 1.0, "lam": 0.25, "use_pre7": True},
    {"name": "v5-ols-ch-pre7", "method": "ols", "shrink": 1.0, "lam": 0.25, "use_ch": True, "use_pre7": True},
    {"name": "v6-lols-b1-lam025", "method": "lols", "bandwidth": 1.0, "lam": 0.25},
    {"name": "v6-lols-b1-lam0", "method": "lols", "bandwidth": 1.0, "lam": 0.0},
    {"name": "v6-lols-b2-lam025", "method": "lols", "bandwidth": 2.0, "lam": 0.25},
    {"name": "v6-lols-b15-lam025", "method": "lols", "bandwidth": 1.5, "lam": 0.25},
    {"name": "v6-lols-b1-lam025-smooth", "method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True},
    {"name": "v6-lols-b1-lam05-smooth", "method": "lols", "bandwidth": 1.0, "lam": 0.5, "smooth": True},
    {"name": "v6-lols-b07-lam025-smooth", "method": "lols", "bandwidth": 0.7, "lam": 0.25, "smooth": True},
    {"name": "v6-lols-b1-lam025-spread115", "method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True, "spread": 1.15},
    {"name": "v6-lols-b1-lam025-spread130", "method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True, "spread": 1.30},
    {"name": "v6-lols-b1-lam0-spread115", "method": "lols", "bandwidth": 1.0, "lam": 0.0, "smooth": True, "spread": 1.15},
    {"name": "v7-ens-lols-kern", "method": "ens",
     "members": [{"method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True},
                 {"method": "quantile", "weight": "kern1", "lam": 0.5}]},
    {"name": "v7-ens-lam05-kern", "method": "ens",
     "members": [{"method": "lols", "bandwidth": 1.0, "lam": 0.5, "smooth": True},
                 {"method": "quantile", "weight": "kern1", "lam": 0.5}]},
    {"name": "v7-ens-lols-kern-spread115", "method": "ens", "spread": 1.15,
     "members": [{"method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True},
                 {"method": "quantile", "weight": "kern1", "lam": 0.5}]},
    {"name": "v7-ens3", "method": "ens",
     "members": [{"method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True},
                 {"method": "lols", "bandwidth": 2.0, "lam": 0.25},
                 {"method": "quantile", "weight": "kern1", "lam": 0.5}]},
    {"name": "v7-ens-lols-lols2", "method": "ens",
     "members": [{"method": "lols", "bandwidth": 1.0, "lam": 0.25, "smooth": True},
                 {"method": "lols", "bandwidth": 2.0, "lam": 0.25}]},
]


def main():
    pools = {}
    for pid in ["p00", "p01", "p02", "p03"]:
        pools[pid] = load_pool(KIT_INPUT + pid + ".in")
        print(f"{pid}: {len(pools[pid])} usable launches")
    names = sorted(p["name"] for p in pools["p00"])
    for pid in ["p01", "p02", "p03"]:
        assert sorted(p["name"] for p in pools[pid]) == names, f"pool mismatch {pid}"
    print("All 4 practice cases share the same 37 usable comparables: OK")
    pool = pools["p00"]

    results = {}
    for cfg in CANDIDATES:
        rows = loo(pool, cfg)
        results[cfg["name"]] = (rows, summarize(rows))
        s = results[cfg["name"]][1]
        print(f"{cfg['name']:28s} mean={s['mean']:.4f} med={s['median']:.4f} "
              f"worst5={s['worst5']:.4f} cov={s['coverage']:.3f} "
              f"big(V1>=50k,n={s['n_big']})={s['big_mean']:.4f} "
              f"band(10k-300k,n={s['n_band']})={s['band_mean']:.4f} "
              f"small(<10k,n={s['n_small']})={s['small_mean']:.4f}")

    # Nested selection over the non-baseline grid
    grid = [c for c in CANDIDATES if c["name"] != "v0-persist"]
    nrows, picks = nested_loo(pool, grid)
    nsum = summarize(nrows)
    print(f"nested-selection: mean={nsum['mean']:.4f} med={nsum['median']:.4f} "
          f"worst5={nsum['worst5']:.4f} cov={nsum['coverage']:.3f} "
          f"big={nsum['big_mean']:.4f} band={nsum['band_mean']:.4f} picks={picks}")
    results["nested-selection"] = (nrows, nsum)

    # Paired bootstrap diffs vs v0 and vs v1
    base_rows = results["v0-persist"][0]
    v1_rows = results["v1-emp-lam0"][0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
    fields = ["candidate", "mean", "median", "worst5", "coverage", "typ_miss",
              "early_h2_8", "mid_h9_15", "late_h16_24",
              "tau05", "tau25", "tau50", "tau75", "tau95",
              "big_mean_V1ge50k", "n_big", "band_mean_V1_10k_300k", "n_band",
              "small_mean_V1lt10k", "n_small",
              "paired_vs_v0", "paired_vs_v0_ci_lo", "paired_vs_v0_ci_hi",
              "paired_vs_v1", "paired_vs_v1_ci_lo", "paired_vs_v1_ci_hi"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cfg in CANDIDATES:
            name = cfg["name"]
            s = results[name][1]
            d0lo, d0hi = bootstrap_paired(results[name][0], base_rows)
            d1lo, d1hi = bootstrap_paired(results[name][0], v1_rows)
            w.writerow({
                "candidate": name, "mean": round(s["mean"], 4),
                "median": round(s["median"], 4), "worst5": round(s["worst5"], 4),
                "coverage": round(s["coverage"], 3), "typ_miss": round(s["typ_miss"], 3),
                "early_h2_8": round(s["early"], 4), "mid_h9_15": round(s["mid"], 4),
                "late_h16_24": round(s["late"], 4),
                "tau05": round(s["tau05"], 4), "tau25": round(s["tau25"], 4),
                "tau50": round(s["tau50"], 4), "tau75": round(s["tau75"], 4),
                "tau95": round(s["tau95"], 4),
                "big_mean_V1ge50k": round(s["big_mean"], 4), "n_big": s["n_big"],
                "band_mean_V1_10k_300k": round(s["band_mean"], 4), "n_band": s["n_band"],
                "small_mean_V1lt10k": round(s["small_mean"], 4), "n_small": s["n_small"],
                "paired_vs_v0": round(s["mean"] - results["v0-persist"][1]["mean"], 4),
                "paired_vs_v0_ci_lo": round(d0lo, 4), "paired_vs_v0_ci_hi": round(d0hi, 4),
                "paired_vs_v1": round(s["mean"] - results["v1-emp-lam0"][1]["mean"], 4),
                "paired_vs_v1_ci_lo": round(d1lo, 4), "paired_vs_v1_ci_hi": round(d1hi, 4),
            })
        s = nsum
        d0lo, d0hi = bootstrap_paired(nrows, base_rows)
        d1lo, d1hi = bootstrap_paired(nrows, v1_rows)
        w.writerow({
            "candidate": "nested-selection", "mean": round(s["mean"], 4),
            "median": round(s["median"], 4), "worst5": round(s["worst5"], 4),
            "coverage": round(s["coverage"], 3), "typ_miss": round(s["typ_miss"], 3),
            "early_h2_8": round(s["early"], 4), "mid_h9_15": round(s["mid"], 4),
            "late_h16_24": round(s["late"], 4),
            "tau05": round(s["tau05"], 4), "tau25": round(s["tau25"], 4),
            "tau50": round(s["tau50"], 4), "tau75": round(s["tau75"], 4),
            "tau95": round(s["tau95"], 4),
            "big_mean_V1ge50k": round(s["big_mean"], 4), "n_big": s["n_big"],
                "band_mean_V1_10k_300k": round(s["band_mean"], 4), "n_band": s["n_band"],
                "small_mean_V1lt10k": round(s["small_mean"], 4), "n_small": s["n_small"],
            "paired_vs_v0": round(s["mean"] - results["v0-persist"][1]["mean"], 4),
            "paired_vs_v0_ci_lo": round(d0lo, 4), "paired_vs_v0_ci_hi": round(d0hi, 4),
            "paired_vs_v1": round(s["mean"] - results["v1-emp-lam0"][1]["mean"], 4),
            "paired_vs_v1_ci_lo": round(d1lo, 4), "paired_vs_v1_ci_hi": round(d1hi, 4),
        })
    print("wrote", out_path)


if __name__ == "__main__":
    main()
