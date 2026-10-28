#!/usr/bin/env python3
"""
gex_levels.py - compute dealer gamma exposure (GEX) levels for the Nasdaq-100
from the free CBOE delayed options chain and write them to a small flat JSON
file that the MT5 EA (GEX_Levels_EA.mq5) reads through WebRequest.

Output keys (all numbers are in NDX index points):
  spot, spot_epoch, net_gex ($ per 1% move), regime, gamma_flip, call_wall,
  put_wall, mag1..mag5 (+ mag1_gex..), updated_epoch, updated_utc

GEX per option = sign * gamma * open_interest * 100 * spot^2 * 0.01
  sign: +1 calls, -1 puts (standard "dealers long calls / short puts" assumption)

Usage:
  python3 gex_levels.py --symbol NDX --out levels.json
  python3 gex_levels.py --file chain.json --out levels.json     # offline test
  python3 gex_levels.py --selftest                              # synthetic test
"""
import argparse
import json
import math
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timezone

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = None

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
OCC = re.compile(r"^(.*?)(\d{6})([CP])(\d{8})$")
INDEX_SYMBOLS = {"NDX", "SPX", "RUT", "VIX", "XSP"}
MULT = 100.0
RATE = 0.04
T_FLOOR = 0.5 / (24 * 365.25)  # 30 minutes, avoids infinite gamma at expiry


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def num(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def fetch(symbol, retries=3, timeout=30):
    sym = "_" + symbol if symbol.upper() in INDEX_SYMBOLS else symbol
    url = CBOE_URL.format(sym=sym)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.cboe.com/",
    }
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def parse_time_utc(s):
    """CBOE timestamps are naive US/Eastern strings. Return UTC epoch or 0."""
    if not s:
        return 0
    s = str(s).strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            dt = datetime.strptime(s, fmt)
            if NY is not None:
                dt = dt.replace(tzinfo=NY)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.astimezone(timezone.utc).timestamp())
        except ValueError:
            continue
    return 0


def bs_gamma(S, K, T, iv, r=RATE):
    if S <= 0 or K <= 0 or T <= 0 or iv <= 0:
        return 0.0
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * iv * iv) * T) / sq
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return pdf / (S * sq)


def parse_chain(raw):
    data = raw.get("data", raw)
    spot = (num(data.get("current_price")) or num(data.get("close"))
            or num(data.get("last_trade_price")))
    spot_epoch = parse_time_utc(data.get("last_trade_time")) or \
        parse_time_utc(raw.get("timestamp"))
    opts = []
    for o in data.get("options", []):
        m = OCC.match(str(o.get("option", "")))
        if not m:
            continue
        _root, ymd, cp, k = m.groups()
        try:
            exp = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        except ValueError:
            continue
        opts.append({
            "exp": exp,
            "cp": cp,
            "K": int(k) / 1000.0,
            "oi": num(o.get("open_interest")),
            "iv": num(o.get("iv")),
            "gamma": num(o.get("gamma")),
        })
    return spot, spot_epoch, opts


def expiry_dt(exp):
    if NY is not None:
        return datetime(exp.year, exp.month, exp.day, 16, 0, tzinfo=NY).astimezone(timezone.utc)
    return datetime(exp.year, exp.month, exp.day, 20, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# core calculation
# --------------------------------------------------------------------------
def compute(spot, opts, max_dte=30.0, window=0.12, now=None):
    now = now or datetime.now(timezone.utc)
    use = []
    for o in opts:
        if o["oi"] <= 0:
            continue
        if abs(o["K"] / spot - 1.0) > window:
            continue
        secs = (expiry_dt(o["exp"]) - now).total_seconds()
        if secs <= 0 or secs / 86400.0 > max_dte:
            continue
        o = dict(o)
        o["T"] = max(secs / (365.25 * 86400.0), T_FLOOR)
        use.append(o)
    if not use:
        raise RuntimeError("no usable options after filters")

    # per-strike GEX at the current spot
    calls, puts = {}, {}
    for o in use:
        g = o["gamma"] if o["gamma"] > 0 else bs_gamma(spot, o["K"], o["T"], o["iv"])
        gex = g * o["oi"] * MULT * spot * spot * 0.01
        if o["cp"] == "C":
            calls[o["K"]] = calls.get(o["K"], 0.0) + gex
        else:
            puts[o["K"]] = puts.get(o["K"], 0.0) - gex
    strikes = sorted(set(calls) | set(puts))
    net_by_strike = {k: calls.get(k, 0.0) + puts.get(k, 0.0) for k in strikes}
    net_gex = sum(net_by_strike.values())

    # walls
    above = [k for k in calls if k >= spot] or list(calls)
    below = [k for k in puts if k <= spot] or list(puts)
    call_wall = max(above, key=lambda k: calls[k]) if above else 0.0
    put_wall = min(below, key=lambda k: puts[k]) if below else 0.0  # most negative

    # gamma flip: re-price gamma on a spot grid and find the zero crossing
    flip = 0.0
    grid_pts = 49
    lo, hi = spot * 0.94, spot * 1.06
    grid = [lo + (hi - lo) * i / (grid_pts - 1) for i in range(grid_pts)]
    vals = []
    for S in grid:
        tot = 0.0
        for o in use:
            if o["iv"] <= 0:
                continue
            g = bs_gamma(S, o["K"], o["T"], o["iv"])
            sgn = 1.0 if o["cp"] == "C" else -1.0
            tot += sgn * g * o["oi"] * MULT * S * S * 0.01
        vals.append(tot)
    crossings = []
    for i in range(1, len(grid)):
        a, b = vals[i - 1], vals[i]
        if a == 0:
            crossings.append(grid[i - 1])
        elif a * b < 0:
            crossings.append(grid[i - 1] + (grid[i] - grid[i - 1]) * (-a) / (b - a))
    if crossings:
        flip = min(crossings, key=lambda x: abs(x - spot))
    else:  # fallback: cumulative strike profile
        cum, prev = 0.0, None
        for k in strikes:
            cum += net_by_strike[k]
            if prev is not None and prev * cum < 0:
                flip = k
                break
            prev = cum

    # magnets: biggest |net GEX| strikes within +-5% of spot
    near = [k for k in strikes if abs(k / spot - 1.0) <= 0.05]
    mags = sorted(near, key=lambda k: abs(net_by_strike[k]), reverse=True)[:5]

    out = {
        "net_gex": round(net_gex, 2),
        "regime": "positive" if net_gex >= 0 else "negative",
        "gamma_flip": round(flip, 2),
        "call_wall": round(call_wall, 2),
        "put_wall": round(put_wall, 2),
        "options_used": len(use),
    }
    for i in range(5):
        k = mags[i] if i < len(mags) else 0.0
        out[f"mag{i + 1}"] = round(k, 2)
        out[f"mag{i + 1}_gex"] = round(net_by_strike.get(k, 0.0), 2)
    return out


# --------------------------------------------------------------------------
# self test with a synthetic chain
# --------------------------------------------------------------------------
def synthetic_chain(spot=25000.0):
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    options = []
    for dte in (1, 3, 8, 22):
        d = (now + timedelta(days=dte)).date()
        ymd = d.strftime("%y%m%d")
        for step in range(-40, 41):
            K = round(spot / 25) * 25 + step * 25
            dist = (K - spot) / spot
            iv = 0.18 + 0.6 * dist * dist - 0.3 * dist
            T = dte / 365.25
            g = bs_gamma(spot, K, T, iv)
            for cp in ("C", "P"):
                # heavy call OI above spot, heavy put OI below spot
                base = 4000.0 * math.exp(-abs(dist) * 25)
                oi = base * (2.2 if (cp == "C" and K > spot) else 1.0) * \
                    (2.2 if (cp == "P" and K < spot) else 1.0)
                options.append({
                    "option": f"NDX{ymd}{cp}{int(K * 1000):08d}",
                    "iv": iv, "gamma": g, "open_interest": oi,
                })
    return {"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "data": {"current_price": spot, "options": options,
                     "last_trade_time": now.strftime("%Y-%m-%dT%H:%M:%S")}}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NDX")
    ap.add_argument("--out", default="levels.json")
    ap.add_argument("--max-dte", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=0.12,
                    help="strike window around spot (0.12 = +-12%%)")
    ap.add_argument("--file", help="read a saved CBOE chain JSON instead of downloading")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raw = synthetic_chain()
    elif a.file:
        with open(a.file, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = fetch(a.symbol)

    spot, spot_epoch, opts = parse_chain(raw)
    if spot <= 0 or not opts:
        raise SystemExit("ERROR: no spot price or options in the data "
                         "(CBOE format may have changed)")
    res = compute(spot, opts, a.max_dte, a.window)

    now = datetime.now(timezone.utc)
    out = {
        "symbol": a.symbol.upper(),
        "spot": round(spot, 2),
        "spot_epoch": int(spot_epoch),
        "updated_epoch": int(now.timestamp()),
        "updated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.update(res)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
