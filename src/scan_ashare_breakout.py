#!/usr/bin/env python3
"""
Whole-market A-share breakout scanner (data via akshare / Eastmoney).

Two-stage design to keep it fast and reliable on GitHub Actions:

  Stage 1 (1 request): ak.stock_zh_a_spot_em() returns a live snapshot of ALL
          A-shares. We cheaply pre-filter to "moving today with volume"
          candidates (up >= min-change-pct, volume-ratio >= volume-ratio,
          price in range), which shrinks ~5400 stocks to a few hundred.

  Stage 2 (per candidate): ak.stock_zh_a_hist() daily history is pulled and we
          confirm a genuine breakout: close > prior N-day high, with volume
          confirmation, price above MA, and (optionally) not overbought.

Output: a CSV/XLSX table of confirmed breakout stocks.

This module is standalone and does NOT depend on Screeni-py internals.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("[!] akshare is not installed. Run: pip install akshare")
    sys.exit(1)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (pure pandas, no TA-Lib needed)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _col(df, *names):
    """Return the first matching column name present in df, else None."""
    for n in names:
        if n in df.columns:
            return n
    return None


def get_snapshot(retries=3):
    """Stage 1: full-market realtime snapshot (all A-shares) in one request."""
    last_err = None
    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch market snapshot: {last_err}")


def prefilter(df, min_price, max_price, min_change_pct, volume_ratio, exclude_st, top_n):
    code_c = _col(df, "代码", "symbol", "code")
    name_c = _col(df, "名称", "name")
    price_c = _col(df, "最新价", "price")
    chg_c = _col(df, "涨跌幅", "changepercent")
    lb_c = _col(df, "量比")  # Eastmoney "volume ratio" (today vs recent avg)

    df = df.copy()
    # Keep only main A-share boards: Shanghai 60/68, Shenzhen 00/30
    df = df[df[code_c].astype(str).str.match(r"^(60|68|00|30)")]
    if exclude_st and name_c:
        df = df[~df[name_c].astype(str).str.contains("ST|st|退", regex=True, na=False)]
    # Numeric coercion
    for c in (price_c, chg_c, lb_c):
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df[price_c] >= min_price) & (df[price_c] <= max_price)]
    df = df[df[chg_c] >= min_change_pct]
    if lb_c:
        df = df[df[lb_c] >= volume_ratio]
        df = df.sort_values(lb_c, ascending=False)
    if top_n and len(df) > top_n:
        df = df.head(top_n)
    out = pd.DataFrame({
        "code": df[code_c].astype(str).values,
        "name": (df[name_c].astype(str).values if name_c else ""),
        "change_pct": (df[chg_c].values if chg_c else 0),
    })
    return out.reset_index(drop=True)


def check_breakout(code, name, lookback, volume_ratio, max_rsi, history_days, retries=2):
    """Stage 2: pull daily history and confirm a genuine breakout for one stock."""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=history_days + lookback + 40)).strftime("%Y%m%d")
    hist = None
    for attempt in range(retries):
        try:
            hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                      start_date=start, end_date=end, adjust="qfq")
            break
        except Exception:
            time.sleep(1 + attempt)
    if hist is None or len(hist) < max(lookback + 2, 25):
        return None

    close_c = _col(hist, "收盘", "close")
    high_c = _col(hist, "最高", "high")
    vol_c = _col(hist, "成交量", "volume")
    if not (close_c and high_c and vol_c):
        return None

    close = pd.to_numeric(hist[close_c], errors="coerce")
    high = pd.to_numeric(hist[high_c], errors="coerce")
    vol = pd.to_numeric(hist[vol_c], errors="coerce")

    last_close = close.iloc[-1]
    # Prior N-day high, EXCLUDING today's candle
    prior_high = high.iloc[-(lookback + 1):-1].max()
    if pd.isna(prior_high) or prior_high <= 0:
        return None
    is_breakout = last_close > prior_high
    if not is_breakout:
        return None

    # Volume confirmation: today's volume vs 20-day avg (excluding today)
    vol_avg20 = vol.iloc[-21:-1].mean()
    vol_ratio_val = (vol.iloc[-1] / vol_avg20) if vol_avg20 and vol_avg20 > 0 else 0
    if vol_ratio_val < volume_ratio:
        return None

    # Moving-average trend filter: close above MA20 and MA20 rising vs MA60
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
    if pd.isna(ma20) or last_close < ma20:
        return None
    ma_signal = "Bullish" if (not pd.isna(ma60) and ma20 >= ma60) else "Above MA20"

    # RSI overbought filter (optional; max_rsi <= 0 disables it)
    rsi_val = rsi(close, 14).iloc[-1]
    if max_rsi and max_rsi > 0 and (not pd.isna(rsi_val)) and rsi_val > max_rsi:
        return None

    pct_above = (last_close / prior_high - 1) * 100
    return {
        "Code": code,
        "Name": name,
        "Close": round(float(last_close), 2),
        "PriorHigh": round(float(prior_high), 2),
        "%AboveHigh": round(float(pct_above), 2),
        "VolRatio": round(float(vol_ratio_val), 2),
        "RSI": (round(float(rsi_val), 1) if not pd.isna(rsi_val) else None),
        "MA-Signal": ma_signal,
    }


def main():
    p = argparse.ArgumentParser(description="Whole-market A-share breakout scanner (akshare)")
    p.add_argument("--lookback", type=int, default=60, help="Breakout window: close must exceed prior N-day high (default 60)")
    p.add_argument("--volume-ratio", type=float, default=1.5, help="Today volume >= this x 20-day avg (default 1.5)")
    p.add_argument("--min-price", type=float, default=3.0, help="Min price CNY (default 3)")
    p.add_argument("--max-price", type=float, default=3000.0, help="Max price CNY (default 3000)")
    p.add_argument("--max-rsi", type=float, default=85.0, help="Exclude if RSI above this; 0 disables (default 85)")
    p.add_argument("--min-change-pct", type=float, default=2.0, help="Stage-1: today up at least this %% (default 2)")
    p.add_argument("--exclude-st", type=str, default="y", help="Exclude ST/*ST/delisting stocks (y/n, default y)")
    p.add_argument("--top-prefilter", type=int, default=800, help="Cap stage-1 candidates to bound runtime (default 800)")
    p.add_argument("--history-days", type=int, default=180, help="Days of daily history to pull per candidate (default 180)")
    p.add_argument("--workers", type=int, default=8, help="Parallel history-fetch threads (default 8)")
    p.add_argument("--output", type=str, default="ashare-breakout.csv", help="Output .csv or .xlsx path")
    args = p.parse_args()

    exclude_st = str(args.exclude_st).lower().startswith("y")

    print(f"[+] Stage 1: fetching full-market snapshot...")
    snap = get_snapshot()
    print(f"[+] Snapshot rows: {len(snap)}")

    cand = prefilter(snap, args.min_price, args.max_price, args.min_change_pct,
                     args.volume_ratio, exclude_st, args.top_prefilter)
    print(f"[+] Stage 1 candidates (up>={args.min_change_pct}%, volratio>={args.volume_ratio}, "
          f"price[{args.min_price},{args.max_price}]): {len(cand)}")
    if len(cand) == 0:
        print("[!] No candidates from stage-1 pre-filter. Nothing to confirm.")
        _save(pd.DataFrame(), args.output)
        return

    print(f"[+] Stage 2: confirming breakouts on {len(cand)} candidates "
          f"(lookback={args.lookback}, workers={args.workers})...")
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(check_breakout, r.code, r.name, args.lookback,
                          args.volume_ratio, args.max_rsi, args.history_days): r.code
                for r in cand.itertuples()}
        for fut in as_completed(futs):
            done += 1
            if done % 50 == 0:
                print(f"    ...checked {done}/{len(cand)}, found {len(results)}")
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception:
                pass

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values(["%AboveHigh", "VolRatio"], ascending=False).reset_index(drop=True)
    print(f"\n[+] Found {len(df)} breakout stocks.")
    if len(df) > 0:
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(df.to_string(index=False))
    _save(df, args.output)


def _save(df, path):
    try:
        if str(path).lower().endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)
        print(f"[+] Results saved to {path}")
    except Exception as e:
        print(f"[!] Failed to save {path}: {e}")


if __name__ == "__main__":
    main()
