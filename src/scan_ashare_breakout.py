#!/usr/bin/env python3
"""
Whole-market A-share breakout scanner.

Data source strategy (important): Chinese market endpoints often refuse
connections from non-China IPs (e.g. GitHub's US-based runners). To be
resilient we try multiple akshare providers for each step:

  Snapshot (universe + today's move):  Eastmoney -> Sina fallback
  Daily history (breakout confirm):    Eastmoney -> Sina fallback

Two-stage design keeps it fast:
  Stage 1: one snapshot request -> cheap pre-filter (up today + in price range,
           and volume-ratio if the source provides it) -> few hundred candidates.
  Stage 2: pull daily history per candidate and confirm a genuine breakout
           (close > prior N-day high, volume confirmation, above MA, RSI ok).

Standalone; does NOT depend on Screeni-py internals.
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
    for n in names:
        if n in df.columns:
            return n
    return None


def _sina_symbol(code: str) -> str:
    code = str(code)
    return ("sh" if code[0] in ("6", "9") else "sz") + code


# ----------------------------- Stage 1: snapshot -----------------------------

def _normalize_snapshot(df):
    """Return DataFrame with columns: code, name, price, change_pct, volratio."""
    code_c = _col(df, "代码", "symbol", "code")
    name_c = _col(df, "名称", "name")
    price_c = _col(df, "最新价", "trade", "price")
    chg_c = _col(df, "涨跌幅", "changepercent", "pctChg")
    lb_c = _col(df, "量比")  # only Eastmoney provides this
    out = pd.DataFrame()
    out["code"] = df[code_c].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True)
    out["name"] = df[name_c].astype(str) if name_c else ""
    out["price"] = pd.to_numeric(df[price_c], errors="coerce") if price_c else float("nan")
    out["change_pct"] = pd.to_numeric(df[chg_c], errors="coerce") if chg_c else float("nan")
    out["volratio"] = pd.to_numeric(df[lb_c], errors="coerce") if lb_c else float("nan")
    return out


def get_snapshot():
    """Try Sina first (Eastmoney is often blocked from non-CN IPs), then Eastmoney.
    Returns (normalized_df, source_name)."""
    providers = [
        ("sina", lambda: ak.stock_zh_a_spot()),
        ("eastmoney", lambda: ak.stock_zh_a_spot_em()),
    ]
    last_err = None
    for name, fn in providers:
        try:
            print(f"[+] Trying snapshot source: {name} ...")
            df = fn()
            if df is not None and len(df) > 0:
                print(f"[+] Snapshot source OK: {name} ({len(df)} rows)")
                return _normalize_snapshot(df), name
        except Exception as e:
            last_err = e
            print(f"[!] Snapshot via {name} failed: {e}")
    raise RuntimeError(f"All snapshot providers failed. Last error: {last_err}")


def prefilter(df, min_price, max_price, min_change_pct, volume_ratio, exclude_st, top_n):
    df = df.copy()
    df = df[df["code"].str.match(r"^(60|68|00|30)")]
    if exclude_st:
        df = df[~df["name"].str.contains("ST|st|退", regex=True, na=False)]
    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
    df = df[df["change_pct"] >= min_change_pct]
    has_volratio = df["volratio"].notna().any()
    if has_volratio:
        df = df[df["volratio"].fillna(0) >= volume_ratio]
        df = df.sort_values("volratio", ascending=False)
    else:
        # Sina snapshot has no volume-ratio; rank by today's move instead.
        df = df.sort_values("change_pct", ascending=False)
    if top_n and len(df) > top_n:
        df = df.head(top_n)
    return df[["code", "name", "change_pct"]].reset_index(drop=True), has_volratio


# --------------------------- Stage 2: breakout check --------------------------

def _get_history(code, start, end):
    """Try Sina daily first, then Eastmoney hist. Returns a DataFrame or None."""
    try:
        h = ak.stock_zh_a_daily(symbol=_sina_symbol(code), start_date=start,
                                end_date=end, adjust="qfq")
        if h is not None and len(h) > 0:
            return h
    except Exception:
        pass
    try:
        h = ak.stock_zh_a_hist(symbol=code, period="daily",
                               start_date=start, end_date=end, adjust="qfq")
        if h is not None and len(h) > 0:
            return h
    except Exception:
        pass
    return None


def check_breakout(code, name, lookback, volume_ratio, max_rsi, history_days):
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=history_days + lookback + 40)).strftime("%Y%m%d")
    hist = _get_history(code, start, end)
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
    prior_high = high.iloc[-(lookback + 1):-1].max()
    if pd.isna(prior_high) or prior_high <= 0 or pd.isna(last_close):
        return None
    if last_close <= prior_high:
        return None

    vol_avg20 = vol.iloc[-21:-1].mean()
    vol_ratio_val = (vol.iloc[-1] / vol_avg20) if vol_avg20 and vol_avg20 > 0 else 0
    if vol_ratio_val < volume_ratio:
        return None

    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma20
    if pd.isna(ma20) or last_close < ma20:
        return None
    ma_signal = "Bullish" if (not pd.isna(ma60) and ma20 >= ma60) else "Above MA20"

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


def _save(df, path):
    try:
        if str(path).lower().endswith(".xlsx"):
            # openpyxl writes native Unicode; string codes keep their leading zeros.
            df.to_excel(path, index=False)
        else:
            # utf-8-sig adds a BOM so Excel (esp. Chinese locale) reads UTF-8
            # correctly instead of mangling Chinese names as GBK.
            df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[+] Results saved to {path}")
    except Exception as e:
        print(f"[!] Failed to save {path}: {e}")


def main():
    p = argparse.ArgumentParser(description="Whole-market A-share breakout scanner (akshare)")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--volume-ratio", type=float, default=1.5)
    p.add_argument("--min-price", type=float, default=3.0)
    p.add_argument("--max-price", type=float, default=3000.0)
    p.add_argument("--max-rsi", type=float, default=85.0)
    p.add_argument("--min-change-pct", type=float, default=2.0)
    p.add_argument("--exclude-st", type=str, default="y")
    p.add_argument("--top-prefilter", type=int, default=800)
    p.add_argument("--history-days", type=int, default=180)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", type=str, default="ashare-breakout.csv")
    args = p.parse_args()

    exclude_st = str(args.exclude_st).lower().startswith("y")

    print("[+] Stage 1: fetching full-market snapshot...")
    snap, source = get_snapshot()

    cand, had_volratio = prefilter(snap, args.min_price, args.max_price, args.min_change_pct,
                                   args.volume_ratio, exclude_st, args.top_prefilter)
    note = "" if had_volratio else " (source has no volume-ratio; stage-1 volume filter skipped)"
    print(f"[+] Stage 1 candidates (up>={args.min_change_pct}%, price[{args.min_price},{args.max_price}]): "
          f"{len(cand)}{note}")
    if len(cand) == 0:
        print("[!] No candidates from stage-1 pre-filter.")
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


if __name__ == "__main__":
    main()
