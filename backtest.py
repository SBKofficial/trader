#!/usr/bin/env python3
"""
Weekly Supertrend regime-based ETF strategy checker.

Behavior:
 - Computes Supertrend(10, 2.5) on weekly candles for:
    * NIFTY master ticker (market regime)
    * Equity ETFs (only trade when NIFTY is green)
    * Gold & Silver (independent)
    * LiquidBees (used only when NIFTY is red)
 - Outputs actionable summary: which ETFs to BUY / SELL / HOLD,
   and whether to move to LIQUIDBEES (100% equity -> LIQUIDBEES) when NIFTY red.

Configure by environment variables or edit DEFAULT lists below.
Optionally posts JSON summary to webhook if WEBHOOK_URL env var is set.

Run in CI weekly (GitHub Actions). Keep your tickers correct for your market/data source.

Author: Generated for user
"""
import os
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------- CONFIGURATION (override with env vars) ----------
# Default tickers - replace with exact tickers you use on your data source.
DEFAULT_NIFTY_TICKER = os.getenv("NIFTY_TICKER", "^NSEI")  # example: '^NSEI' (TradingView/yfinance)
DEFAULT_EQUITY_ETFS = os.getenv("EQUITY_ETFS", "NIFTYBEES.NS,MOMENTUM.NS,MON100.NS,HDFCSML250.NS,MID150BEES.NS")
DEFAULT_GOLD_SILVER = os.getenv("GOLD_SILVER", "GOLDBEES.NS,SILVERBEES.NS")
DEFAULT_LIQUID = os.getenv("LIQUID_TICKER", "LIQUIDBEES.NS")
DATA_PERIOD = os.getenv("DATA_PERIOD", "5y")  # how far back for weekly data
SUPER_PERIOD = int(os.getenv("SUPER_PERIOD", "10"))
SUPER_MULT = float(os.getenv("SUPER_MULT", "2.5"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # optional, POSTs JSON summary to this URL
VERBOSE = os.getenv("VERBOSE", "1") == "1"

# parse lists
EQUITY_ETFS = [t.strip() for t in DEFAULT_EQUITY_ETFS.split(",") if t.strip()]
GOLD_SILVER = [t.strip() for t in DEFAULT_GOLD_SILVER.split(",") if t.strip()]

# ---------- UTILITIES ----------
def fetch_weekly(ticker: str, period=DATA_PERIOD):
    """Fetch weekly OHLCV data for ticker using yfinance."""
    df = yf.download(tickers=ticker, period=period, interval="1wk", progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for ticker: {ticker}")
    # Ensure proper column names
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range / ATR calculation."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=1).mean()
    return atr

def supertrend(df: pd.DataFrame, period: int = SUPER_PERIOD, multiplier: float = SUPER_MULT):
    """
    Compute Supertrend. Returns df with columns:
      - 'ST' : True if trend is bullish (green), False if bearish (red)
      - 'ST_value' : numeric band value
    Algorithm based on ATR bands (common implementation).
    """
    df = df.copy()
    atr_series = atr(df, period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    basic_upperband = hl2 + (multiplier * atr_series)
    basic_lowerband = hl2 - (multiplier * atr_series)

    final_upperband = pd.Series(index=df.index, dtype=float)
    final_lowerband = pd.Series(index=df.index, dtype=float)
    st = pd.Series(index=df.index, dtype=bool)
    st_value = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            final_upperband.iloc[i] = basic_upperband.iloc[i]
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
            st.iloc[i] = True  # start with bullish by default
            st_value.iloc[i] = final_lowerband.iloc[i]  # just pick one
            continue

        # final upper band
        if (basic_upperband.iloc[i] < final_upperband.iloc[i-1]) or (df["Close"].iloc[i-1] > final_upperband.iloc[i-1]):
            final_upperband.iloc[i] = basic_upperband.iloc[i]
        else:
            final_upperband.iloc[i] = final_upperband.iloc[i-1]

        # final lower band
        if (basic_lowerband.iloc[i] > final_lowerband.iloc[i-1]) or (df["Close"].iloc[i-1] < final_lowerband.iloc[i-1]):
            final_lowerband.iloc[i] = basic_lowerband.iloc[i]
        else:
            final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

        # determine trend
        if st.iloc[i-1] and df["Close"].iloc[i] <= final_upperband.iloc[i]:
            st.iloc[i] = False
        elif (not st.iloc[i-1]) and df["Close"].iloc[i] >= final_lowerband.iloc[i]:
            st.iloc[i] = True
        else:
            st.iloc[i] = st.iloc[i-1]

        st_value.iloc[i] = final_lowerband.iloc[i] if st.iloc[i] else final_upperband.iloc[i]

    df["ST_bool"] = st
    df["ST_value"] = st_value
    df["ATR"] = atr_series
    return df

# ---------- STRATEGY LOGIC ----------
def analyze_all(nifty_ticker, equity_tickers, gold_silver_tickers, liquid_ticker):
    """Main analysis flow returning actionable plan."""
    report = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "nifty_ticker": nifty_ticker,
        "equity_tickers": equity_tickers,
        "gold_silver_tickers": gold_silver_tickers,
        "liquid_ticker": liquid_ticker,
        "nifty": {},
        "etfs": {},
        "gold_silver": {},
        "action_summary": []
    }

    # 1) NIFTY weekly ST
    try:
        df_nifty = fetch_weekly(nifty_ticker)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NIFTY data: {e}")

    st_nifty = supertrend(df_nifty, period=SUPER_PERIOD, multiplier=SUPER_MULT)
    # take latest week (most recent bar)
    latest_n = st_nifty.iloc[-1]
    nifty_is_green = bool(latest_n["ST_bool"])
    report["nifty"]["last_close"] = float(latest_n["Close"])
    report["nifty"]["st_is_green"] = nifty_is_green
    report["nifty"]["st_value"] = float(latest_n["ST_value"])

    # Decision: if NIFTY red -> move equities to LiquidBees only
    if not nifty_is_green:
        report["action_summary"].append({
            "action": "MARKET_RED",
            "note": "NIFTY Weekly Supertrend is RED. Move equity allocations to LIQUIDBEES only (per your rule)."
        })

    # 2) Equity ETFs: only meaningful if NIFTY is green; but we still compute ST for info
    for t in equity_tickers:
        try:
            df = fetch_weekly(t)
            st = supertrend(df, period=SUPER_PERIOD, multiplier=SUPER_MULT)
            last = st.iloc[-1]
            prev = st.iloc[-2] if len(st) >= 2 else last
            is_green = bool(last["ST_bool"])
            was_green = bool(prev["ST_bool"])
            report["etfs"][t] = {
                "last_close": float(last["Close"]),
                "st_is_green": is_green,
                "st_prev_green": was_green,
                "st_value": float(last["ST_value"])
            }

            # Actions
            if not nifty_is_green:
                # Nifty red: rule says exit equities
                report["action_summary"].append({
                    "ticker": t,
                    "action": "SELL",
                    "reason": "NIFTY weekly Supertrend is RED - per master rule exit equity ETFs and park in LIQUIDBEES."
                })
            else:
                # Nifty green: follow ETF's own weekly ST
                if is_green and not was_green:
                    report["action_summary"].append({
                        "ticker": t,
                        "action": "BUY",
                        "reason": "ETF weekly Supertrend turned GREEN and NIFTY is GREEN."
                    })
                elif not is_green and was_green:
                    report["action_summary"].append({
                        "ticker": t,
                        "action": "SELL",
                        "reason": "ETF weekly Supertrend turned RED while NIFTY is GREEN."
                    })
                else:
                    report["action_summary"].append({
                        "ticker": t,
                        "action": "HOLD",
                        "reason": "No change in ETF weekly Supertrend."
                    })
        except Exception as e:
            report["etfs"][t] = {"error": str(e)}
            report["action_summary"].append({
                "ticker": t,
                "action": "ERROR",
                "reason": str(e)
            })

    # 3) Gold & Silver - independent of NIFTY
    for t in gold_silver_tickers:
        try:
            df = fetch_weekly(t)
            st = supertrend(df, period=SUPER_PERIOD, multiplier=SUPER_MULT)
            last = st.iloc[-1]
            prev = st.iloc[-2] if len(st) >= 2 else last
            is_green = bool(last["ST_bool"])
            was_green = bool(prev["ST_bool"])
            report["gold_silver"][t] = {
                "last_close": float(last["Close"]),
                "st_is_green": is_green,
                "st_prev_green": was_green,
                "st_value": float(last["ST_value"])
            }
            if is_green and not was_green:
                report["action_summary"].append({
                    "ticker": t,
                    "action": "BUY",
                    "reason": "Gold/Silver weekly Supertrend turned GREEN (independent of NIFTY)."
                })
            elif not is_green and was_green:
                report["action_summary"].append({
                    "ticker": t,
                    "action": "SELL",
                    "reason": "Gold/Silver weekly Supertrend turned RED (independent of NIFTY)."
                })
            else:
                report["action_summary"].append({
                    "ticker": t,
                    "action": "HOLD",
                    "reason": "No change in weekly Supertrend for Gold/Silver."
                })
        except Exception as e:
            report["gold_silver"][t] = {"error": str(e)}
            report["action_summary"].append({
                "ticker": t,
                "action": "ERROR",
                "reason": str(e)
            })

    # 4) LiquidBees handling note
    report["liquidbees"] = {"ticker": liquid_ticker}
    if not nifty_is_green:
        report["action_summary"].append({
            "ticker": liquid_ticker,
            "action": "PARK",
            "reason": "NIFTY red -> move equity allocations to LiquidBees per system rule."
        })
    else:
        report["action_summary"].append({
            "ticker": liquid_ticker,
            "action": "STANDBY",
            "reason": "NIFTY green -> LiquidBees used only for leftover cash, not main allocation."
        })

    return report

def post_webhook(url: str, payload: dict):
    try:
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        return {"status_code": resp.status_code, "text": resp.text}
    except Exception as e:
        return {"error": str(e)}

# ---------- MAIN ----------
def main():
    nifty = os.getenv("NIFTY_TICKER", DEFAULT_NIFTY_TICKER)
    equity_list = os.getenv("EQUITY_ETFS", DEFAULT_EQUITY_ETFS).split(",")
    equity_list = [s.strip() for s in equity_list if s.strip()]
    gold_silver_list = os.getenv("GOLD_SILVER", DEFAULT_GOLD_SILVER).split(",")
    gold_silver_list = [s.strip() for s in gold_silver_list if s.strip()]
    liquid = os.getenv("LIQUID_TICKER", DEFAULT_LIQUID)

    # Analyze
    try:
        summary = analyze_all(nifty, equity_list, gold_silver_list, liquid)
    except Exception as e:
        print(f"ERROR during analysis: {e}", file=sys.stderr)
        sys.exit(2)

    # Print pretty summary
    print(json.dumps(summary, indent=2, default=str))

    # Optional: post to webhook
    if WEBHOOK_URL:
        result = post_webhook(WEBHOOK_URL, summary)
        print("Webhook result:", result)

    # Persist output to file (artifact)
    out_path = os.getenv("OUTPUT_PATH", "strategy_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if VERBOSE:
        print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()