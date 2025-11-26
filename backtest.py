import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning) # Ignore divide by zero warnings

# ==========================================
#        CONSERVATIVE BACKTEST CONFIG
# ==========================================
START_CAPITAL = 15000
MONTHLY_SIP = 2000
LEVERAGE = 5
BUFFER_PCT = 0.0005
TARGET_PCT = 0.006
SL_PCT = 0.002

STOCKS = ["TATASTEEL.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", 
          "BPCL.NS", "COALINDIA.NS", "ITC.NS", "BEL.NS"]

def calculate_zerodha_charges(turnover):
    brokerage = min(20, turnover * 0.0003)
    stt = (turnover / 2) * 0.00025
    txn_charge = turnover * 0.0000297
    gst = (brokerage + txn_charge) * 0.18
    stamp = (turnover / 2) * 0.00003
    sebi = turnover * 0.000001
    return round(brokerage + stt + txn_charge + gst + stamp + sebi, 2)

def round_price(num):
    return round(0.05 * round(num/0.05), 2)

def safe_float(val):
    """Ensures the value is a clean float, not NaN or Inf"""
    try:
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except:
        return None

def run_conservative_simulation():
    print("⏳ Fetching Data for Conservative Stress Test...")
    tickers = STOCKS + ["^NSEI"]
    df = yf.download(tickers, period="1y", interval="1d", progress=False)
    
    # Flatten MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.swaplevel(0, 1, axis=1)
        except: pass

    capital = START_CAPITAL
    total_invested = START_CAPITAL
    trade_log = []
    stats = {"Wins": 0, "Losses": 0}

    dates = df.index
    
    for i in range(1, len(dates)):
        today = dates[i]
        prev_date = dates[i-1]
        
        # SIP
        if today.month != prev_date.month:
            capital += MONTHLY_SIP
            total_invested += MONTHLY_SIP

        # Nifty Filter
        try:
            n_open = safe_float(df["^NSEI"]['Open'].iloc[i])
            n_prev = safe_float(df["^NSEI"]['Close'].iloc[i-1])
            
            if n_open is None or n_prev is None: continue
            
            n_gap = (n_open - n_prev)/n_prev
            allowed = "BUY" if n_gap > 0.001 else ("SELL" if n_gap < -0.001 else "BOTH")
        except: continue

        # Select Best Stock
        candidates = []
        for stock in STOCKS:
            try:
                # Use safe_float to prevent NaNs
                p_high = safe_float(df[stock]['High'].iloc[i-1])
                p_low = safe_float(df[stock]['Low'].iloc[i-1])
                p_close = safe_get_value = safe_float(df[stock]['Close'].iloc[i-1])
                c_open = safe_float(df[stock]['Open'].iloc[i])
                
                # Skip if any data is bad
                if None in [p_high, p_low, p_close, c_open]: continue
                
                rng = p_high - p_low
                
                # Safety Check: If Range is 0, skip division
                if rng <= 0: continue
                
                buy_trig = round_price(p_high + p_close * BUFFER_PCT)
                sell_trig = round_price(p_low - p_close * BUFFER_PCT)
                
                signal = None
                if p_close >= (p_high - rng*0.25): signal = "BUY"
                elif p_close <= (p_low + rng*0.25): signal = "SELL"
                
                if allowed == "BUY" and signal == "SELL": continue
                if allowed == "SELL" and signal == "BUY": continue
                
                # Gap Filter
                if signal == "BUY" and c_open > buy_trig * 1.002: continue
                if signal == "SELL" and c_open < sell_trig * 0.998: continue
                
                # Calculate Score (The Error Fix)
                if signal == "BUY":
                    score = (p_close - p_low) / rng
                else:
                    score = (p_high - p_close) / rng
                
                # Check for infinity score
                if np.isinf(score) or np.isnan(score): continue

                candidates.append({"stock": stock, "type": signal, "score": score, 
                                   "entry": buy_trig if signal=="BUY" else sell_trig})
            except: continue

        if not candidates: continue
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        trade = candidates[0] # Top Pick

        # EXECUTION (CONSERVATIVE LOGIC)
        try:
            c_high = safe_float(df[trade['stock']]['High'].iloc[i])
            c_low = safe_float(df[trade['stock']]['Low'].iloc[i])
            
            if c_high is None or c_low is None: continue
            
            entry = trade['entry']
            
            # Verify Trigger Happened
            triggered = False
            if trade['type'] == "BUY" and c_high >= entry: triggered = True
            if trade['type'] == "SELL" and c_low <= entry: triggered = True
            
            if not triggered: continue

            qty = int((capital * LEVERAGE) / entry)
            
            # PESSIMISTIC OUTCOME CHECK
            outcome = "WIN"
            exit_price = 0
            
            if trade['type'] == "BUY":
                sl = round_price(entry * (1 - SL_PCT))
                tgt = round_price(entry * (1 + TARGET_PCT))
                
                if c_low <= sl: # SL touched? Assume Loss.
                    outcome = "LOSS"
                    exit_price = sl
                elif c_high >= tgt:
                    outcome = "WIN"
                    exit_price = tgt
                else:
                    outcome = "EOD"
                    exit_price = safe_float(df[trade['stock']]['Close'].iloc[i])
                    
            else: # SELL
                sl = round_price(entry * (1 + SL_PCT))
                tgt = round_price(entry * (1 - TARGET_PCT))
                
                if c_high >= sl: # SL touched? Assume Loss.
                    outcome = "LOSS"
                    exit_price = sl
                elif c_low <= tgt:
                    outcome = "WIN"
                    exit_price = tgt
                else:
                    outcome = "EOD"
                    exit_price = safe_float(df[trade['stock']]['Close'].iloc[i])

            # Calc PnL
            gross = (exit_price - entry)*qty if trade['type']=="BUY" else (entry - exit_price)*qty
            turnover = (entry * qty) + (exit_price * qty)
            charges = calculate_zerodha_charges(turnover)
            net = gross - charges
            capital += net
            
            if outcome == "WIN": stats["Wins"] += 1
            else: stats["Losses"] += 1
            
            trade_log.append([today.date(), trade['stock'], trade['type'], outcome, int(net), int(capital)])

        except: continue

    # Print Full Log
    print(tabulate(trade_log[-20:], headers=["Date", "Stock", "Type", "Result", "PnL", "Balance"], tablefmt="simple"))
    print(f"\n🏆 FINAL CONSERVATIVE BALANCE: ₹{int(capital)}")
    print(f"💰 NET PROFIT: ₹{int(capital - total_invested)}")
    try:
        roi = round(((capital-total_invested)/total_invested)*100, 2)
        print(f"📊 ROI: {roi}%")
    except:
        print("📊 ROI: 0%")

if __name__ == "__main__":
    run_conservative_simulation()
