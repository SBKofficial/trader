import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

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

def run_conservative_simulation():
    print("⏳ Fetching Data for Conservative Stress Test...")
    tickers = STOCKS + ["^NSEI"]
    df = yf.download(tickers, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df = df.swaplevel(0, 1, axis=1)

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
            n_open = df["^NSEI"]['Open'].iloc[i]
            n_prev = df["^NSEI"]['Close'].iloc[i-1]
            n_gap = (n_open - n_prev)/n_prev
            allowed = "BUY" if n_gap > 0.001 else ("SELL" if n_gap < -0.001 else "BOTH")
        except: continue

        # Select Best Stock
        candidates = []
        for stock in STOCKS:
            try:
                p_high, p_low, p_close = df[stock]['High'].iloc[i-1], df[stock]['Low'].iloc[i-1], df[stock]['Close'].iloc[i-1]
                c_open = df[stock]['Open'].iloc[i]
                
                rng = p_high - p_low
                if rng == 0: continue
                
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
                
                score = (p_close-p_low)/rng if signal=="BUY" else (p_high-p_close)/rng
                
                candidates.append({"stock": stock, "type": signal, "score": score, 
                                   "entry": buy_trig if signal=="BUY" else sell_trig})
            except: continue

        if not candidates: continue
        candidates.sort(key=lambda x: x['score'], reverse=True)
        trade = candidates[0] # Top Pick

        # EXECUTION (CONSERVATIVE LOGIC)
        # Check today's High/Low
        c_high = df[trade['stock']]['High'].iloc[i]
        c_low = df[trade['stock']]['Low'].iloc[i]
        entry = trade['entry']
        
        # Verify Trigger Happened
        triggered = False
        if trade['type'] == "BUY" and c_high >= entry: triggered = True
        if trade['type'] == "SELL" and c_low <= entry: triggered = True
        
        if not triggered: continue

        qty = int((capital * LEVERAGE) / entry)
        
        # PESSIMISTIC OUTCOME CHECK
        # If SL range is touched, we assume LOSS first.
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
                exit_price = df[trade['stock']]['Close'].iloc[i]
                
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
                exit_price = df[trade['stock']]['Close'].iloc[i]

        # Calc PnL
        gross = (exit_price - entry)*qty if trade['type']=="BUY" else (entry - exit_price)*qty
        charges = calculate_zerodha_charges((entry*qty), (exit_price*qty))
        net = gross - charges
        capital += net
        
        if outcome == "WIN": stats["Wins"] += 1
        else: stats["Losses"] += 1
        
        trade_log.append([today.date(), trade['stock'], trade['type'], outcome, int(net), int(capital)])

    # Print Full Log
    print(tabulate(trade_log, headers=["Date", "Stock", "Type", "Result", "PnL", "Balance"], tablefmt="simple"))
    print(f"\n🏆 FINAL CONSERVATIVE BALANCE: ₹{int(capital)}")
    print(f"💰 NET PROFIT: ₹{int(capital - total_invested)}")
    print(f"📊 ROI: {round(((capital-total_invested)/total_invested)*100, 2)}%")

run_conservative_simulation()
