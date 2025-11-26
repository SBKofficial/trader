import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
#        CONFIGURATION
# ==========================================
START_CAPITAL = 15000
MONTHLY_SIP = 2000
LEVERAGE = 5
BUFFER_PCT = 0.0005

# Strategy Rules (1:3 Ratio)
TARGET_PCT = 0.006   # 0.6% Profit Target
SL_PCT = 0.002       # 0.2% Initial Stop Loss
STEP_1_PCT = 0.002   # Move SL to Entry
STEP_2_PCT = 0.004   # Move SL to +0.2%

STOCKS = [
    "TATASTEEL.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", 
    "BPCL.NS", "COALINDIA.NS", "ITC.NS", "BEL.NS"
]

# ==========================================
#        LOGIC ENGINE
# ==========================================

def calculate_zerodha_charges(buy_val, sell_val):
    """Calculates exact Zerodha Intraday Equity Charges"""
    turnover = buy_val + sell_val
    brokerage = min(20, turnover * 0.0003)
    stt = sell_val * 0.00025
    txn_charge = turnover * 0.0000297
    gst = (brokerage + txn_charge) * 0.18
    stamp = buy_val * 0.00003
    sebi = turnover * 0.000001
    return round(brokerage + stt + txn_charge + gst + stamp + sebi, 2)

def round_price(num):
    return round(0.05 * round(num/0.05), 2)

def get_data():
    print("⏳ Fetching 1 Year of Historical Data... (Please Wait)")
    tickers = STOCKS + ["^NSEI"]
    data = yf.download(tickers, period="1y", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.swaplevel(0, 1, axis=1)
    return data

def run_simulation():
    df = get_data()
    capital = START_CAPITAL
    total_invested = START_CAPITAL
    
    trade_log = []
    stats = {"Wins": 0, "Losses": 0, "Breakeven": 0, "Partial": 0}
    
    dates = df.index
    print(f"\n🚀 Simulation Started: ₹{capital} Initial Capital")

    for i in range(1, len(dates)):
        today = dates[i]
        prev_date = dates[i-1]
        
        # 1. SIP LOGIC
        if today.month != prev_date.month:
            capital += MONTHLY_SIP
            total_invested += MONTHLY_SIP

        # 2. NIFTY FILTER (Step 0)
        try:
            nifty_open = df["^NSEI"]['Open'].iloc[i]
            nifty_prev = df["^NSEI"]['Close'].iloc[i-1]
            nifty_change = (nifty_open - nifty_prev) / nifty_prev
            
            allowed = "NEUTRAL"
            if nifty_change > 0.001: allowed = "BUY"   # Gap Up > 0.1%
            elif nifty_change < -0.001: allowed = "SELL" # Gap Down < -0.1%
        except: continue

        # 3. RANKING
        candidates = []
        for stock in STOCKS:
            try:
                p_high = df[stock]['High'].iloc[i-1]
                p_low = df[stock]['Low'].iloc[i-1]
                p_close = df[stock]['Close'].iloc[i-1]
                c_open = df[stock]['Open'].iloc[i]
                c_high = df[stock]['High'].iloc[i]
                c_low = df[stock]['Low'].iloc[i]
                
                if np.isnan(p_close) or np.isnan(c_open): continue
                rng = p_high - p_low
                
                buy_trig = round_price(p_high + (p_close * BUFFER_PCT))
                sell_trig = round_price(p_low - (p_close * BUFFER_PCT))
                
                signal = None
                if p_close >= (p_high - rng*0.25): signal = "BUY"
                elif p_close <= (p_low + rng*0.25): signal = "SELL"
                
                # Nifty Filter
                if allowed == "BUY" and signal == "SELL": continue
                if allowed == "SELL" and signal == "BUY": continue
                
                # Gap Filter (Step 1)
                if signal == "BUY":
                    if c_open > buy_trig * 1.002: continue
                    if c_high < buy_trig: continue # Did not trigger
                    entry = buy_trig
                    score = (p_close - p_low)/rng
                else: # SELL
                    if c_open < sell_trig * 0.998: continue
                    if c_low > sell_trig: continue # Did not trigger
                    entry = sell_trig
                    score = (p_high - p_close)/rng
                
                candidates.append({
                    "stock": stock, "type": signal, "score": score,
                    "entry": entry, "high": c_high, "low": c_low
                })
            except: continue
            
        if not candidates: continue
        
        # 4. EXECUTION (Pick Top 1)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        trade = candidates[0]
        
        buying_power = capital * LEVERAGE
        qty = int(buying_power / trade['entry'])
        entry = trade['entry']
        outcome = ""
        exit_price = 0
        
        # Step-Ladder Simulation
        if trade['type'] == "BUY":
            tgt = round_price(entry * (1 + TARGET_PCT))
            sl = round_price(entry * (1 - SL_PCT))
            step1 = round_price(entry * (1 + STEP_1_PCT))
            step2 = round_price(entry * (1 + STEP_2_PCT))
            step2_lock = round_price(entry * (1 + 0.002))
            
            # Priority: High vs Low (Assumption: Trend continuation)
            if trade['high'] >= tgt:
                outcome, exit_price = "WIN 🎯", tgt
            elif trade['high'] >= step2:
                outcome, exit_price = "PARTIAL 🔒", step2_lock
            elif trade['high'] >= step1:
                outcome, exit_price = "BREAKEVEN 🛡️", entry
            elif trade['low'] <= sl:
                outcome, exit_price = "LOSS ❌", sl
            else:
                outcome, exit_price = "EOD CLOSE", (trade['high']+trade['low'])/2
                
            gross = (exit_price - entry) * qty
            
        else: # SELL
            tgt = round_price(entry * (1 - TARGET_PCT))
            sl = round_price(entry * (1 + SL_PCT))
            step1 = round_price(entry * (1 - STEP_1_PCT))
            step2 = round_price(entry * (1 - STEP_2_PCT))
            step2_lock = round_price(entry * (1 - 0.002))
            
            if trade['low'] <= tgt:
                outcome, exit_price = "WIN 🎯", tgt
            elif trade['low'] <= step2:
                outcome, exit_price = "PARTIAL 🔒", step2_lock
            elif trade['low'] <= step1:
                outcome, exit_price = "BREAKEVEN 🛡️", entry
            elif trade['high'] >= sl:
                outcome, exit_price = "LOSS ❌", sl
            else:
                outcome, exit_price = "EOD CLOSE", (trade['high']+trade['low'])/2

            gross = (entry - exit_price) * qty

        # Charges
        turnover = (entry + exit_price) * qty
        charges = calculate_zerodha_charges(turnover/2, turnover/2)
        net_pnl = gross - charges
        capital += net_pnl
        
        if "WIN" in outcome: stats["Wins"] += 1
        elif "LOSS" in outcome: stats["Losses"] += 1
        elif "BREAKEVEN" in outcome: stats["Breakeven"] += 1
        elif "PARTIAL" in outcome: stats["Partial"] += 1
        
        trade_log.append([today.strftime("%Y-%m-%d"), trade['stock'], trade['type'], outcome, int(net_pnl), int(capital)])

    print("\n" + "="*60)
    print(tabulate(trade_log[-10:], headers=["Date", "Stock", "Type", "Result", "PnL", "Balance"], tablefmt="grid"))
    print(f"\n📊 FINAL RESULTS")
    print(f"Start: ₹{START_CAPITAL} | SIP: ₹{total_invested - START_CAPITAL} | Invested: ₹{total_invested}")
    print(f"Final: ₹{int(capital)} | Profit: ₹{int(capital - total_invested)}")
    print(f"ROI:   {round(((capital-total_invested)/total_invested)*100, 2)}%")
    print(f"Stats: {stats}")
    print("="*60)

if __name__ == "__main__":
    run_simulation()
