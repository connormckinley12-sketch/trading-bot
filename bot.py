import os
import time
import traceback
import requests
import numpy as np
from datetime import datetime
import anthropic
import pytz

DISCORD_WEBHOOK  = "https://discord.com/api/webhooks/1493061685349584948/lqJsz0Bov6a67p9mvz8OTyzHH-5QguAwWrsSitEVmr2kvOe5ebA0YD1GockTUZJqbxxi"
NEWS_API_KEY     = "ce3679f211b6484ca94be0db9022d7d5"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MIN_CONFIDENCE   = 8
SCAN_INTERVAL    = 300  # 5 minutes

WATCHLIST = {
    "BTC":  {"type": "crypto", "symbol": "BTC-USD",  "news_query": "Bitcoin BTC crypto"},
    "ETH":  {"type": "crypto", "symbol": "ETH-USD",  "news_query": "Ethereum ETH crypto"},
    "NVDA": {"type": "stock",  "symbol": "NVDA",     "news_query": "NVIDIA NVDA stock"},
    "AAPL": {"type": "stock",  "symbol": "AAPL",     "news_query": "Apple AAPL stock"},
}

# ─── MARKET HOURS ────────────────────────────────────────────
def is_market_open():
    et  = pytz.timezone('America/New_York')
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    c = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return o <= now <= c

# ─── DATA FETCHING ────────────────────────────────────────────
def get_crypto_candles(symbol, granularity=300, limit=100):
    resp = requests.get(
        f"https://api.exchange.coinbase.com/products/{symbol}/candles",
        params={"granularity": granularity},
        timeout=10
    )
    data = resp.json()
    if not isinstance(data, list) or len(data) == 0:
        raise Exception(f"Coinbase error: {data}")
    data = sorted(data, key=lambda x: x[0])[-limit:]
    opens  = [float(c[3]) for c in data]
    highs  = [float(c[2]) for c in data]
    lows   = [float(c[1]) for c in data]
    closes = [float(c[4]) for c in data]
    vols   = [float(c[5]) for c in data]
    return opens, highs, lows, closes, vols

def get_stock_candles(symbol):
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "5m", "range": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10
    )
    data   = resp.json()
    result = data["chart"]["result"][0]
    q      = result["indicators"]["quote"][0]
    opens  = [x for x in q["open"]   if x is not None]
    highs  = [x for x in q["high"]   if x is not None]
    lows   = [x for x in q["low"]    if x is not None]
    closes = [x for x in q["close"]  if x is not None]
    vols   = [x for x in q["volume"] if x is not None]
    if len(closes) < 10:
        raise Exception("Not enough stock data")
    return opens, highs, lows, closes, vols

# ─── INDICATORS ───────────────────────────────────────────────
def calc_rsi(closes, period=14):
    diffs  = np.diff(closes)
    gains  = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    if al == 0:
        return 100.0
    return float(100 - (100 / (1 + ag / al)))

def calc_macd(closes):
    c      = np.array(closes)
    ema12  = c[-12:].mean()
    ema26  = c[-26:].mean() if len(c) >= 26 else c.mean()
    macd   = ema12 - ema26
    signal = c[-9:].mean() - c[-18:].mean() if len(c) >= 18 else 0.0
    return float(macd), float(signal)

def calc_ema(closes, period):
    c   = np.array(closes, dtype=float)
    k   = 2.0 / (period + 1)
    ema = c[0]
    for p in c[1:]:
        ema = p * k + ema * (1 - k)
    return float(ema)

def calc_vwap(closes, vols):
    c = np.array(closes)
    v = np.array(vols)
    return float(np.sum(c * v) / np.sum(v)) if np.sum(v) > 0 else float(c[-1])

def calc_bollinger(closes, period=20):
    arr  = np.array(closes[-period:])
    mean = float(arr.mean())
    std  = float(arr.std())
    return mean + 2*std, mean, mean - 2*std

def calc_atr(highs, lows, closes, period=14):
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        trs.append(tr)
    return float(np.mean(trs[-period:])) if trs else 0.0

# ─── SMC ANALYSIS ─────────────────────────────────────────────
def find_fvg(opens, highs, lows, closes):
    fvgs  = []
    ifvgs = []
    for i in range(2, len(closes)):
        if lows[i] > highs[i-2]:
            fvgs.append({"type": "bullish_fvg", "top": lows[i], "bottom": highs[i-2]})
        if highs[i] < lows[i-2]:
            fvgs.append({"type": "bearish_fvg", "top": lows[i-2], "bottom": highs[i]})
    current = closes[-1]
    for fvg in fvgs[-15:]:
        if fvg["bottom"] <= current <= fvg["top"]:
            t = "bearish_ifvg" if fvg["type"] == "bullish_fvg" else "bullish_ifvg"
            ifvgs.append({"type": t, "zone_top": fvg["top"], "zone_bottom": fvg["bottom"]})
    return fvgs[-5:], ifvgs

def find_order_blocks(opens, highs, lows, closes):
    obs = []
    for i in range(1, len(closes)-1):
        if closes[i] > opens[i] and closes[i+1] < opens[i+1]:
            if (opens[i+1]-closes[i+1]) > (closes[i]-opens[i]) * 1.5:
                obs.append({"type": "bearish_ob", "top": highs[i], "bottom": lows[i]})
        if closes[i] < opens[i] and closes[i+1] > opens[i+1]:
            if (closes[i+1]-opens[i+1]) > (opens[i]-closes[i]) * 1.5:
                obs.append({"type": "bullish_ob", "top": highs[i], "bottom": lows[i]})
    return obs[-5:]

def find_bos_choch(closes, highs, lows):
    signals = []
    if len(closes) < 20:
        return signals
    rh = max(highs[-20:-1])
    rl = min(lows[-20:-1])
    if closes[-1] > rh:
        signals.append("BOS Bullish — broke above recent high")
    if closes[-1] < rl:
        signals.append("BOS Bearish — broke below recent low")
    mid = len(closes) // 2
    ft  = closes[mid] - closes[0]
    st  = closes[-1]  - closes[mid]
    if ft < 0 and st > 0:
        signals.append("CHoCH — bearish to bullish flip")
    elif ft > 0 and st < 0:
        signals.append("CHoCH — bullish to bearish flip")
    return signals

def find_liquidity_sweeps(highs, lows, closes):
    sweeps = []
    if len(closes) < 10:
        return sweeps
    ph = max(highs[-10:-1])
    pl = min(lows[-10:-1])
    if highs[-1] > ph and closes[-1] < ph:
        sweeps.append("Sweep above highs (bearish)")
    if lows[-1] < pl and closes[-1] > pl:
        sweeps.append("Sweep below lows (bullish)")
    return sweeps

def find_support_resistance(highs, lows, closes, lookback=50):
    if len(closes) < lookback:
        lookback = len(closes)
    recent_highs = sorted(highs[-lookback:], reverse=True)[:3]
    recent_lows  = sorted(lows[-lookback:])[:3]
    return recent_highs, recent_lows

# ─── NEWS & MACRO ─────────────────────────────────────────────
def get_news(query, n=5):
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "sortBy": "publishedAt",
                    "pageSize": n, "apiKey": NEWS_API_KEY},
            timeout=10
        )
        return [a["title"] for a in resp.json().get("articles", [])[:n]]
    except:
        return []

def get_macro():
    return get_news("Federal Reserve interest rates inflation GDP earnings", 3)

# ─── AI STRATEGY ──────────────────────────────────────────────
def get_ai_strategy(asset, price, atr, rsi, macd, msig, vwap,
                    bb_u, bb_l, ema9, ema21, ema50,
                    fvgs, ifvgs, obs, bos_choch, liquidity,
                    resistances, supports, headlines, macro):

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are an elite SMC day trader. Analyze {asset} strictly. Only signal STRONG BUY or STRONG SELL when 5+ factors align with high conviction. Otherwise NEUTRAL.

═══ PRICE ACTION ═══
Asset: {asset} | Price: ${price:,.2f} | ATR: ${atr:.2f}

═══ TECHNICALS ═══
RSI(14): {rsi:.1f} {'⚠️ OVERBOUGHT' if rsi>70 else '⚠️ OVERSOLD' if rsi<30 else ''}
MACD: {macd:.4f} vs Signal {msig:.4f} → {'🟢 Bullish crossover' if macd>msig else '🔴 Bearish crossover'}
EMA9: {ema9:.2f} | EMA21: {ema21:.2f} | EMA50: {ema50:.2f}
EMA Trend: {'🟢 Bullish stack' if ema9>ema21>ema50 else '🔴 Bearish stack' if ema9<ema21<ema50 else '⚪ Mixed'}
VWAP: {vwap:.2f} → Price {'🟢 above' if price>vwap else '🔴 below'} VWAP
Bollinger: Upper {bb_u:.2f} | Lower {bb_l:.2f}
BB Position: {'Near upper band' if price>bb_u*0.99 else 'Near lower band' if price<bb_l*1.01 else 'Mid range'}

═══ SMC ANALYSIS ═══
IFVGs Active: {[f"{i['type']} ${i['zone_bottom']:.2f}-${i['zone_top']:.2f}" for i in ifvgs] or ['None']}
Order Blocks: {[f"{o['type']} ${o['bottom']:.2f}-${o['top']:.2f}" for o in obs[-3:]] or ['None']}
Market Structure: {bos_choch or ['No major BOS/CHoCH']}
Liquidity: {liquidity or ['None detected']}
Key Resistance: {[f"${r:.2f}" for r in resistances[:2]]}
Key Support: {[f"${s:.2f}" for s in supports[:2]]}

═══ NEWS & SENTIMENT ═══
{chr(10).join(f'• {h}' for h in headlines[:3]) or '• No recent news'}

═══ MACRO ═══
{chr(10).join(f'• {m}' for m in macro[:2]) or '• No macro events'}

Respond EXACTLY in this format (no extra text):
SIGNAL: [STRONG BUY / STRONG SELL / NEUTRAL]
CONFIDENCE: [1-10]
ENTRY: $[price]
STOP LOSS: $[price]
TAKE PROFIT 1: $[price]
TAKE PROFIT 2: $[price]
RISK/REWARD: [e.g. 1:2.5]
TIMEFRAME: [e.g. 15-60 min hold]
REASON: [2-3 sentences combining technicals + SMC + news]
KEY RISK: [one sentence]
SMC SETUP: [key SMC pattern]
INVALIDATION: [what would invalidate this trade]"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

# ─── DISCORD ALERT ────────────────────────────────────────────
def send_discord(asset, price, strategy, ifvgs, liquidity, atr):
    lines  = strategy.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            k, v = line.split(':', 1)
            parsed[k.strip()] = v.strip()

    signal    = parsed.get('SIGNAL', 'NEUTRAL')
    conf      = parsed.get('CONFIDENCE', '?')
    entry     = parsed.get('ENTRY', '?')
    stop      = parsed.get('STOP LOSS', '?')
    tp1       = parsed.get('TAKE PROFIT 1', '?')
    tp2       = parsed.get('TAKE PROFIT 2', '?')
    rr        = parsed.get('RISK/REWARD', '?')
    tf        = parsed.get('TIMEFRAME', '?')
    reason    = parsed.get('REASON', '')
    risk      = parsed.get('KEY RISK', '')
    smc       = parsed.get('SMC SETUP', '')
    invalid   = parsed.get('INVALIDATION', '')

    emoji     = "🟢" if "BUY" in signal else "🔴"
    ifvg_txt  = f"\n⚡ **IFVG:** {ifvgs[0]['type']} ${ifvgs[0]['zone_bottom']:.2f}-${ifvgs[0]['zone_top']:.2f}" if ifvgs else ""
    liq_txt   = f"\n💧 **Liquidity:** {liquidity[0]}" if liquidity else ""
    et        = pytz.timezone('America/New_York')
    now_str   = datetime.now(et).strftime('%I:%M %p ET')

    msg = f"""{emoji} **{asset} — {signal}** | Confidence: {conf}/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **Price:** ${price:,.2f} | ATR: ${atr:.2f}
📍 **Entry:** {entry}
🛑 **Stop Loss:** {stop}
🎯 **TP1:** {tp1} | **TP2:** {tp2}
⚖️ **R/R:** {rr} | ⏱️ **Hold:** {tf}{ifvg_txt}{liq_txt}

🧠 **SMC:** {smc}
📝 **Reason:** {reason}
❌ **Invalidation:** {invalid}
⚠️ **Risk:** {risk}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {now_str}"""

    resp = requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)
    if resp.status_code == 204:
        print(f"  ✅ Discord alert sent!")
    else:
        print(f"  ❌ Discord error: {resp.status_code}")

# ─── MAIN ANALYSIS ────────────────────────────────────────────
def analyze(name, info, macro):
    print(f"\n  [{name}]", end=" ")
    try:
        if info["type"] == "stock":
            if not is_market_open():
                print("market closed")
                return
            opens, highs, lows, closes, vols = get_stock_candles(info["symbol"])
        else:
            opens, highs, lows, closes, vols = get_crypto_candles(info["symbol"])

        if len(closes) < 30:
            print("not enough data")
            return

        price  = closes[-1]
        rsi    = calc_rsi(closes)
        macd, msig = calc_macd(closes)
        vwap   = calc_vwap(closes, vols)
        bb_u, _, bb_l = calc_bollinger(closes)
        ema9   = calc_ema(closes, 9)
        ema21  = calc_ema(closes, 21)
        ema50  = calc_ema(closes, 50)
        atr    = calc_atr(highs, lows, closes)

        fvgs, ifvgs = find_fvg(opens, highs, lows, closes)
        obs         = find_order_blocks(opens, highs, lows, closes)
        bos_choch   = find_bos_choch(closes, highs, lows)
        liquidity   = find_liquidity_sweeps(highs, lows, closes)
        resistances, supports = find_support_resistance(highs, lows, closes)
        headlines   = get_news(info["news_query"])

        trend = "▲" if ema9 > ema21 else "▼"
        print(f"${price:,.2f} RSI:{rsi:.0f} MACD:{'▲' if macd>msig else '▼'} EMA:{trend}", end="")
        if ifvgs:    print(f" ⚡IFVG", end="")
        if bos_choch: print(f" 📊{bos_choch[0][:10]}", end="")
        if liquidity: print(f" 💧", end="")
        print()

        strategy = get_ai_strategy(
            name, price, atr, rsi, macd, msig, vwap,
            bb_u, bb_l, ema9, ema21, ema50,
            fvgs, ifvgs, obs, bos_choch, liquidity,
            resistances, supports, headlines, macro
        )

        lines  = strategy.strip().split('\n')
        parsed = {}
        for line in lines:
            if ':' in line:
                k, v = line.split(':', 1)
                parsed[k.strip()] = v.strip()

        signal = parsed.get('SIGNAL', 'NEUTRAL')
        try:
            conf = int(''.join(filter(str.isdigit, parsed.get('CONFIDENCE', '0'))))
        except:
            conf = 0

        print(f"  → {signal} ({conf}/10)", end="")

        if "STRONG" in signal and conf >= MIN_CONFIDENCE:
            print(" 🚨 ALERTING!")
            send_discord(name, price, strategy, ifvgs, liquidity, atr)
        else:
            print(" — no alert")

    except Exception as e:
        print(f"ERROR")
        traceback.print_exc()

# ─── RUN LOOP ─────────────────────────────────────────────────
def run():
    et = pytz.timezone('America/New_York')
    print("🤖 AI Day Trading Signal Bot")
    print(f"Watchlist: {list(WATCHLIST.keys())}")
    print(f"Min signal: STRONG | Min confidence: {MIN_CONFIDENCE}/10")
    print(f"Scan interval: {SCAN_INTERVAL//60} minutes")
    print(f"Crypto: 24/7 | Stocks: Market hours only\n")

    while True:
        now = datetime.now(et)
        print(f"{'='*50}")
        print(f"🔍 Scan — {now.strftime('%I:%M %p ET | %A %b %d')}")
        market_status = "🟢 OPEN" if is_market_open() else "🔴 CLOSED"
        print(f"Market: {market_status}")

        macro = get_macro()
        if macro:
            print(f"Macro: {macro[0][:70]}...")

        for name, info in WATCHLIST.items():
            analyze(name, info, macro)
            time.sleep(2)

        print(f"\n⏳ Next scan in {SCAN_INTERVAL//60} min...")
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    run()
