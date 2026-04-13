import os
import time
import traceback
import requests
import numpy as np
from datetime import datetime, timedelta
import anthropic
import pytz

DISCORD_WEBHOOK   = "https://discord.com/api/webhooks/1493061685349584948/lqJsz0Bov6a67p9mvz8OTyzHH-5QguAwWrsSitEVmr2kvOe5ebA0YD1GockTUZJqbxxi"
NEWS_API_KEY      = "ce3679f211b6484ca94be0db9022d7d5"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MIN_CONFIDENCE    = 8
MIN_RR            = 1.5
SCAN_INTERVAL     = 300

WATCHLIST = {
    "BTC":  {"type": "crypto", "symbol": "BTC-USD",  "news_query": "Bitcoin BTC crypto"},
    "ETH":  {"type": "crypto", "symbol": "ETH-USD",  "news_query": "Ethereum ETH crypto"},
    "NVDA": {"type": "stock",  "symbol": "NVDA",     "news_query": "NVIDIA NVDA stock"},
    "AAPL": {"type": "stock",  "symbol": "AAPL",     "news_query": "Apple AAPL stock"},
}

ET = pytz.timezone('America/New_York')

# ─── OPENING RANGE TRACKER ───────────────────────────────────
opening_ranges = {}  # { symbol: {"high": x, "low": x, "established": bool} }

def get_et_now():
    return datetime.now(ET)

def is_market_open():
    now = get_et_now()
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    c = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return o <= now <= c

def is_opening_range_window():
    """True during 9:30-9:45 AM ET — observe only, no signals."""
    now = get_et_now()
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    end   = now.replace(hour=9,  minute=45, second=0, microsecond=0)
    return start <= now <= end

def is_too_close_to_open():
    """Avoid first 15 minutes of market — too choppy."""
    return is_opening_range_window()

def update_opening_range(symbol, highs, lows, closes):
    """Track the opening range (9:30-9:45 AM candles)."""
    now = get_et_now()
    today = now.date()

    # Reset at start of each day
    if symbol in opening_ranges:
        if opening_ranges[symbol].get("date") != today:
            opening_ranges[symbol] = {"date": today, "high": None, "low": None, "established": False}

    if symbol not in opening_ranges:
        opening_ranges[symbol] = {"date": today, "high": None, "low": None, "established": False}

    # After 9:45 AM, mark as established using first few candles
    market_open = now.replace(hour=9, minute=45, second=0, microsecond=0)
    if now >= market_open and not opening_ranges[symbol]["established"]:
        # Use last 3 candles as opening range
        or_high = max(highs[-3:]) if len(highs) >= 3 else highs[-1]
        or_low  = min(lows[-3:])  if len(lows)  >= 3 else lows[-1]
        opening_ranges[symbol]["high"] = or_high
        opening_ranges[symbol]["low"]  = or_low
        opening_ranges[symbol]["established"] = True
        print(f"  📊 ORB established: High ${or_high:.2f} | Low ${or_low:.2f}")

def get_orb_signal(symbol, current_price):
    """Check if price has broken the opening range."""
    if symbol not in opening_ranges:
        return None, None, None
    orb = opening_ranges[symbol]
    if not orb.get("established"):
        return None, None, None
    if current_price > orb["high"]:
        return "BULLISH_BREAKOUT", orb["high"], orb["low"]
    if current_price < orb["low"]:
        return "BEARISH_BREAKOUT", orb["high"], orb["low"]
    return "INSIDE_RANGE", orb["high"], orb["low"]

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
    data   = sorted(data, key=lambda x: x[0])[-limit:]
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
        signals.append("BOS Bullish")
    if closes[-1] < rl:
        signals.append("BOS Bearish")
    mid = len(closes) // 2
    ft  = closes[mid] - closes[0]
    st  = closes[-1]  - closes[mid]
    if ft < 0 and st > 0:
        signals.append("CHoCH Bullish flip")
    elif ft > 0 and st < 0:
        signals.append("CHoCH Bearish flip")
    return signals

def find_liquidity_sweeps(highs, lows, closes):
    sweeps = []
    if len(closes) < 10:
        return sweeps
    ph = max(highs[-10:-1])
    pl = min(lows[-10:-1])
    if highs[-1] > ph and closes[-1] < ph:
        sweeps.append({"direction": "bearish", "level": ph, "desc": "Sweep above highs (bearish)"})
    if lows[-1] < pl and closes[-1] > pl:
        sweeps.append({"direction": "bullish", "level": pl, "desc": "Sweep below lows (bullish reversal)"})
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
    return get_news("Federal Reserve interest rates inflation GDP earnings tariffs", 3)

# ─── AI STRATEGY ──────────────────────────────────────────────
def get_ai_strategy(asset, price, atr, rsi, macd, msig, vwap,
                    bb_u, bb_l, ema9, ema21, ema50,
                    fvgs, ifvgs, obs, bos_choch, liquidity,
                    resistances, supports, headlines, macro,
                    orb_signal, orb_high, orb_low):

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    orb_text = ""
    if orb_signal:
        orb_text = f"""
═══ OPENING RANGE BREAKOUT ═══
ORB High: ${orb_high:.2f} | ORB Low: ${orb_low:.2f}
Breakout: {orb_signal}
{'⚠️ Price inside range — wait for breakout' if orb_signal == 'INSIDE_RANGE' else '✅ Confirmed breakout — trade with momentum'}"""

    liq_descs = [l["desc"] for l in liquidity]

    prompt = f"""You are an elite SMC + ORB day trader. Analyze {asset} strictly.

RULES:
- If ORB is established and price is INSIDE_RANGE → signal NEUTRAL, wait for breakout
- If liquidity sweep detected + price reversed → consider counter-trend entry
- Only signal STRONG BUY or STRONG SELL when 5+ factors align
- R/R must be at least 1:1.5 or signal NEUTRAL
- Never fight a confirmed ORB breakout direction

═══ PRICE ═══
{asset}: ${price:,.2f} | ATR: ${atr:.2f}
{orb_text}

═══ TECHNICALS ═══
RSI(14): {rsi:.1f} {'⚠️ OVERBOUGHT' if rsi>70 else '⚠️ OVERSOLD' if rsi<30 else ''}
MACD: {macd:.4f} vs {msig:.4f} → {'🟢 Bullish' if macd>msig else '🔴 Bearish'}
EMA9: {ema9:.2f} | EMA21: {ema21:.2f} | EMA50: {ema50:.2f}
EMA Stack: {'🟢 Bullish' if ema9>ema21>ema50 else '🔴 Bearish' if ema9<ema21<ema50 else '⚪ Mixed'}
VWAP: {vwap:.2f} → {'🟢 Above' if price>vwap else '🔴 Below'}
Bollinger: {bb_u:.2f} / {bb_l:.2f}

═══ SMC ═══
IFVGs: {[f"{i['type']} ${i['zone_bottom']:.2f}-${i['zone_top']:.2f}" for i in ifvgs] or ['None']}
Order Blocks: {[f"{o['type']} ${o['bottom']:.2f}-${o['top']:.2f}" for o in obs[-3:]] or ['None']}
Structure: {bos_choch or ['None']}
Liquidity: {liq_descs or ['None']}
Resistance: {[f"${r:.2f}" for r in resistances[:2]]}
Support: {[f"${s:.2f}" for s in supports[:2]]}

═══ NEWS ═══
{chr(10).join(f'• {h}' for h in headlines[:3]) or '• None'}

═══ MACRO ═══
{chr(10).join(f'• {m}' for m in macro[:2]) or '• None'}

Respond EXACTLY:
SIGNAL: [STRONG BUY / STRONG SELL / NEUTRAL]
CONFIDENCE: [1-10]
ENTRY: $[price]
STOP LOSS: $[price]
TAKE PROFIT 1: $[price]
TAKE PROFIT 2: $[price]
RISK/REWARD: [e.g. 1:2.5]
TIMEFRAME: [hold duration]
REASON: [2-3 sentences]
KEY RISK: [one sentence]
SMC SETUP: [key pattern]
ORB NOTE: [how ORB affects this trade]
INVALIDATION: [what invalidates trade]"""

   for attempt in range(3):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            return msg.content[0].text
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                print(f"  ⏳ Anthropic busy, retrying in 10s...")
                time.sleep(10)
            else:
                raise
# ─── DISCORD ALERT ────────────────────────────────────────────
def send_discord(asset, price, strategy, ifvgs, liquidity, atr, orb_signal, orb_high, orb_low):
    lines  = strategy.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            k, v = line.split(':', 1)
            parsed[k.strip()] = v.strip()

    signal  = parsed.get('SIGNAL', 'NEUTRAL')
    conf    = parsed.get('CONFIDENCE', '?')
    entry   = parsed.get('ENTRY', '?')
    stop    = parsed.get('STOP LOSS', '?')
    tp1     = parsed.get('TAKE PROFIT 1', '?')
    tp2     = parsed.get('TAKE PROFIT 2', '?')
    rr      = parsed.get('RISK/REWARD', '?')
    tf      = parsed.get('TIMEFRAME', '?')
    reason  = parsed.get('REASON', '')
    risk    = parsed.get('KEY RISK', '')
    smc     = parsed.get('SMC SETUP', '')
    orb_note= parsed.get('ORB NOTE', '')
    invalid = parsed.get('INVALIDATION', '')

    emoji    = "🟢" if "BUY" in signal else "🔴"
    ifvg_txt = f"\n⚡ **IFVG:** {ifvgs[0]['type']} ${ifvgs[0]['zone_bottom']:.2f}-${ifvgs[0]['zone_top']:.2f}" if ifvgs else ""
    liq_txt  = f"\n💧 **Liquidity:** {liquidity[0]['desc']}" if liquidity else ""
    orb_txt  = f"\n📊 **ORB:** {orb_signal} | High ${orb_high:.2f} Low ${orb_low:.2f}" if orb_signal and orb_signal != "INSIDE_RANGE" else ""

    msg = f"""{emoji} **{asset} — {signal}** | Confidence: {conf}/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **Price:** ${price:,.2f} | ATR: ${atr:.2f}
📍 **Entry:** {entry}
🛑 **Stop Loss:** {stop}
🎯 **TP1:** {tp1} | **TP2:** {tp2}
⚖️ **R/R:** {rr} | ⏱️ **Hold:** {tf}{ifvg_txt}{liq_txt}{orb_txt}

🧠 **SMC:** {smc}
📊 **ORB Note:** {orb_note}
📝 **Reason:** {reason}
❌ **Invalidation:** {invalid}
⚠️ **Risk:** {risk}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {get_et_now().strftime('%I:%M %p ET')}"""

    resp = requests.post(DISCORD_WEBHOOK, json={"content": msg}, timeout=10)
    print(f"  {'✅ Discord sent!' if resp.status_code == 204 else f'❌ Discord error {resp.status_code}'}")

# ─── MAIN ANALYSIS ────────────────────────────────────────────
def analyze(name, info, macro):
    print(f"\n  [{name}]", end=" ")
    try:
        if info["type"] == "stock":
            if not is_market_open():
                print("market closed")
                return
            if is_too_close_to_open():
                print("⏳ opening range window — observing only")
                return
            opens, highs, lows, closes, vols = get_stock_candles(info["symbol"])
            update_opening_range(name, highs, lows, closes)
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

        fvgs, ifvgs   = find_fvg(opens, highs, lows, closes)
        obs            = find_order_blocks(opens, highs, lows, closes)
        bos_choch      = find_bos_choch(closes, highs, lows)
        liquidity      = find_liquidity_sweeps(highs, lows, closes)
        resistances, supports = find_support_resistance(highs, lows, closes)
        headlines      = get_news(info["news_query"])

        orb_signal, orb_high, orb_low = get_orb_signal(name, price)

        trend = "▲" if ema9 > ema21 else "▼"
        print(f"${price:,.2f} RSI:{rsi:.0f} MACD:{'▲' if macd>msig else '▼'} EMA:{trend}", end="")
        if ifvgs:     print(f" ⚡IFVG", end="")
        if bos_choch: print(f" 📊{bos_choch[0][:12]}", end="")
        if liquidity: print(f" 💧{liquidity[0]['direction']}", end="")
        if orb_signal: print(f" 📊ORB:{orb_signal[:7]}", end="")
        print()

        # Skip if inside opening range
        if orb_signal == "INSIDE_RANGE":
            print(f"  → Inside ORB range — waiting for breakout")
            return

        strategy = get_ai_strategy(
            name, price, atr, rsi, macd, msig, vwap,
            bb_u, bb_l, ema9, ema21, ema50,
            fvgs, ifvgs, obs, bos_choch, liquidity,
            resistances, supports, headlines, macro,
            orb_signal, orb_high, orb_low
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

        # Parse R/R
        rr_val = 0
        try:
            rr_str = parsed.get('RISK/REWARD', '0')
            parts  = rr_str.replace(' ', '').split(':')
            if len(parts) == 2:
                rr_val = float(parts[1]) / float(parts[0])
        except:
            rr_val = 0

        print(f"  → {signal} ({conf}/10) R/R:{rr_val:.1f}", end="")

        if "STRONG" in signal and conf >= MIN_CONFIDENCE and rr_val >= MIN_RR:
            print(" 🚨 ALERTING!")
            send_discord(name, price, strategy, ifvgs, liquidity, atr, orb_signal, orb_high, orb_low)
        else:
            reasons = []
            if "STRONG" not in signal: reasons.append("not STRONG")
            if conf < MIN_CONFIDENCE:  reasons.append(f"conf {conf}<{MIN_CONFIDENCE}")
            if rr_val < MIN_RR:        reasons.append(f"R/R {rr_val:.1f}<{MIN_RR}")
            print(f" — no alert ({', '.join(reasons)})")

    except Exception as e:
        print(f"ERROR")
        traceback.print_exc()

# ─── RUN LOOP ─────────────────────────────────────────────────
def run():
    print("🤖 AI Day Trading Signal Bot v2 (ORB Edition)")
    print(f"Watchlist: {list(WATCHLIST.keys())}")
    print(f"Filters: STRONG signal | Confidence {MIN_CONFIDENCE}+ | R/R {MIN_RR}+")
    print(f"ORB: Stocks wait for 9:30-9:45 range, then trade breakouts only")
    print(f"Scan: every {SCAN_INTERVAL//60} min | Crypto 24/7 | Stocks market hours\n")

    while True:
        now = get_et_now()
        print(f"{'='*50}")
        print(f"🔍 {now.strftime('%I:%M %p ET | %A %b %d')}", end="")

        if is_opening_range_window():
            print(" ⏳ OPENING RANGE WINDOW — stocks paused")
        elif is_market_open():
            print(" 🟢 MARKET OPEN")
        else:
            print(" 🔴 MARKET CLOSED")

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
