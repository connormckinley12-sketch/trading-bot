import os
import time
import requests
import numpy as np
from datetime import datetime
import anthropic
import pytz

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1493061685349584948/lqJsz0Bov6a67p9mvz8OTyzHH-5QguAwWrsSitEVmr2kvOe5ebA0YD1GockTUZJqbxxi"
NEWS_API_KEY = "ce3679f211b6484ca94be0db9022d7d5"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MIN_CONFIDENCE = 8

WATCHLIST = {
    "BTC":  {"type": "crypto", "symbol": "BTC-USD",  "news_query": "Bitcoin BTC crypto"},
    "ETH":  {"type": "crypto", "symbol": "ETH-USD",  "news_query": "Ethereum ETH crypto"},
    "NVDA": {"type": "stock",  "symbol": "NVDA",     "news_query": "NVIDIA NVDA stock"},
    "AAPL": {"type": "stock",  "symbol": "AAPL",     "news_query": "Apple AAPL stock"},
}

def is_market_open():
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close

def get_crypto_candles(symbol, interval=900, limit=100):
    resp = requests.get(
        f"https://api.exchange.coinbase.com/products/{symbol}/candles",
        params={"granularity": interval},
        timeout=10
    )
    candles = resp.json()
    if not isinstance(candles, list):
        raise Exception(f"Coinbase error: {candles}")
    candles = sorted(candles, key=lambda x: x[0])[-limit:]
    opens  = [float(c[3]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[1]) for c in candles]
    closes = [float(c[4]) for c in candles]
    vols   = [float(c[5]) for c in candles]
    return opens, highs, lows, closes, vols

def get_stock_candles(symbol):
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "15m", "range": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10
    )
    data = resp.json()
    result = data["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    opens  = [x for x in q["open"]   if x is not None]
    highs  = [x for x in q["high"]   if x is not None]
    lows   = [x for x in q["low"]    if x is not None]
    closes = [x for x in q["close"]  if x is not None]
    vols   = [x for x in q["volume"] if x is not None]
    return opens, highs, lows, closes, vols

def calc_rsi(closes, period=14):
    diffs  = np.diff(closes)
    gains  = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_macd(closes):
    c = np.array(closes)
    ema12   = c[-12:].mean()
    ema26   = c[-26:].mean() if len(c) >= 26 else c.mean()
    macd    = ema12 - ema26
    signal  = c[-9:].mean() - c[-18:].mean() if len(c) >= 18 else 0
    return macd, signal

def calc_vwap(closes, vols):
    c = np.array(closes)
    v = np.array(vols)
    return np.sum(c * v) / np.sum(v)

def calc_bollinger(closes, period=20):
    arr  = np.array(closes[-period:])
    mean = arr.mean()
    std  = arr.std()
    return mean + 2*std, mean, mean - 2*std

def calc_ema(closes, period):
    arr = np.array(closes)
    k   = 2 / (period + 1)
    ema = arr[0]
    for price in arr[1:]:
        ema = price * k + ema * (1 - k)
    return ema

def find_fvg(opens, highs, lows, closes):
    fvgs  = []
    ifvgs = []
    for i in range(2, len(closes)):
        if lows[i] > highs[i-2]:
            fvgs.append({"type": "bullish_fvg", "top": lows[i], "bottom": highs[i-2]})
        if highs[i] < lows[i-2]:
            fvgs.append({"type": "bearish_fvg", "top": lows[i-2], "bottom": highs[i]})
    current = closes[-1]
    for fvg in fvgs[-10:]:
        if fvg["bottom"] <= current <= fvg["top"]:
            ifvg_type = "bearish_ifvg" if fvg["type"] == "bullish_fvg" else "bullish_ifvg"
            ifvgs.append({"type": ifvg_type, "zone_top": fvg["top"], "zone_bottom": fvg["bottom"]})
    return fvgs[-5:], ifvgs

def find_order_blocks(opens, highs, lows, closes):
    obs = []
    for i in range(1, len(closes)-1):
        if closes[i] > opens[i] and closes[i+1] < opens[i+1]:
            if (opens[i+1] - closes[i+1]) > (closes[i] - opens[i]) * 1.5:
                obs.append({"type": "bearish_ob", "top": highs[i], "bottom": lows[i]})
        if closes[i] < opens[i] and closes[i+1] > opens[i+1]:
            if (closes[i+1] - opens[i+1]) > (opens[i] - closes[i]) * 1.5:
                obs.append({"type": "bullish_ob", "top": highs[i], "bottom": lows[i]})
    return obs[-5:]

def find_bos_choch(closes, highs, lows):
    signals = []
    if len(closes) < 20:
        return signals
    recent_high = max(highs[-20:-1])
    recent_low  = min(lows[-20:-1])
    current     = closes[-1]
    if current > recent_high:
        signals.append("BOS Bullish")
    if current < recent_low:
        signals.append("BOS Bearish")
    mid = len(closes) // 2
    first_trend  = closes[mid] - closes[0]
    second_trend = closes[-1] - closes[mid]
    if first_trend < 0 and second_trend > 0:
        signals.append("CHoCH Bullish flip")
    elif first_trend > 0 and second_trend < 0:
        signals.append("CHoCH Bearish flip")
    return signals

def find_liquidity_sweeps(highs, lows, closes):
    sweeps = []
    if len(closes) < 10:
        return sweeps
    prev_high = max(highs[-10:-1])
    prev_low  = min(lows[-10:-1])
    current   = closes[-1]
    if highs[-1] > prev_high and closes[-1] < prev_high:
        sweeps.append("Liquidity sweep above highs (bearish)")
    if lows[-1] < prev_low and closes[-1] > prev_low:
        sweeps.append("Liquidity sweep below lows (bullish)")
    return sweeps

def get_macro_context():
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "Federal Reserve interest rates inflation economy",
                "sortBy": "publishedAt",
                "pageSize": 3,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10
        )
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles[:3]]
    except:
        return []

def get_news_sentiment(query):
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": 5,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10
        )
        return [a["title"] for a in resp.json().get("articles", [])[:5]]
    except:
        return []

def get_ai_strategy(asset, price, rsi, macd, macd_signal, vwap, bb_upper, bb_lower,
                    ema9, ema21, fvgs, ifvgs, obs, bos_choch, liquidity, headlines, macro):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are an elite day trader and SMC analyst. Analyze {asset} and provide ONLY the best high-probability trade setups. Be strict — only signal STRONG BUY or STRONG SELL when multiple factors align perfectly. Otherwise signal NEUTRAL.

PRICE: ${price:,.2f}
TECHNICALS:
- RSI: {rsi:.1f} {'(overbought)' if rsi > 70 else '(oversold)' if rsi < 30 else '(neutral)'}
- MACD: {macd:.4f} vs Signal {macd_signal:.4f} → {'Bullish crossover' if macd > macd_signal else 'Bearish crossover'}
- EMA9: {ema9:.2f} | EMA21: {ema21:.2f} → {'Bullish' if ema9 > ema21 else 'Bearish'} trend
- VWAP: {vwap:.2f} → Price {'above' if price > vwap else 'below'} VWAP
- Bollinger: Upper {bb_upper:.2f} | Lower {bb_lower:.2f}

SMC ANALYSIS:
- IFVGs: {[f"{i['type']} {i['zone_bottom']:.2f}-{i['zone_top']:.2f}" for i in ifvgs] or 'None'}
- Order Blocks: {[f"{o['type']} {o['bottom']:.2f}-{o['top']:.2f}" for o in obs] or 'None'}
- Structure: {bos_choch or ['No major BOS/CHoCH']}
- Liquidity: {liquidity or ['None detected']}

NEWS (last 24h):
{chr(10).join(f'- {h}' for h in headlines[:3])}

MACRO:
{chr(10).join(f'- {m}' for m in macro[:2])}

Respond in EXACTLY this format:
SIGNAL: [STRONG BUY / STRONG SELL / NEUTRAL]
CONFIDENCE: [1-10]
ENTRY: $[price]
STOP LOSS: $[price]
TAKE PROFIT 1: $[price]
TAKE PROFIT 2: $[price]
RISK/REWARD: [ratio]
REASON: [2-3 sentences]
KEY RISK: [one sentence]
SMC SETUP: [describe the key SMC pattern driving this trade]"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def send_discord_alert(asset, price, strategy, ifvgs, liquidity):
    lines = strategy.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            k, v = line.split(':', 1)
            parsed[k.strip()] = v.strip()

    signal     = parsed.get('SIGNAL', 'NEUTRAL')
    confidence = parsed.get('CONFIDENCE', '?')
    entry      = parsed.get('ENTRY', '?')
    stop       = parsed.get('STOP LOSS', '?')
    tp1        = parsed.get('TAKE PROFIT 1', '?')
    tp2        = parsed.get('TAKE PROFIT 2', '?')
    rr         = parsed.get('RISK/REWARD', '?')
    reason     = parsed.get('REASON', '')
    risk       = parsed.get('KEY RISK', '')
    smc        = parsed.get('SMC SETUP', '')

    emoji = "🟢" if "BUY" in signal else "🔴"
    ifvg_text = f"\n⚡ **IFVG:** {ifvgs[0]['type']} zone ${ifvgs[0]['zone_bottom']:.2f}-${ifvgs[0]['zone_top']:.2f}" if ifvgs else ""
    liq_text  = f"\n💧 **Liquidity:** {liquidity[0]}" if liquidity else ""

    message = f"""{emoji} **{asset} — {signal}** | Confidence: {confidence}/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 **Price:** ${price:,.2f}
📍 **Entry:** {entry}
🛑 **Stop Loss:** {stop}
🎯 **TP1:** {tp1} | **TP2:** {tp2}
⚖️ **R/R:** {rr}{ifvg_text}{liq_text}

🧠 **SMC Setup:** {smc}
📝 **Reason:** {reason}
⚠️ **Risk:** {risk}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now(pytz.timezone('America/New_York')).strftime('%I:%M %p ET')}"""

    requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    print(f"  ✅ Discord alert sent!")

def analyze_asset(name, info, macro):
    print(f"\nAnalyzing {name}...")
    try:
        if info["type"] == "stock":
            if not is_market_open():
                print(f"  Market closed, skipping")
                return
            opens, highs, lows, closes, vols = get_stock_candles(info["symbol"])
        else:
            opens, highs, lows, closes, vols = get_crypto_candles(info["symbol"])

        if len(closes) < 30:
            print(f"  Not enough data")
            return

        price       = closes[-1]
        rsi         = calc_rsi(closes)
        macd, msig  = calc_macd(closes)
        vwap        = calc_vwap(closes, vols)
        bb_u, _, bb_l = calc_bollinger(closes)
        ema9        = calc_ema(closes, 9)
        ema21       = calc_ema(closes, 21)
        fvgs, ifvgs = find_fvg(opens, highs, lows, closes)
        obs         = find_order_blocks(opens, highs, lows, closes)
        bos_choch   = find_bos_choch(closes, highs, lows)
        liquidity   = find_liquidity_sweeps(highs, lows, closes)
        headlines   = get_news_sentiment(info["news_query"])

        print(f"  ${price:,.2f} | RSI:{rsi:.0f} | MACD:{'▲' if macd > msig else '▼'} | EMA:{'▲' if ema9 > ema21 else '▼'}")
        if ifvgs:    print(f"  ⚡ IFVG: {ifvgs[0]['type']}")
        if bos_choch: print(f"  📊 {bos_choch[0]}")
        if liquidity: print(f"  💧 {liquidity[0]}")

        strategy = get_ai_strategy(
            name, price, rsi, macd, msig, vwap, bb_u, bb_l,
            ema9, ema21, fvgs, ifvgs, obs, bos_choch, liquidity, headlines, macro
        )

        lines = strategy.strip().split('\n')
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

        print(f"  Signal: {signal} | Confidence: {conf}/10")

        if "STRONG" in signal and conf >= MIN_CONFIDENCE:
            send_discord_alert(name, price, strategy, ifvgs, liquidity)
        else:
            print(f"  No alert (need STRONG signal + confidence {MIN_CONFIDENCE}+)")

    except Exception as e:
        print(f"  Error: {e}")

def run():
    print("🤖 AI Day Trading Signal Bot")
    print(f"Watchlist: {list(WATCHLIST.keys())}")
    print(f"Min confidence: {MIN_CONFIDENCE}/10 | STRONG signals only")
    print(f"Scanning every 15 minutes")

    while True:
        et  = pytz.timezone('America/New_York')
        now = datetime.now(et)
        print(f"\n{'='*50}")
        print(f"Scanning... {now.strftime('%I:%M %p ET | %A')}")

        macro = get_macro_context()
        if macro:
            print(f"Macro: {macro[0][:60]}...")

        for name, info in WATCHLIST.items():
            analyze_asset(name, info, macro)
            time.sleep(3)

        print(f"\nNext scan in 15 min...")
        time.sleep(900)

if __name__ == "__main__":
    run()
