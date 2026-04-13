import time
import requests
import numpy as np
from datetime import datetime
import anthropic

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1493061685349584948/lqJsz0Bov6a67p9mvz8OTyzHH-5QguAwWrsSitEVmr2kvOe5ebA0YD1GockTUZJqbxxi"
NEWS_API_KEY = "ce3679f211b6484ca94be0db9022d7d5"

WATCHLIST = {
    "BTC": {"type": "crypto", "symbol": "BTCUSDT", "news_query": "Bitcoin BTC crypto"},
    "ETH": {"type": "crypto", "symbol": "ETHUSDT", "news_query": "Ethereum ETH crypto"},
    "NVDA": {"type": "stock", "symbol": "NVDA", "news_query": "NVIDIA NVDA stock"},
    "AAPL": {"type": "stock", "symbol": "AAPL", "news_query": "Apple AAPL stock"},
}

def get_crypto_candles(symbol, interval="15m", limit=100):
    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10
    )
    candles = resp.json()
    opens  = [float(c[1]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[3]) for c in candles]
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
    opens  = result["indicators"]["quote"][0]["open"]
    highs  = result["indicators"]["quote"][0]["high"]
    lows   = result["indicators"]["quote"][0]["low"]
    closes = result["indicators"]["quote"][0]["close"]
    vols   = result["indicators"]["quote"][0]["volume"]
    opens  = [x for x in opens if x is not None]
    highs  = [x for x in highs if x is not None]
    lows   = [x for x in lows if x is not None]
    closes = [x for x in closes if x is not None]
    vols   = [x for x in vols if x is not None]
    return opens, highs, lows, closes, vols

def calc_rsi(closes, period=14):
    diffs = np.diff(closes)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(closes):
    closes = np.array(closes)
    ema12 = closes[-12:].mean()
    ema26 = closes[-26:].mean() if len(closes) >= 26 else closes.mean()
    macd = ema12 - ema26
    signal = closes[-9:].mean() - closes[-18:].mean() if len(closes) >= 18 else 0
    return macd, signal

def calc_vwap(closes, vols):
    closes = np.array(closes)
    vols = np.array(vols)
    return np.sum(closes * vols) / np.sum(vols)

def calc_bollinger(closes, period=20):
    arr = np.array(closes[-period:])
    mean = arr.mean()
    std = arr.std()
    return mean + 2*std, mean, mean - 2*std

def find_fvg(opens, highs, lows, closes):
    """Find Fair Value Gaps and Inverse Fair Value Gaps."""
    fvgs = []
    ifvgs = []
    
    for i in range(2, len(closes)):
        # Bullish FVG: gap between candle[i-2] low and candle[i] high
        if lows[i] > highs[i-2]:
            fvgs.append({
                "type": "bullish_fvg",
                "top": lows[i],
                "bottom": highs[i-2],
                "index": i
            })
        
        # Bearish FVG: gap between candle[i-2] high and candle[i] low
        if highs[i] < lows[i-2]:
            fvgs.append({
                "type": "bearish_fvg",
                "top": lows[i-2],
                "bottom": highs[i],
                "index": i
            })
    
    # IFVG: price has returned to fill an FVG (inverting it)
    current_price = closes[-1]
    for fvg in fvgs[-10:]:  # Check last 10 FVGs
        if fvg["type"] == "bullish_fvg":
            # If price drops back into bullish FVG, it becomes bearish IFVG
            if fvg["bottom"] <= current_price <= fvg["top"]:
                ifvgs.append({
                    "type": "bearish_ifvg",
                    "zone_top": fvg["top"],
                    "zone_bottom": fvg["bottom"],
                    "current_price": current_price
                })
        elif fvg["type"] == "bearish_fvg":
            # If price rises back into bearish FVG, it becomes bullish IFVG
            if fvg["bottom"] <= current_price <= fvg["top"]:
                ifvgs.append({
                    "type": "bullish_ifvg",
                    "zone_top": fvg["top"],
                    "zone_bottom": fvg["bottom"],
                    "current_price": current_price
                })
    
    return fvgs[-5:], ifvgs

def find_order_blocks(opens, highs, lows, closes):
    """Find bullish and bearish order blocks."""
    obs = []
    for i in range(1, len(closes)-1):
        # Bearish OB: last up candle before strong down move
        if closes[i] > opens[i] and closes[i+1] < opens[i+1]:
            if (opens[i+1] - closes[i+1]) > (closes[i] - opens[i]) * 1.5:
                obs.append({
                    "type": "bearish_ob",
                    "top": highs[i],
                    "bottom": lows[i],
                    "index": i
                })
        # Bullish OB: last down candle before strong up move
        if closes[i] < opens[i] and closes[i+1] > opens[i+1]:
            if (closes[i+1] - opens[i+1]) > (opens[i] - closes[i]) * 1.5:
                obs.append({
                    "type": "bullish_ob",
                    "top": highs[i],
                    "bottom": lows[i],
                    "index": i
                })
    return obs[-5:]

def find_bos_choch(closes, highs, lows):
    """Detect Break of Structure and Change of Character."""
    signals = []
    if len(closes) < 20:
        return signals
    
    recent_high = max(highs[-20:-1])
    recent_low = min(lows[-20:-1])
    current = closes[-1]
    
    if current > recent_high:
        signals.append("BOS Bullish — broke above recent high")
    if current < recent_low:
        signals.append("BOS Bearish — broke below recent low")
    
    # CHoCH: recent swing structure change
    mid = len(closes) // 2
    first_half_trend = closes[mid] - closes[0]
    second_half_trend = closes[-1] - closes[mid]
    
    if first_half_trend < 0 and second_half_trend > 0:
        signals.append("CHoCH — bearish to bullish flip")
    elif first_half_trend > 0 and second_half_trend < 0:
        signals.append("CHoCH — bullish to bearish flip")
    
    return signals

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
        articles = resp.json().get("articles", [])
        headlines = [a["title"] for a in articles[:5]]
        return headlines
    except Exception as e:
        return []

def get_ai_strategy(asset, price, rsi, macd, macd_signal, vwap, bb_upper, bb_lower, fvgs, ifvgs, obs, bos_choch, headlines):
    """Use Claude AI to generate a trading strategy."""
    client = anthropic.Anthropic()
    
    prompt = f"""You are an expert day trader analyzing {asset} for a trade signal.

PRICE DATA:
- Current Price: ${price:,.2f}
- RSI (14): {rsi:.1f}
- MACD: {macd:.4f} | Signal: {macd_signal:.4f} | {'Bullish' if macd > macd_signal else 'Bearish'}
- VWAP: ${vwap:,.2f} | Price is {'above' if price > vwap else 'below'} VWAP
- Bollinger Upper: ${bb_upper:,.2f} | Lower: ${bb_lower:,.2f}

SMC ANALYSIS:
- Recent FVGs: {[f"{f['type']} at {f.get('top', 0):.2f}-{f.get('bottom', 0):.2f}" for f in fvgs]}
- Active IFVGs: {[f"{i['type']} zone {i.get('zone_bottom', 0):.2f}-{i.get('zone_top', 0):.2f}" for i in ifvgs]}
- Order Blocks: {[f"{o['type']} at {o.get('bottom', 0):.2f}-{o.get('top', 0):.2f}" for o in obs]}
- Structure: {bos_choch if bos_choch else ['No major BOS/CHoCH']}

NEWS HEADLINES:
{chr(10).join(f'- {h}' for h in headlines[:3])}

Based on this data, provide a day trading strategy in this exact format:
SIGNAL: [STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL]
CONFIDENCE: [1-10]
ENTRY: $[price]
STOP LOSS: $[price]
TAKE PROFIT: $[price]
REASON: [2-3 sentence explanation combining technicals, SMC, and news]
KEY RISK: [main risk to watch]"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

def send_discord_alert(asset, price, strategy, ifvgs):
    """Send a formatted alert to Discord."""
    lines = strategy.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            key, val = line.split(':', 1)
            parsed[key.strip()] = val.strip()
    
    signal = parsed.get('SIGNAL', 'NEUTRAL')
    confidence = parsed.get('CONFIDENCE', '?')
    entry = parsed.get('ENTRY', '?')
    stop = parsed.get('STOP LOSS', '?')
    target = parsed.get('TAKE PROFIT', '?')
    reason = parsed.get('REASON', '')
    risk = parsed.get('KEY RISK', '')

    emoji = "🚨" if "STRONG" in signal else "📊"
    color = 3066993 if "BUY" in signal else 15158332

    ifvg_text = ""
    if ifvgs:
        ifvg_text = f"\n⚡ **IFVG Active:** {ifvgs[0]['type']} zone ${ifvgs[0]['zone_bottom']:.2f}-${ifvgs[0]['zone_top']:.2f}"

    message = f"""{emoji} **{asset} — {signal}** | Confidence: {confidence}/10
━━━━━━━━━━━━━━━━━━━━━━
💰 **Current Price:** ${price:,.2f}
📈 **Entry:** {entry}
🛑 **Stop Loss:** {stop}
✅ **Take Profit:** {target}{ifvg_text}

📝 **Strategy:** {reason}
⚠️ **Risk:** {risk}
━━━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%I:%M %p ET')}"""

    requests.post(DISCORD_WEBHOOK, json={"content": message}, timeout=10)
    print(f"Sent Discord alert for {asset}")

def analyze_asset(name, info):
    print(f"\nAnalyzing {name}...")
    try:
        if info["type"] == "crypto":
            opens, highs, lows, closes, vols = get_crypto_candles(info["symbol"])
        else:
            opens, highs, lows, closes, vols = get_stock_candles(info["symbol"])

        if not closes:
            print(f"No data for {name}")
            return

        price = closes[-1]
        rsi = calc_rsi(closes)
        macd, macd_signal = calc_macd(closes)
        vwap = calc_vwap(closes, vols)
        bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
        fvgs, ifvgs = find_fvg(opens, highs, lows, closes)
        obs = find_order_blocks(opens, highs, lows, closes)
        bos_choch = find_bos_choch(closes, highs, lows)
        headlines = get_news_sentiment(info["news_query"])

        print(f"  Price: ${price:,.2f} | RSI: {rsi:.1f} | MACD: {'Bull' if macd > macd_signal else 'Bear'}")
        if ifvgs:
            print(f"  ⚡ IFVG detected: {ifvgs[0]['type']}")
        if bos_choch:
            print(f"  📊 Structure: {bos_choch[0]}")

        strategy = get_ai_strategy(
            name, price, rsi, macd, macd_signal, vwap,
            bb_upper, bb_lower, fvgs, ifvgs, obs, bos_choch, headlines
        )

        print(f"  Strategy: {strategy[:100]}...")

        # Only alert on confidence 7+
        if any(word in strategy for word in ["STRONG BUY", "STRONG SELL", "BUY", "SELL"]):
            confidence_line = [l for l in strategy.split('\n') if 'CONFIDENCE' in l]
            if confidence_line:
                try:
                    conf = int(''.join(filter(str.isdigit, confidence_line[0])))
                    if conf >= 7:
                        send_discord_alert(name, price, strategy, ifvgs)
                except:
                    pass

    except Exception as e:
        print(f"Error analyzing {name}: {e}")

def run():
    print("Starting AI Day Trading Signal Bot")
    print(f"Watchlist: {list(WATCHLIST.keys())}")
    print(f"Scanning every 15 minutes")
    print(f"Alerts sent to Discord when confidence >= 7/10")
    
    while True:
        print(f"\n{'='*50}")
        print(f"Scanning... {datetime.now().strftime('%I:%M %p ET')}")
        
        for name, info in WATCHLIST.items():
            analyze_asset(name, info)
            time.sleep(3)
        
        print(f"\nNext scan in 15 minutes...")
        time.sleep(900)

if __name__ == "__main__":
    run()
