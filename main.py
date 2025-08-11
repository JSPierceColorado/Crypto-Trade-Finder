import os, json, time, math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import gspread
import pandas as pd
import numpy as np
from coinbase.rest import RESTClient  # pip install coinbase-advanced-py

# ============== Config ==============
SHEET_NAME        = os.getenv("SHEET_NAME", "Trading Log")
PRODUCTS_TAB      = os.getenv("CRYPTO_PRODUCTS_TAB", "crypto_products")
SCREENER_TAB      = os.getenv("CRYPTO_SCREENER_TAB", "crypto_screener")

LOOKBACK_4H       = int(os.getenv("LOOKBACK_4H", "200"))   # bars
MIN_24H_NOTIONAL  = float(os.getenv("MIN_24H_NOTIONAL", "2000000"))  # USD
RSI_MIN           = float(os.getenv("RSI_MIN", "50"))
RSI_MAX           = float(os.getenv("RSI_MAX", "65"))
MAX_EXT_EMA20_PCT = float(os.getenv("MAX_EXT_EMA20_PCT", "0.08"))
REQUIRE_7D_HIGH   = os.getenv("REQUIRE_7D_HIGH", "true").lower() in ("1","true","yes")

# ============== Sheets helpers ==============
def now_iso(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def get_google_client():
    raw = os.getenv("GOOGLE_CREDS_JSON")
    if not raw: raise RuntimeError("Missing GOOGLE_CREDS_JSON")
    return gspread.service_account_from_dict(json.loads(raw))

def _get_ws(gc, tab):
    sh = gc.open(SHEET_NAME)
    try: return sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab, rows="2000", cols="50")

def write_products(ws, products: List[str]):
    ws.clear()
    ws.append_row(["Product"])
    if products:
        ws.append_rows([[p] for p in products], value_input_option="USER_ENTERED")

def write_screener(ws, rows: List[List[Any]]):
    ws.clear()
    headers = ["Product","Price","EMA_20","SMA_50","RSI_14",
               "MACD","Signal","MACD_Hist","MACD_Hist_Δ",
               "Vol24hUSD","7D_High","Breakout","Bullish Signal","Buy Reason","Timestamp"]
    ws.append_row(headers)
    for i in range(0, len(rows), 100):
        ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")

# ============== Coinbase helpers ==============
CB = RESTClient()  # reads COINBASE_API_KEY / COINBASE_API_SECRET envs

def all_usd_products() -> List[str]:
    prods = []
    page = None
    while True:
        resp = CB.get_products(limit=250, cursor=page) if page else CB.get_products(limit=250)
        data = resp.get("products", resp) if isinstance(resp, dict) else resp.products  # be defensive
        for p in data:
            pid = p.get("product_id") or p.get("productId") or p.get("id")
            status = (p.get("status") or "").upper()
            if not pid: continue
            if not pid.endswith("-USD"): continue
            if status != "ONLINE": continue
            prods.append(pid)
        page = (resp.get("cursor") if isinstance(resp, dict) else getattr(resp, "cursor", None))
        if not page: break
    # unique, sorted
    seen, out = set(), []
    for x in prods:
        if x not in seen: seen.add(x); out.append(x)
    return sorted(out)

def get_candles_4h(product_id: str, bars: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=bars*4 + 12)  # pad a bit
    # Advanced Trade candles granularity strings often like: "FOUR_HOUR" / "ONE_DAY"
    resp = CB.get_candles(product_id=product_id, start=start.isoformat(), end=end.isoformat(), granularity="FOUR_HOUR")
    # Expect list of dicts with keys: start, low, high, open, close, volume
    rows = resp.get("candles", resp) if isinstance(resp, dict) else resp.candles
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize column names if needed
    for k in ["open","high","low","close","volume","start"]:
        if k not in df.columns and k.capitalize() in df.columns:
            df.rename(columns={k.capitalize(): k}, inplace=True)
    # Sort ascending by start
    if "start" in df.columns:
        df = df.sort_values("start")
    return df.tail(bars).reset_index(drop=True)

# ============== Indicators ==============
def ema(s, w): return s.ewm(span=w, adjust=False).mean()
def sma(s, w): return s.rolling(w).mean()
def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/window, adjust=False).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
def macd(series, fast=12, slow=26, signal=9):
    ef = ema(series, fast); es = ema(series, slow)
    line = ef - es; sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist

# ============== Analyze one ==============
def analyze(product_id: str):
    df = get_candles_4h(product_id, LOOKBACK_4H)
    if df.empty or df.shape[0] < 60: return None
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    vol   = pd.to_numeric(df["volume"], errors="coerce").astype(float)

    price = float(close.iloc[-1])
    ema20 = float(ema(close, 20).iloc[-1])
    sma50 = float(sma(close, 50).iloc[-1])
    rsi14 = float(rsi(close, 14).iloc[-1])

    macd_line, macd_sig, macd_hist = macd(close)
    macd_v    = float(macd_line.iloc[-1])
    signal_v  = float(macd_sig.iloc[-1])
    hist_v    = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2]) if macd_hist.shape[0] >= 2 else np.nan
    hist_delta= hist_v - hist_prev if not math.isnan(hist_prev) else np.nan

    # 24h notional = sum of last 6x4h (close * volume)
    vol24_usd = float((close.tail(6) * vol.tail(6)).sum())

    # 7D high = last 42x4h bars high of closes
    high_7d = float(close.tail(42).max())
    breakout = price >= high_7d - 1e-9

    # Filters
    if not (price > ema20 > sma50): return None
    if (price / ema20 - 1.0) > MAX_EXT_EMA20_PCT: return None
    if not (RSI_MIN < rsi14 < RSI_MAX): return None
    if not (macd_v > signal_v and hist_v > 0 and (not math.isnan(hist_delta) and hist_delta > 0)): return None
    if vol24_usd < MIN_24H_NOTIONAL: return None
    if REQUIRE_7D_HIGH and not breakout: return None

    reason = (
        f"4h Uptrend (P>EMA20>SMA50), RSI {RSI_MIN}-{RSI_MAX}, "
        f"MACD>Signal & Hist↑, ≤{int(MAX_EXT_EMA20_PCT*100)}% above EMA20, "
        f"24h notional ≥ ${int(MIN_24H_NOTIONAL):,}"
        + (" + 7D breakout" if REQUIRE_7D_HIGH else "")
    )
    row = [
        product_id, round(price, 6), round(ema20, 6), round(sma50, 6), round(rsi14, 2),
        round(macd_v, 6), round(signal_v, 6), round(hist_v, 6),
        round(hist_delta, 6) if not math.isnan(hist_delta) else "",
        int(vol24_usd), round(high_7d, 6),
        "✅" if breakout else "", "✅", reason, now_iso()
    ]
    return row

# ============== Main ==============
def main():
    print("🚀 crypto-finder starting")
    gc = get_google_client()
    ws_products = _get_ws(gc, PRODUCTS_TAB)
    ws_screener = _get_ws(gc, SCREENER_TAB)

    products = all_usd_products()
    write_products(ws_products, products)
    print(f"📦 ONLINE USD products: {len(products)}")

    rows = []
    for i, pid in enumerate(products, 1):
        try:
            r = analyze(pid)
            if r: rows.append(r)
        except Exception as e:
            print(f"⚠️ {pid} analyze error: {e}")
        if i % 20 == 0: print(f"   • analyzed {i}/{len(products)}")
        time.sleep(0.05)
    write_screener(ws_screener, rows)
    print(f"✅ Screener wrote {len(rows)} picks to {SCREENER_TAB}")

if __name__ == "__main__":
    main()
