import os, json, time, math, random
from datetime import datetime, timedelta, timezone
from typing import List, Any

import gspread
import pandas as pd
import numpy as np
from coinbase.rest import RESTClient  # coinbase-advanced-py

# ---------- Config ----------
SHEET_NAME        = os.getenv("SHEET_NAME", "Trading Log")
PRODUCTS_TAB      = os.getenv("CRYPTO_PRODUCTS_TAB", "crypto_products")
SCREENER_TAB      = os.getenv("CRYPTO_SCREENER_TAB", "crypto_screener")
LOG_TAB           = os.getenv("CRYPTO_LOG_TAB", "crypto_log")

LOOKBACK_4H       = int(os.getenv("LOOKBACK_4H", "200"))
MIN_24H_NOTIONAL  = float(os.getenv("MIN_24H_NOTIONAL", "2000000"))
RSI_MIN           = float(os.getenv("RSI_MIN", "50"))
RSI_MAX           = float(os.getenv("RSI_MAX", "65"))
MAX_EXT_EMA20_PCT = float(os.getenv("MAX_EXT_EMA20_PCT", "0.08"))
REQUIRE_7D_HIGH   = os.getenv("REQUIRE_7D_HIGH", "true").lower() in ("1","true","yes")
PER_PRODUCT_SLEEP = float(os.getenv("PER_PRODUCT_SLEEP", "0.15"))

# ---------- Headers ----------
SCREENER_HEADERS = [
    "Product","Price","EMA_20","SMA_50","RSI_14",
    "MACD","Signal","MACD_Hist","MACD_Hist_Δ",
    "Vol24hUSD","7D_High","Breakout","Bullish Signal","Buy Reason","Timestamp"
]
LOG_HEADERS = ["Timestamp","Action","Product","ProceedsUSD","Qty","OrderID","Status","Note"]

# ---------- Small utils ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def g(obj: Any, *names: str, default=None):
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] not in (None, ""):
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v not in (None, ""):
                return v
    return default

def _floor_to_4h(dt_utc: datetime) -> datetime:
    dt_utc = dt_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return dt_utc.replace(hour=(dt_utc.hour // 4) * 4)

def dyn_decimals(price: float, base=6, cap=12) -> int:
    if price <= 0: return base
    mag = int(max(0, -math.floor(math.log10(price))))
    return min(cap, base + mag)

def r_prec(x: float, d: int): return round(float(x), d)

# ---------- Sheets helpers (creates only needed tabs) ----------
def get_google_client():
    raw = os.getenv("GOOGLE_CREDS_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_CREDS_JSON")
    return gspread.service_account_from_dict(json.loads(raw))

def _open_sheet(gc):
    try:
        return gc.open(SHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
        return gc.create(SHEET_NAME)

def _ensure_tab(sh, title: str, headers: List[str], clear_first: bool):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows="2000", cols="50")
    if clear_first:
        ws.clear()
    end_col = chr(ord('A') + len(headers) - 1)
    vals = ws.get_values(f"A1:{end_col}1")
    if not vals or vals[0] != headers:
        ws.update(range_name=f"A1:{end_col}1", values=[headers])
    try:
        ws.freeze(rows=1)
    except Exception:
        pass
    return ws

def ensure_tabs(gc):
    sh = _open_sheet(gc)
    ws_products = _ensure_tab(sh, PRODUCTS_TAB, ["Product"], clear_first=True)         # finder rewrites
    ws_screener = _ensure_tab(sh, SCREENER_TAB, SCREENER_HEADERS, clear_first=True)    # finder rewrites
    ws_log      = _ensure_tab(sh, LOG_TAB,      LOG_HEADERS,      clear_first=False)   # preserve history
    return ws_products, ws_screener, ws_log

def write_products(ws, products: List[str]):
    ws.clear()
    ws.update("A1:A1", [["Product"]])
    if products:
        ws.append_rows([[p] for p in products], value_input_option="USER_ENTERED")

def write_screener(ws, rows: List[List[Any]]):
    ws.clear()
    ws.append_row(SCREENER_HEADERS)
    for i in range(0, len(rows), 100):
        ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")

# ---------- Coinbase helpers ----------
CB = RESTClient()

def all_usd_products() -> List[str]:
    prods, cursor = [], None
    while True:
        resp = CB.get_products(limit=250, cursor=cursor) if cursor else CB.get_products(limit=250)
        products = g(resp, "products") or (resp if isinstance(resp, list) else [])
        for p in products:
            pid = g(p, "product_id", "productId", "id")
            if not pid:
                base = g(p, "base_currency", "baseCurrency", "base")
                quote = g(p, "quote_currency", "quoteCurrency", "quote")
                if base and quote:
                    pid = f"{base}-{quote}"
            status = (g(p, "status", "trading_status", "tradingStatus", default="") or "").upper()
            if pid and pid.endswith("-USD") and status == "ONLINE":
                prods.append(pid)
        cursor = g(resp, "cursor")
        if not cursor:
            break
    return sorted(dict.fromkeys(prods))

def _floor_endpoints_4h(bars:int):
    end_dt = _floor_to_4h(datetime.now(timezone.utc))
    start_dt = end_dt - timedelta(hours=bars * 4)
    return int(start_dt.timestamp()), int(end_dt.timestamp())

def get_candles_4h(product_id: str, bars: int) -> pd.DataFrame:
    start_epoch, end_epoch = _floor_endpoints_4h(bars)
    resp = CB.get_candles(product_id=product_id, start=start_epoch, end=end_epoch, granularity="FOUR_HOUR")
    rows = g(resp, "candles") or (resp if isinstance(resp, list) else [])
    if not rows:
        return pd.DataFrame()
    out = [{
        "start":  g(c, "start"),
        "open":   float(g(c, "open",   default=0) or 0),
        "high":   float(g(c, "high",   default=0) or 0),
        "low":    float(g(c, "low",    default=0) or 0),
        "close":  float(g(c, "close",  default=0) or 0),
        "volume": float(g(c, "volume", default=0) or 0),
    } for c in rows]
    df = pd.DataFrame(out).sort_values("start")
    return df.tail(bars).reset_index(drop=True)

# ---------- Indicators ----------
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

# ---------- Analyze one ----------
def analyze(product_id: str):
    df = get_candles_4h(product_id, LOOKBACK_4H)
    if df.empty or df.shape[0] < 60:
        return None

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

    vol24_usd = float((close.tail(6) * vol.tail(6)).sum())
    high_7d   = float(close.tail(42).max())
    breakout  = price >= high_7d - 1e-9

    # Filters
    if not (price > ema20 > sma50):
        return None
    if (price / ema20 - 1.0) > MAX_EXT_EMA20_PCT:
        return None
    if not (RSI_MIN < rsi14 < RSI_MAX):
        return None
    if not (macd_v > signal_v and hist_v > 0 and (not math.isnan(hist_delta) and hist_delta > 0)):
        return None
    if vol24_usd < MIN_24H_NOTIONAL:
        return None
    if REQUIRE_7D_HIGH and not breakout:
        return None

    reason = (
        f"4h Uptrend (P>EMA20>SMA50), RSI {RSI_MIN}-{RSI_MAX}, "
        f"MACD>Signal & Hist↑, ≤{int(MAX_EXT_EMA20_PCT*100)}% above EMA20, "
        f"24h notional ≥ ${int(MIN_24H_NOTIONAL):,}" + (" + 7D breakout" if REQUIRE_7D_HIGH else "")
    )

    d  = dyn_decimals(price)
    d2 = min(14, d + 2)

    return [
        product_id,
        r_prec(price, d),
        r_prec(ema20, d),
        r_prec(sma50, d),
        round(rsi14, 2),
        r_prec(macd_v, d),
        r_prec(signal_v, d),
        r_prec(hist_v, d),
        r_prec(hist_delta, d2) if not math.isnan(hist_delta) else "",
        int(vol24_usd),
        r_prec(high_7d, d),
        "✅" if breakout else "",
        "✅",
        reason,
        now_iso(),
    ]

# ---------- Main ----------
def main():
    print("🚀 crypto-finder starting (no cost tab)")
    gc = get_google_client()
    ws_products, ws_screener, ws_log = ensure_tabs(gc)

    products = all_usd_products()
    write_products(ws_products, products)
    print(f"📦 ONLINE USD products: {len(products)}")

    rows = []
    for i, pid in enumerate(products, 1):
        try:
            r = analyze(pid)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"⚠️ {pid} analyze error: {e}")
        if i % 20 == 0:
            print(f"   • analyzed {i}/{len(products)}")
        time.sleep(PER_PRODUCT_SLEEP * (0.8 + 0.4 * random.random()))

    write_screener(ws_screener, rows)
    print(f"✅ Screener wrote {len(rows)} picks to {SCREENER_TAB}")
    print("🧰 Tabs ensured:", PRODUCTS_TAB, SCREENER_TAB, LOG_TAB)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
