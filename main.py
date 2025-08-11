import os, json, time, math
from datetime import datetime, timedelta, timezone
from typing import List, Any

import gspread
import pandas as pd
import numpy as np
from coinbase.rest import RESTClient  # coinbase-advanced-py


# ============== Config ==============
SHEET_NAME        = os.getenv("SHEET_NAME", "Trading Log")
PRODUCTS_TAB      = os.getenv("CRYPTO_PRODUCTS_TAB", "crypto_products")
SCREENER_TAB      = os.getenv("CRYPTO_SCREENER_TAB", "crypto_screener")

LOOKBACK_4H       = int(os.getenv("LOOKBACK_4H", "200"))                 # number of 4h bars
MIN_24H_NOTIONAL  = float(os.getenv("MIN_24H_NOTIONAL", "2000000"))      # USD
RSI_MIN           = float(os.getenv("RSI_MIN", "50"))
RSI_MAX           = float(os.getenv("RSI_MAX", "65"))
MAX_EXT_EMA20_PCT = float(os.getenv("MAX_EXT_EMA20_PCT", "0.08"))        # 8%
REQUIRE_7D_HIGH   = os.getenv("REQUIRE_7D_HIGH", "true").lower() in ("1","true","yes")


# ============== Small utils ==============
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def g(obj: Any, *names: str, default=None):
    """Get the first present attribute/key from an object or dict."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj and obj[n] not in (None, ""):
                return obj[n]
        else:
            v = getattr(obj, n, None)
            if v not in (None, ""):
                return v
    return default


# ============== Sheets helpers ==============
def get_google_client():
    raw = os.getenv("GOOGLE_CREDS_JSON")
    if not raw:
        raise RuntimeError("Missing GOOGLE_CREDS_JSON")
    return gspread.service_account_from_dict(json.loads(raw))


def _get_ws(gc, tab_name: str):
    sh = gc.open(SHEET_NAME)
    try:
        return sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=tab_name, rows="2000", cols="50")


def write_products(ws, products: List[str]):
    ws.clear()
    ws.append_row(["Product"])
    if products:
        ws.append_rows([[p] for p in products], value_input_option="USER_ENTERED")


def write_screener(ws, rows: List[List[Any]]):
    ws.clear()
    headers = [
        "Product","Price","EMA_20","SMA_50","RSI_14",
        "MACD","Signal","MACD_Hist","MACD_Hist_Δ",
        "Vol24hUSD","7D_High","Breakout","Bullish Signal","Buy Reason","Timestamp"
    ]
    ws.append_row(headers)
    for i in range(0, len(rows), 100):
        ws.append_rows(rows[i:i+100], value_input_option="USER_ENTERED")


# ============== Coinbase helpers ==============
CB = RESTClient()  # reads COINBASE_API_KEY / COINBASE_API_SECRET from env


def all_usd_products() -> List[str]:
    """All ONLINE *-USD products from Coinbase Advanced (unique, sorted)."""
    prods = []
    cursor = None
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
    # de-dup & sort
    return sorted(dict.fromkeys(prods))


def get_candles_4h(product_id: str, bars: int) -> pd.DataFrame:
    """Fetch 4H candles and return a clean ascending DataFrame."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=bars * 4 + 12)  # padding
    resp = CB.get_candles(
        product_id=product_id,
        start=start.isoformat(),
        end=end.isoformat(),
        granularity="FOUR_HOUR",
    )
    rows = g(resp, "candles") or (resp if isinstance(resp, list) else [])
    out = []
    for c in rows:
        out.append({
            "start":  g(c, "start"),
            "open":   float(g(c, "open",   default=0) or 0),
            "high":   float(g(c, "high",   default=0) or 0),
            "low":    float(g(c, "low",    default=0) or 0),
            "close":  float(g(c, "close",  default=0) or 0),
            "volume": float(g(c, "volume", default=0) or 0),
        })
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out).sort_values("start")
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

    # 24h notional = sum of last 6x4h (close * volume)
    vol24_usd = float((close.tail(6) * vol.tail(6)).sum())

    # 7D high (42 x 4h bars) based on closes
    high_7d = float(close.tail(42).max())
    breakout = price >= high_7d - 1e-9

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
        f"24h notional ≥ ${int(MIN_24H_NOTIONAL):,}"
        + (" + 7D breakout" if REQUIRE_7D_HIGH else "")
    )

    row = [
        product_id,
        round(price, 6),
        round(ema20, 6),
        round(sma50, 6),
        round(rsi14, 2),
        round(macd_v, 6),
        round(signal_v, 6),
        round(hist_v, 6),
        round(hist_delta, 6) if not math.isnan(hist_delta) else "",
        int(vol24_usd),
        round(high_7d, 6),
        "✅" if breakout else "",
        "✅",
        reason,
        now_iso(),
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
            if r:
                rows.append(r)
        except Exception as e:
            print(f"⚠️ {pid} analyze error: {e}")
        if i % 20 == 0:
            print(f"   • analyzed {i}/{len(products)}")
        time.sleep(0.05)

    write_screener(ws_screener, rows)
    print(f"✅ Screener wrote {len(rows)} picks to {SCREENER_TAB}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Fatal error:", e)
        traceback.print_exc()
