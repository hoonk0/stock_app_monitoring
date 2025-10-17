from __future__ import annotations
"""
Expected Move Scanner (NASDAQ / S&P500)
--------------------------------------
- Trading logic removed: computes Expected Move (EM) per symbol and prints a table.
- Two EM formulas provided:
  1) Straddle: mid(Call ATM) + mid(Put ATM) ≈ near-term move implied by options
  2) IV formula: S * IV_atm * sqrt(T)  (T in years)

Usage:
$ python f_search_oversea_expectedmove.py

Defaults:
- UNIVERSE: 'nasdaq' | 'sp500'
- TOP_MKT_CAP: scan top N by market cap
- NEAREST_EXPIRY_INDEX: 0 (nearest), 1 (next), ...
- OVERRIDE_TICKERS: set list to test specific tickers only

Notes:
- Depending on yfinance data quality/latency, bid/ask can be empty, fallback applies.
- If yahoo_fin is available we fetch NASDAQ full list, otherwise fallback to NASDAQ-100.
"""

import concurrent.futures as cf
import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import json
import os
import subprocess
from zoneinfo import ZoneInfo

from datetime import datetime, timezone

# ===== Settings =====
UNIVERSE: str = "nasdaq"       # 'nasdaq' | 'sp500'
TOP_MKT_CAP: int = 500         # scan top N by market cap
NEAREST_EXPIRY_INDEX: int = 0  # 0=nearest, 1=next...
MAX_WORKERS: int = 8           # concurrency
REQUEST_DELAY_SEC: float = 0.03 # gap between ticker requests (civility)

# Price basis for EM bands: 'last' (live) or 'prev_close' (many sites use previous close)
PRICE_BASIS: str = "prev_close"  # "last" | "prev_close"
# Linear interpolation of straddle between nearest strikes (if not using common strike)
STRADDLE_INTERP: bool = True

# EM accuracy options
STRICT_MID_ONLY: bool = False       # True: use mid only; False: fallback to last/bid/ask allowed
USE_COMMON_ATM_STRIKE: bool = True  # True: use common strike K* where both call/put exist

# ===== Pre/Post-market monitoring =====
USE_PREPOST: bool = True
US_MARKET_TZ = ZoneInfo("America/New_York")

# ===== Skew mode =====
SKEW_MODE: bool = False
SKEW_T1: float = -6.0
SKEW_T2: float = -10.0
SKEW_BUF_T1: float = 0.03
SKEW_BUF_T2: float = 0.02
SKEW_STOP_T1: float = 0.12
SKEW_STOP_T2: float = 0.15
SKEW_SIZE_T1: float = 0.7
SKEW_SIZE_T2: float = 0.5

# ===== Long-only parameters =====
ENTRY_BAND_BUFFER_PCT: float = 0.05  # allow entry up to EM*5% above lower band
STOP_BEYOND_EM_PCT: float = 0.10     # stop if price breaches EM lower by extra 10%

# Test specific symbols only (e.g., ["AAPL","MSFT","NVDA"])
OVERRIDE_TICKERS: Optional[List[str]] = None

# ----- yahoo_fin fallback -----
try:
    from yahoo_fin import stock_info as si
    HAVE_YFIN = True
except Exception:
    HAVE_YFIN = False

# ===== Utils =====

def _fetch_html_with_ua(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text

def _read_html_tables(url: str) -> list[pd.DataFrame]:
    html = _fetch_html_with_ua(url)
    return pd.read_html(StringIO(html))

def _clean_symbol(sym: str) -> str:
    return sym.replace(".", "-").strip()

# --- Pre/Post-market helpers ---

def _us_session_status(now_utc: Optional[datetime] = None) -> str:
    """Return one of: 'pre', 'regular', 'post', 'closed' (US ET)."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    et = now_utc.astimezone(US_MARKET_TZ)
    wd = et.weekday()  # 0=Mon ... 6=Sun
    if wd >= 5:
        return "closed"
    t = et.time()
    if t >= datetime.strptime("04:00", "%H:%M").time() and t < datetime.strptime("09:30", "%H:%M").time():
        return "pre"
    if t >= datetime.strptime("09:30", "%H:%M").time() and t < datetime.strptime("16:00", "%H:%M").time():
        return "regular"
    if t >= datetime.strptime("16:00", "%H:%M").time() and t < datetime.strptime("20:00", "%H:%M").time():
        return "post"
    return "closed"

def safe_price_with_prepost(tk: yf.Ticker) -> Tuple[Optional[float], str]:
    """
    Returns (price, session). If USE_PREPOST and in pre/post session,
    try pre/post price from fast_info/info; else fallback to live/prev mechanisms.
    """
    status = _us_session_status()
    fi = getattr(tk, 'fast_info', None)
    try:
        if USE_PREPOST and status == "pre":
            pre_p = getattr(fi, "pre_market_price", None) if fi else None
            if pre_p:
                return float(pre_p), "pre"
        if USE_PREPOST and status == "post":
            post_p = getattr(fi, "post_market_price", None) if fi else None
            if post_p:
                return float(post_p), "post"
    except Exception:
        pass

    try:
        info = tk.info
        if USE_PREPOST and status == "pre" and isinstance(info, dict):
            pre_p = info.get("preMarketPrice")
            if pre_p:
                return float(pre_p), "pre"
        if USE_PREPOST and status == "post" and isinstance(info, dict):
            post_p = info.get("postMarketPrice")
            if post_p:
                return float(post_p), "post"
    except Exception:
        pass

    live = safe_current_price(tk)
    if live is not None:
        return float(live), ("regular" if status == "regular" else "closed")
    return None, ("regular" if status == "regular" else "closed")

def get_sp500_tickers() -> List[str]:
    try:
        if hasattr(yf, "tickers_sp500"):
            lst = yf.tickers_sp500()
            if lst:
                return [_clean_symbol(t) for t in lst]
    except Exception:
        pass
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_tables(url)
    if not tables:
        raise RuntimeError("Failed to fetch S&P 500 table.")
    df = tables[0]
    col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
    tickers = [_clean_symbol(str(s)) for s in df[col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t.lower() != 'nan']
    if not tickers:
        tickers = [
            'AAPL','MSFT','NVDA','AMZN','GOOGL','META','BRK-B','LLY','AVGO','JPM',
            'TSLA','XOM','V','WMT','UNH','MA','JNJ','PG','HD','COST',
            'ABBV','BAC','CVX','MRK','PEP','KO','ADBE','NFLX','CRM','CSCO'
        ]
    return tickers

def get_nasdaq_tickers() -> List[str]:
    try:
        if hasattr(yf, "tickers_nasdaq"):
            lst = yf.tickers_nasdaq()
            if lst:
                return [_clean_symbol(t) for t in lst]
    except Exception:
        pass
    if HAVE_YFIN:
        try:
            tickers = si.tickers_nasdaq()
            return [t.replace('.', '-').strip() for t in tickers if isinstance(t, str) and t.strip()]
        except Exception:
            pass
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = _read_html_tables(url)
    for tb in tables:
        cols = [str(c) for c in tb.columns]
        if 'Ticker' in cols or 'Symbol' in cols:
            col = 'Ticker' if 'Ticker' in cols else 'Symbol'
            tickers = [str(s) for s in tb[col].tolist()]
            return [s.replace('.', '-').strip() for s in tickers if s and s.lower() != 'nan']
    return [
        'AAPL','MSFT','NVDA','AMZN','META','GOOGL','AVGO','TSLA','PEP','COST',
        'ADBE','CSCO','NFLX','AMD','INTC','PYPL','QCOM','TXN','AMAT','INTU',
        'SBUX','CHTR','MU','BKNG','LRCX','HON','MDLZ','PDD','ABNB','REGN'
    ]

def safe_current_price(tk: yf.Ticker) -> Optional[float]:
    try:
        fi = getattr(tk, 'fast_info', None)
        if fi and getattr(fi, 'last_price', None):
            return float(fi.last_price)
    except Exception:
        pass
    try:
        info = tk.info
        if info and 'currentPrice' in info and info['currentPrice']:
            return float(info['currentPrice'])
    except Exception:
        pass
    try:
        hist = tk.history(period="2d", interval="1d", auto_adjust=False)
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None

def safe_previous_close(tk: yf.Ticker) -> Optional[float]:
    try:
        fi = getattr(tk, 'fast_info', None)
        if fi and getattr(fi, 'previous_close', None):
            return float(fi.previous_close)
    except Exception:
        pass
    try:
        hist = tk.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty and 'Close' in hist.columns:
            # timezone-aware "today"
            if len(hist) >= 2 and hist.index[-1].date() == datetime.now(timezone.utc).date():
                return float(hist['Close'].iloc[-2])
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None

def price_by_basis(tk: yf.Ticker, fallback_to_last: bool = True) -> Optional[float]:
    if PRICE_BASIS == "prev_close":
        p = safe_previous_close(tk)
        if p is not None:
            return p
        if fallback_to_last:
            return safe_current_price(tk)
        return None
    return safe_current_price(tk)

def safe_market_cap(tk: yf.Ticker) -> int:
    try:
        info = tk.info
        mc = info.get('marketCap', 0) if isinstance(info, dict) else 0
        return 0 if mc is None else int(mc)
    except Exception:
        return 0

def universe_top_by_mktcap(source: str, top_n: int) -> List[str]:
    tickers = OVERRIDE_TICKERS if OVERRIDE_TICKERS else (get_nasdaq_tickers() if source.lower()=="nasdaq" else get_sp500_tickers())
    rows: List[tuple[str,int]] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fut_map = {}
        for i, t in enumerate(tickers):
            if REQUEST_DELAY_SEC > 0 and i:
                time.sleep(REQUEST_DELAY_SEC)
            tk = yf.Ticker(t)
            fu = ex.submit(safe_market_cap, tk)
            fut_map[fu] = t
        for fu in cf.as_completed(list(fut_map.keys())):
            t = fut_map[fu]
            try:
                mc = fu.result()
            except Exception:
                mc = 0
            rows.append((t, mc))
    rows.sort(key=lambda x: x[1], reverse=True)
    top = [t for t, mc in rows if mc > 0][:top_n]
    return top

# ===== Expected Move =====

@dataclass
class EMRow:
    symbol: str
    price: float          # unified current price (pre/regular/post or fallback)
    price_ref: float      # basis used for EM calc (prev_close or last)
    expiry: str
    dte: float
    atm_strike: float
    call_mid: float
    put_mid: float
    straddle_mid: float
    em_straddle: float
    em_iv: float
    em_pct_straddle: float
    em_pct_iv: float
    lower_straddle: float
    upper_straddle: float
    lower_iv: float
    upper_iv: float
    iv_atm: float
    skew_rr5: float
    session: str

def _mid_or_last(row: pd.Series) -> float:
    bid = row.get('bid', np.nan)
    ask = row.get('ask', np.nan)
    last = row.get('lastPrice', np.nan)
    try:
        bid = float(bid) if not pd.isna(bid) else np.nan
        ask = float(ask) if not pd.isna(ask) else np.nan
        last = float(last) if not pd.isna(last) else np.nan
    except Exception:
        bid = ask = last = np.nan
    if not np.isnan(bid) and not np.isnan(ask) and ask > 0:
        return (bid + ask) / 2.0
    if STRICT_MID_ONLY:
        return float('nan')
    if not np.isnan(last) and last > 0:
        return last
    if not np.isnan(bid) and bid > 0:
        return bid
    if not np.isnan(ask) and ask > 0:
        return ask
    return float('nan')

def _days_to_expiry_yr(expiry_str: str) -> float:
    try:
        dt_exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0.0, (dt_exp - now).total_seconds() / 86400.0)
        return days / 365.0
    except Exception:
        return float('nan')

def _safe_option_chain(tk: yf.Ticker, expiry: str, retries: int = 3, pause: float = 0.6):
    last_err = None
    for i in range(retries):
        try:
            return tk.option_chain(expiry)
        except Exception as e:
            last_err = e
            time.sleep(pause)
    raise last_err if last_err else RuntimeError("option_chain failed")

def _nearest_iv(df: pd.DataFrame, target_strike: float) -> float:
    if df is None or df.empty or 'strike' not in df.columns or 'impliedVolatility' not in df.columns:
        return float('nan')
    row = df.iloc[(df['strike'] - target_strike).abs().argsort()].iloc[0]
    try:
        return float(row.get('impliedVolatility', float('nan')))
    except Exception:
        return float('nan')

def compute_expected_move(symbol: str, expiry_idx: int = 0) -> Optional[EMRow]:
    try:
        tk = yf.Ticker(symbol)
        expiries = tk.options or []
        if not expiries:
            return None
        expiry = expiries[min(expiry_idx, len(expiries)-1)]
        chain = _safe_option_chain(tk, expiry)
        calls, puts = chain.calls, chain.puts

        # basis for EM bands
        price_ref = price_by_basis(tk)

        # unified current price (pre/regular/post → session price; else fallback to basis)
        session_price, session_str = safe_price_with_prepost(tk)
        if USE_PREPOST and session_str in ("pre", "regular", "post") and session_price is not None and not np.isnan(session_price):
            price = float(session_price)
        elif price_ref is not None and not np.isnan(price_ref):
            price = float(price_ref)
        else:
            return None

        # pick common strike near basis price_ref for ATM selection
        def _nearest_common_strike(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ref_price: float) -> Optional[float]:
            if calls_df is None or calls_df.empty or puts_df is None or puts_df.empty:
                return None
            c_strikes = pd.Series(calls_df['strike']).astype(float)
            p_strikes = pd.Series(puts_df['strike']).astype(float)
            common = np.intersect1d(c_strikes.values, p_strikes.values)
            if common.size == 0:
                return None
            idx = np.argmin(np.abs(common - ref_price))
            return float(common[idx])

        # EM 밴드는 기준가(PRICE_BASIS)로 잡는 게 일반적 → ATM 탐색에는 price_ref 사용
        ref_for_atm = float(price_ref) if (price_ref is not None and not np.isnan(price_ref)) else float(price)
        K = _nearest_common_strike(calls, puts, ref_for_atm)
        if K is None:
            return None
        call_atm = calls.loc[(calls['strike'].astype(float) == K)].head(1)
        put_atm  = puts.loc[(puts['strike'].astype(float) == K)].head(1)
        if call_atm.empty or put_atm.empty:
            return None
        call_atm = call_atm.iloc[0]
        put_atm  = put_atm.iloc[0]

        atm_strike = float(call_atm['strike'])
        call_mid = _mid_or_last(call_atm)
        put_mid  = _mid_or_last(put_atm)

        if np.isnan(call_mid) or np.isnan(put_mid):
            if STRICT_MID_ONLY:
                return None

        straddle = float(call_mid) + float(put_mid)

        # Optional interpolation between strikes (only if not using common strike)
        if not USE_COMMON_ATM_STRIKE and STRADDLE_INTERP and 'strike' in calls.columns and len(calls) > 1 and 'strike' in puts.columns and len(puts) > 1:
            try:
                c_sorted = calls.sort_values('strike').reset_index(drop=True)
                p_sorted = puts.sort_values('strike').reset_index(drop=True)
                ci_low = max(0, int((c_sorted['strike'] <= ref_for_atm).sum() - 1))
                ci_high = min(len(c_sorted)-1, ci_low + 1)
                pi_low = max(0, int((p_sorted['strike'] <= ref_for_atm).sum() - 1))
                pi_high = min(len(p_sorted)-1, pi_low + 1)
                k1 = float(c_sorted.loc[ci_low, 'strike'])
                k2 = float(c_sorted.loc[ci_high, 'strike'])
                if k2 > k1 and (k1 <= ref_for_atm <= k2):
                    def mid_at(df_, idx_): return _mid_or_last(df_.loc[idx_])
                    s1 = mid_at(c_sorted, ci_low) + mid_at(p_sorted, pi_low)
                    s2 = mid_at(c_sorted, ci_high) + mid_at(p_sorted, pi_high)
                    w = (ref_for_atm - k1) / (k2 - k1)
                    straddle = (1-w) * s1 + w * s2
                    call_mid = float('nan'); put_mid = float('nan')
            except Exception:
                pass

        T_years = _days_to_expiry_yr(expiry)
        iv_atm = np.nan
        try:
            if 'impliedVolatility' in calls.columns:
                iv_atm = float(calls.iloc[(calls['strike'] - ref_for_atm).abs().argsort()]['impliedVolatility'].iloc[0])
            if (np.isnan(iv_atm)) and ('impliedVolatility' in puts.columns):
                iv_atm = float(puts.iloc[(puts['strike'] - ref_for_atm).abs().argsort()]['impliedVolatility'].iloc[0])
        except Exception:
            iv_atm = np.nan

        iv_call_up = _nearest_iv(calls, ref_for_atm * 1.05)
        iv_put_dn  = _nearest_iv(puts,  ref_for_atm * 0.95)
        if not np.isnan(iv_call_up) and not np.isnan(iv_put_dn):
            skew_rr5 = (iv_call_up - iv_put_dn) * 100.0
        else:
            skew_rr5 = float('nan')

        em_straddle = float(straddle)
        em_iv = float(ref_for_atm * (iv_atm if not np.isnan(iv_atm) else np.nan) * math.sqrt(T_years)) if (T_years>0 and not np.isnan(iv_atm)) else float('nan')

        em_pct_straddle = em_straddle / ref_for_atm if ref_for_atm>0 else float('nan')
        em_pct_iv = em_iv / ref_for_atm if (ref_for_atm>0 and not np.isnan(em_iv)) else float('nan')

        lower_straddle = ref_for_atm - em_straddle
        upper_straddle = ref_for_atm + em_straddle
        lower_iv = ref_for_atm - em_iv if not np.isnan(em_iv) else float('nan')
        upper_iv = ref_for_atm + em_iv if not np.isnan(em_iv) else float('nan')

        return EMRow(
            symbol=symbol,
            price=float(price),  # unified current price
            price_ref=float(ref_for_atm),
            expiry=expiry,
            dte=T_years*365.0 if not np.isnan(T_years) else float('nan'),
            atm_strike=float(atm_strike),
            call_mid=float(call_mid),
            put_mid=float(put_mid),
            straddle_mid=float(straddle),
            em_straddle=float(em_straddle),
            em_iv=float(em_iv),
            em_pct_straddle=float(em_pct_straddle),
            em_pct_iv=float(em_pct_iv),
            lower_straddle=float(lower_straddle),
            upper_straddle=float(upper_straddle),
            lower_iv=float(lower_iv),
            upper_iv=float(upper_iv),
            iv_atm=float(iv_atm),
            skew_rr5=float(skew_rr5),
            session=session_str
        )
    except Exception:
        return None

# ===== JSON save & (optional) Git push =====

def write_json_for_app(df: pd.DataFrame, path: str) -> None:
    rows = []
    for _, r in df.sort_values(["symbol","expiry"]).iterrows():
        rows.append({
            "symbol": str(r["symbol"]),
            "expiry": str(r["expiry"]),
            "dte": None if pd.isna(r["dte"]) else float(r["dte"]),
            "price": None if pd.isna(r["price"]) else float(r["price"]),  # unified
            "em_dollar": None if pd.isna(r["em_straddle"]) else float(r["em_straddle"]),
            "em_percent": None if pd.isna(r["em_pct_straddle"]) else float(r["em_pct_straddle"]),
            "lower": None if pd.isna(r["lower_straddle"]) else float(r["lower_straddle"]),
            "upper": None if pd.isna(r["upper_straddle"]) else float(r["upper_straddle"]),
            "iv_atm_pct": None if pd.isna(r["iv_atm"]) else round(float(r["iv_atm"])*100, 2),
            "skew_rr5": None if pd.isna(r["skew_rr5"]) else float(r["skew_rr5"]),
            "session": str(r.get("session", "")),
        })
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
        "universe": UNIVERSE,
        "top_mkt_cap": TOP_MKT_CAP,
        "price_basis": PRICE_BASIS,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def git_push_optional(commit_msg: str = "auto update"):
    if os.environ.get("PUSH","0") != "1":
        return
    try:
        st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if st.returncode != 0 or not st.stdout.strip():
            return
        subprocess.check_call(["git", "add", "-A"])
        subprocess.check_call(["git", "commit", "-m", commit_msg])
        subprocess.check_call(["git", "push", "origin", "main"])
    except Exception:
        pass

# ===== Signal generation & printing =====

def _now_strings():
    now_utc = datetime.now(timezone.utc)
    kst = now_utc.astimezone(ZoneInfo("Asia/Seoul"))
    return now_utc.isoformat().replace("+00:00","Z"), kst.strftime("%Y-%m-%d %H:%M:%S KST")

def generate_long_signal(row: pd.Series) -> Optional[dict]:
    price = float(row.get('price', np.nan))          # unified current
    price_ref = float(row.get('price_ref', np.nan))  # EM center line

    lower = float(row['lower_straddle'])
    em = float(row['em_straddle'])

    buf = ENTRY_BAND_BUFFER_PCT
    stp = STOP_BEYOND_EM_PCT
    size = 1.0
    skew = float(row.get('skew_rr5', float('nan')))
    if SKEW_MODE and not np.isnan(skew):
        if skew <= SKEW_T2:
            buf = SKEW_BUF_T2; stp = SKEW_STOP_T2; size = SKEW_SIZE_T2
        elif skew <= SKEW_T1:
            buf = SKEW_BUF_T1; stp = SKEW_STOP_T1; size = SKEW_SIZE_T1

    if price <= lower + buf * em:
        entry = price
        stop = lower - stp * em
        target = float(price_ref)
        return {
            'symbol': row['symbol'],
            'expiry': str(row['expiry']),
            'dte': float(row['dte']),
            'entry': round(entry, 2),
            'stop': round(stop, 2),
            'target': round(target, 2),
            'basis': round(float(price_ref), 2),
            'lower': round(lower, 2),
            'em$': round(em, 2),
            'skew': None if np.isnan(skew) else round(skew, 2),
            'size': round(size, 2),
        }
    return None

def print_long_signals(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo data for signals.")
        return
    sigs = []
    for _, r in df.iterrows():
        s = generate_long_signal(r)
        if s:
            sigs.append(s)
    utc_iso, kst_str = _now_strings()
    if not sigs:
        print(f"\nSearch time: {kst_str} | {utc_iso}")
        print("[Buy signal] None (no stocks near lower band)")
        return
    title_suffix = ", skew applied" if SKEW_MODE else ""
    print(f"\nSearch time: {kst_str} | {utc_iso}")
    print(f"[Buy signal] (Near EM lower band{title_suffix})")
    for s in sigs:
        extra = ""
        if 'skew' in s and s['skew'] is not None:
            extra = f", skew={s['skew']}, buy ratio={int(s['size']*100)}%"
        print(f"{s['symbol']} ({s['expiry']}, DTE={s['dte']}) : Buy {s['entry']} | Stop {s['stop']} | Target {s['target']}  "
              f"(Basis={s['basis']}, Lower={s['lower']}, EM=${s['em$']}{extra})")

def generate_upper_signal(row: pd.Series) -> Optional[dict]:
    price = float(row.get('price', np.nan))
    price_ref = float(row.get('price_ref', np.nan))

    upper = float(row['upper_straddle'])
    em = float(row['em_straddle'])

    if price >= upper - ENTRY_BAND_BUFFER_PCT * em:
        return {
            'symbol': row['symbol'],
            'expiry': str(row['expiry']),
            'dte': float(row['dte']),
            'sell_price': round(price, 2),
            'price': round(price, 2),
            'upper': round(upper, 2),
            'basis': round(float(price_ref), 2),
            'em$': round(em, 2),
        }
    return None

def print_upper_signals(df: pd.DataFrame) -> None:
    if df.empty:
        print("\nNo data for upper-band signals.")
        return
    sigs = []
    for _, r in df.iterrows():
        s = generate_upper_signal(r)
        if s:
            sigs.append(s)
    if not sigs:
        print("\n[Sell signal] None (no stocks near upper band)")
        return
    print("\n[Sell signal] (Near EM upper band)")
    for s in sigs:
        skew_disp = ""
        try:
            skew_val = float(df.loc[df['symbol']==s['symbol'], 'skew_rr5'].iloc[0])
            if not np.isnan(skew_val):
                skew_disp = f", skew={round(skew_val,2)}"
        except Exception:
            pass
        print(f"{s['symbol']} ({s['expiry']}, DTE={s['dte']}) : Sell {s['sell_price']}  "
              f"(Current price={s['price']}, Upper={s['upper']}, Basis={s['basis']}, EM=${s['em$']}{skew_disp})")

# ===== Main pipeline =====

def scan_expected_moves() -> pd.DataFrame:
    symbols = universe_top_by_mktcap(UNIVERSE, TOP_MKT_CAP)
    results: List[EMRow] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = []
        for i, s in enumerate(symbols):
            if REQUEST_DELAY_SEC > 0 and i:
                time.sleep(REQUEST_DELAY_SEC)
            futs.append(ex.submit(compute_expected_move, s, NEAREST_EXPIRY_INDEX))
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r is not None:
                results.append(r)

    if not results:
        print("[DEBUG] No EM rows produced. Possible causes: ticker list empty, price basis None, option chain missing, or STRICT_MID_ONLY filtered all.")
        return pd.DataFrame()

    df = pd.DataFrame([r.__dict__ for r in results])

    # format
    df['dte'] = df['dte'].round(1)
    df['price'] = df['price'].round(4)
    df['price_ref'] = df['price_ref'].round(4)
    df['atm_strike'] = df['atm_strike'].round(2)
    df['call_mid'] = df['call_mid'].round(3)
    df['put_mid'] = df['put_mid'].round(3)
    df['straddle_mid'] = df['straddle_mid'].round(3)
    df['em_straddle'] = df['em_straddle'].round(3)
    df['iv_atm_pct'] = (df['iv_atm'] * 100)
    df['iv_atm_pct'] = df['iv_atm_pct'].where(~df['iv_atm_pct'].isna(), other=np.nan)
    df['iv_atm_pct'] = df['iv_atm_pct'].round(2)
    df['em_pct_straddle'] = (df['em_pct_straddle'] * 100).round(2)
    df['lower_straddle'] = df['lower_straddle'].round(3)
    df['upper_straddle'] = df['upper_straddle'].round(3)
    df['skew_rr5'] = df['skew_rr5'].round(2)

    return df.sort_values('symbol').reset_index(drop=True)

def print_table(df: pd.DataFrame, top_n: Optional[int] = None):
    if df.empty:
        print("⚠️ No rows. Option/price fetch failed or no option chain.")
        return
    base_cols = {
        'symbol': df['symbol'],
        'expiry': df['expiry'],
        'dte': df['dte'].round(1),
        'Price': df['price'].round(2),  # unified price
        'Expected Move ($)': df['em_straddle'].round(2),
        'Expected Move (%)': df['em_pct_straddle'].round(2),
        'Lower Price': df['lower_straddle'].round(2),
        'Upper Price': df['upper_straddle'].round(2),
        'Implied Volatility (%)': df['iv_atm_pct'],
    }
    if SKEW_MODE:
        base_cols['Skew RR5 (volp)'] = df['skew_rr5']
    out = pd.DataFrame(base_cols)
    if top_n:
        out = out.head(top_n)
    out = out.sort_values(['symbol','expiry']).reset_index(drop=True)
    print("\n[Abridged Expected Move Table] (Straddle basis, price_basis = {} )".format(PRICE_BASIS))
    try:
        cur_sess = _us_session_status()
        print(f"(US session: {cur_sess})")
    except Exception:
        pass
    print(out.to_string(index=False, justify='left'))

if __name__ == "__main__":
    try:
        time.sleep(0.2)
        df = scan_expected_moves()
        print_table(df)
        print_long_signals(df)
        print_upper_signals(df)

        # Save app JSON (./output/data.json)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.json")
        write_json_for_app(df, out_path)

        # Optional: push to GitHub when PUSH=1
        git_push_optional(commit_msg=f"auto: EM update @ {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}")
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted (user)")
    except Exception as e:
        print(f"❌ Critical error: {e}")