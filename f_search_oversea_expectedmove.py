from __future__ import annotations
"""
Expected Move 스캐너 (NASDAQ / S&P500)
--------------------------------------
- 전략/매매 로직은 전부 제거했습니다.
- 가까운 만기의 옵션 체인을 이용해 심볼별 Expected Move(EM)를 계산해 표로 출력합니다.
- EM 산식 2가지 모두 제공:
  1) Straddle 방식: ATM Call/Put 미드프라이스 합계(콜+풋) ≒ 시장이 가격에 반영한 단기 변동폭
  2) IV 방식: S * IV_atm * sqrt(T)  (T=년 단위 잔존기간)

사용법 (예):
$ python f_search_oversea_expectedmove.py

기본설정에서:
- UNIVERSE: 'nasdaq' | 'sp500'
- TOP_MKT_CAP: 시가총액 상위 N개를 스캔
- NEAREST_EXPIRY_INDEX: 0(가장 가까운 만기), 1(그 다음), ...
- OVERRIDE_TICKERS: 특정 티커 리스트만 테스트하고 싶을 때 사용

주의:
- yfinance 데이터 품질/지연에 따라 bid/ask가 비어있을 수 있습니다. 이 경우 lastPrice로 대체합니다.
- yahoo_fin이 설치되어 있으면 NASDAQ 전량을 가져오고, 없으면 NASDAQ-100로 폴백합니다.
"""

import concurrent.futures as cf
import math
import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import json
import os
import subprocess

from datetime import datetime, timezone
from math import sqrt

# ===== 기본설정 =====
UNIVERSE: str = "nasdaq"       # 'nasdaq' | 'sp500'
TOP_MKT_CAP: int = 500          # 시총 상위 N개만 스캔
NEAREST_EXPIRY_INDEX: int = 0  # 0=가장 가깝고, 1=그 다음...
MAX_WORKERS: int = 8           # 동시 요청 개수
REQUEST_DELAY_SEC: float = 0.03 # 티커별 요청 사이 텀(서버 부하 방지)


# 기준가 설정: 'last' = 실시간/최근가, 'prev_close' = 전일 종가 기준 (많은 사이트가 이 기준으로 EM 밴드 산출)
PRICE_BASIS: str = "prev_close"  # "last" | "prev_close"
# Straddle 값을 ATM 근처 두 행사가 사이에서 선형 보간하여 좀 더 정확하게 추정할지 여부
STRADDLE_INTERP: bool = True

# EM 정밀도 옵션
STRICT_MID_ONLY: bool = False  # True: mid만 사용(권장 정확도), False: mid 없으면 last/bid/ask 폴백 허용
USE_COMMON_ATM_STRIKE: bool = True  # True: 콜/풋 동일 행사가(K*)에서만 스트래들 합산

# ===== Skew 모드 설정 =====
SKEW_MODE: bool = False   # True=스큐 반영, False=현재 로직 유지
SKEW_T1: float = -6.0    # 중간 경고 (volp)
SKEW_T2: float = -10.0   # 강한 경고 (volp)
SKEW_BUF_T1: float = 0.03   # 진입 버퍼 조정(기본 0.05 → 0.03)
SKEW_BUF_T2: float = 0.02   # 진입 버퍼(강한 음수)
SKEW_STOP_T1: float = 0.12  # 손절 조정(기본 0.10 → 0.12)
SKEW_STOP_T2: float = 0.15  # 손절(강한 음수)
SKEW_SIZE_T1: float = 0.7   # 포지션 크기 배율(=매수 비율 추천)
SKEW_SIZE_T2: float = 0.5   # 포지션 크기 배율(=매수 비율 추천)

# ===== Long-only 전략 파라미터 =====
ENTRY_BAND_BUFFER_PCT: float = 0.05  # 하단 밴드에서 EM의 5% 위까지 진입 허용
STOP_BEYOND_EM_PCT: float = 0.10     # 하단 밴드 바깥으로 EM의 10% 추가 이탈 시 손절

# 테스트용으로 특정 심볼만 보려면 설정 (예: ["AAPL","MSFT","NVDA"]) 
OVERRIDE_TICKERS: Optional[List[str]] = None

# ----- 외부 모듈 폴백 처리 -----
try:
    from yahoo_fin import stock_info as si
    HAVE_YFIN = True
except Exception:
    HAVE_YFIN = False

# ===== 유틸 =====

def _fetch_html_with_ua(url: str) -> str:
    """403 회피용: User-Agent를 지정해 HTML을 문자열로 가져옴."""
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
    """pandas.read_html에 직접 HTML 문자열을 넣어서 403 문제 완화."""
    html = _fetch_html_with_ua(url)
    return pd.read_html(StringIO(html))

def _clean_symbol(sym: str) -> str:
    """Wikipedia 표기를 yfinance 호환(BRK.B → BRK-B)으로 정리."""
    return sym.replace(".", "-").strip()

def get_sp500_tickers() -> List[str]:
    # 0) yfinance 내장 리스트 우선 시도 (빠르고 403 무관)
    try:
        if hasattr(yf, "tickers_sp500"):
            lst = yf.tickers_sp500()
            if lst:
                return [_clean_symbol(t) for t in lst]
    except Exception:
        pass
    # 1) 위키를 User-Agent로 파싱
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = _read_html_tables(url)
    if not tables:
        raise RuntimeError("S&P500 테이블을 찾지 못했습니다.")
    df = tables[0]
    col = 'Symbol' if 'Symbol' in df.columns else df.columns[0]
    tickers = [_clean_symbol(str(s)) for s in df[col].astype(str).tolist()]
    tickers = [t for t in tickers if t and t.lower() != 'nan']
    # 2) 폴백: 축약 리스트 (상위 대표 30개 정도)
    if not tickers:
        tickers = [
            'AAPL','MSFT','NVDA','AMZN','GOOGL','META','BRK-B','LLY','AVGO','JPM',
            'TSLA','XOM','V','WMT','UNH','MA','JNJ','PG','HD','COST',
            'ABBV','BAC','CVX','MRK','PEP','KO','ADBE','NFLX','CRM','CSCO'
        ]
    return tickers

def get_nasdaq_tickers() -> List[str]:
    # 0) yfinance 내장 리스트 우선
    try:
        if hasattr(yf, "tickers_nasdaq"):
            lst = yf.tickers_nasdaq()
            if lst:
                return [_clean_symbol(t) for t in lst]
    except Exception:
        pass
    # 1) yahoo_fin 사용 (있을 때)
    if HAVE_YFIN:
        try:
            tickers = si.tickers_nasdaq()
            return [t.replace('.', '-').strip() for t in tickers if isinstance(t, str) and t.strip()]
        except Exception:
            pass
    # 2) 위키 NASDAQ-100을 UA로 파싱
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    tables = _read_html_tables(url)
    for tb in tables:
        cols = [str(c) for c in tb.columns]
        if 'Ticker' in cols or 'Symbol' in cols:
            col = 'Ticker' if 'Ticker' in cols else 'Symbol'
            tickers = [str(s) for s in tb[col].tolist()]
            return [s.replace('.', '-').strip() for s in tickers if s and s.lower() != 'nan']
    # 3) 폴백: 대표주 30개 내외
    return [
        'AAPL','MSFT','NVDA','AMZN','META','GOOGL','AVGO','TSLA','PEP','COST',
        'ADBE','CSCO','NFLX','AMD','INTC','PYPL','QCOM','TXN','AMAT','INTU',
        'SBUX','CHTR','MU','BKNG','LRCX','HON','MDLZ','PDD','ABNB','REGN'
    ]

def safe_current_price(tk: yf.Ticker) -> Optional[float]:
    """현재가 조회(우선순위: fast_info → info → 최근 종가)."""
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
            # 마지막 행이 오늘이면 그 전날 종가, 아니면 마지막 종가
            if len(hist) >= 2 and hist.index[-1].date() == datetime.utcnow().date():
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
    # default: last
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
    # 시총 수집 후 상위 N개 선별
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

# ===== Expected Move 계산 =====

@dataclass
class EMRow:
    symbol: str
    price: float
    price_ref: float
    price_live: float
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
        return float('nan')  # mid가 없으면 사용하지 않음
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
    raise last_err if last_err else RuntimeError("option_chain 실패")

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
        price_live = safe_current_price(tk)
        price_ref = price_by_basis(tk)
        price = price_ref  # 아래 계산은 기준가로 진행
        if price is None or price <= 0:
            return None

        # ATM에 가장 가까운 공통 행사가 선택
        def _nearest_common_strike(calls_df: pd.DataFrame, puts_df: pd.DataFrame, ref_price: float) -> Optional[float]:
            if calls_df is None or calls_df.empty or puts_df is None or puts_df.empty:
                return None
            # 공통 스트라이크 교집합에서 ref_price에 가장 가까운 행사가 선택
            c_strikes = pd.Series(calls_df['strike']).astype(float)
            p_strikes = pd.Series(puts_df['strike']).astype(float)
            common = np.intersect1d(c_strikes.values, p_strikes.values)
            if common.size == 0:
                return None
            idx = np.argmin(np.abs(common - ref_price))
            return float(common[idx])

        K = _nearest_common_strike(calls, puts, price)
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

        # mid 유효성 검사: 둘 중 하나라도 NaN이면 (STRICT_MID_ONLY가 True인 경우) 스킵
        if np.isnan(call_mid) or np.isnan(put_mid):
            if STRICT_MID_ONLY:
                return None

        straddle = float(call_mid) + float(put_mid)

        # --- (선택) ATM 선형 보간 ---
        if USE_COMMON_ATM_STRIKE:
            STRADDLE_INTERP_LOCAL = False
        else:
            STRADDLE_INTERP_LOCAL = STRADDLE_INTERP
        if STRADDLE_INTERP_LOCAL and 'strike' in calls.columns and len(calls) > 1 and 'strike' in puts.columns and len(puts) > 1:
            try:
                c_sorted = calls.sort_values('strike').reset_index(drop=True)
                p_sorted = puts.sort_values('strike').reset_index(drop=True)
                ci_low = max(0, int((c_sorted['strike'] <= price).sum() - 1))
                ci_high = min(len(c_sorted)-1, ci_low + 1)
                pi_low = max(0, int((p_sorted['strike'] <= price).sum() - 1))
                pi_high = min(len(p_sorted)-1, pi_low + 1)
                k1 = float(c_sorted.loc[ci_low, 'strike'])
                k2 = float(c_sorted.loc[ci_high, 'strike'])
                if k2 > k1 and (k1 <= price <= k2):
                    def mid_at(df_, idx_): return _mid_or_last(df_.loc[idx_])
                    s1 = mid_at(c_sorted, ci_low) + mid_at(p_sorted, pi_low)
                    s2 = mid_at(c_sorted, ci_high) + mid_at(p_sorted, pi_high)
                    w = (price - k1) / (k2 - k1)
                    straddle = (1-w) * s1 + w * s2
                    call_mid = float('nan')
                    put_mid  = float('nan')
            except Exception:
                pass

        T = _days_to_expiry_yr(expiry)
        iv_atm = np.nan
        try:
            # call 우선, 없으면 put
            if 'impliedVolatility' in calls.columns:
                iv_atm = float(calls.iloc[(calls['strike'] - price).abs().argsort()]['impliedVolatility'].iloc[0])
            if (np.isnan(iv_atm)) and ('impliedVolatility' in puts.columns):
                iv_atm = float(puts.iloc[(puts['strike'] - price).abs().argsort()]['impliedVolatility'].iloc[0])
        except Exception:
            iv_atm = np.nan

        # ----- 스큐(±5% 리스크리버설 근사) 계산 -----
        iv_call_up = _nearest_iv(calls, price * 1.05)
        iv_put_dn  = _nearest_iv(puts,  price * 0.95)
        if not np.isnan(iv_call_up) and not np.isnan(iv_put_dn):
            skew_rr5 = (iv_call_up - iv_put_dn) * 100.0  # vol points(%p)
        else:
            skew_rr5 = float('nan')

        em_straddle = float(straddle)
        em_iv = float(price * (iv_atm if not np.isnan(iv_atm) else np.nan) * math.sqrt(T)) if (T>0 and not np.isnan(iv_atm)) else float('nan')

        em_pct_straddle = em_straddle / price if price>0 else float('nan')
        em_pct_iv = em_iv / price if (price>0 and not np.isnan(em_iv)) else float('nan')

        lower_straddle = price - em_straddle
        upper_straddle = price + em_straddle
        lower_iv = price - em_iv if not np.isnan(em_iv) else float('nan')
        upper_iv = price + em_iv if not np.isnan(em_iv) else float('nan')

        return EMRow(
            symbol=symbol,
            price=float(price),
            price_ref=float(price),
            price_live=float(price_live) if price_live else float('nan'),
            expiry=expiry,
            dte=T*365.0 if not np.isnan(T) else float('nan'),
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
        )
    except Exception:
        return None



# ===== JSON 저장 & (옵션) Git 푸시 헬퍼 =====

def write_json_for_app(df: pd.DataFrame, path: str) -> None:
    """앱에서 쉽게 읽을 수 있는 간단 JSON으로 저장."""
    rows = []
    for _, r in df.sort_values(["symbol","expiry"]).iterrows():
        rows.append({
            "symbol": str(r["symbol"]),
            "expiry": str(r["expiry"]),
            "dte": None if pd.isna(r["dte"]) else float(r["dte"]),
            "price": None if pd.isna(r["price_ref"]) else float(r["price_ref"]),
            "price_live": None if pd.isna(r["price_live"]) else float(r["price_live"]),
            "em_dollar": None if pd.isna(r["em_straddle"]) else float(r["em_straddle"]),
            "em_percent": None if pd.isna(r["em_pct_straddle"]) else float(r["em_pct_straddle"]),
            "lower": None if pd.isna(r["lower_straddle"]) else float(r["lower_straddle"]),
            "upper": None if pd.isna(r["upper_straddle"]) else float(r["upper_straddle"]),
            "iv_atm_pct": None if pd.isna(r["iv_atm"]) else round(float(r["iv_atm"])*100, 2),
            "skew_rr5": None if pd.isna(r["skew_rr5"]) else float(r["skew_rr5"]),
        })
    payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
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
    """
    환경변수 PUSH=1 일 때만 git add/commit/push 수행.
    (앱 새로고침 시 수동 실행 용도이므로 기본은 파일만 저장)
    """
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
        # 푸시 실패해도 전체 실행은 성공으로 간주
        pass

# ===== 간단 Long 시그널 생성 & 출력 =====

def generate_long_signal(row: pd.Series) -> Optional[dict]:
    """EM 하단 밴드 근처에서 Long 진입 시그널 생성 (숏 없음)."""
    # 실시간 가격이 있으면 우선 사용, 없으면 기준가 사용
    price_live = row.get('price_live', np.nan)
    price_ref = row.get('price', np.nan)
    price = float(price_live) if not pd.isna(price_live) else float(price_ref)

    lower = float(row['lower_straddle'])
    em = float(row['em_straddle'])

    # --- 스큐 모드에 따른 동적 조정 및 매수 비율 추천 ---
    buf = ENTRY_BAND_BUFFER_PCT
    stp = STOP_BEYOND_EM_PCT
    size = 1.0
    skew = float(row.get('skew_rr5', float('nan')))
    if SKEW_MODE and not np.isnan(skew):
        if skew <= SKEW_T2:
            buf = SKEW_BUF_T2; stp = SKEW_STOP_T2; size = SKEW_SIZE_T2
        elif skew <= SKEW_T1:
            buf = SKEW_BUF_T1; stp = SKEW_STOP_T1; size = SKEW_SIZE_T1

    # 진입 조건: 가격이 하단 밴드 + 버퍼(EM * buf) 이하
    if price <= lower + buf * em:
        entry = price
        stop = lower - stp * em
        target = float(price_ref)  # 중앙선(기준가)로 간단 설정
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
            'size': round(size, 2),  # 매수 비율 추천
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
    if not sigs:
        print("\n[매수 시점] 없음 (하단 밴드 근처 종목 없음)")
        return
    title_suffix = ", 스큐 적용" if SKEW_MODE else ""
    print(f"\n[매수 시점] (EM 하단 밴드 근처{title_suffix})")
    for s in sigs:
        extra = ""
        if 'skew' in s and s['skew'] is not None:
            extra = f", skew={s['skew']}, 매수비율={int(s['size']*100)}%"
        print(f"{s['symbol']} ({s['expiry']}, DTE={s['dte']}) : 매수 {s['entry']} | 손절 {s['stop']} | 목표 {s['target']}  "
              f"(기준가={s['basis']}, 하단={s['lower']}, EM=${s['em$']}{extra})")

# ===== 상단 밴드 근처 (청산/매도 후보) 시그널 =====

def generate_upper_signal(row: pd.Series) -> Optional[dict]:
    """EM 상단 밴드 근처에서 청산(매도) 후보 시그널 생성."""
    price_live = row.get('price_live', np.nan)
    price_ref = row.get('price', np.nan)
    price = float(price_live) if not pd.isna(price_live) else float(price_ref)

    upper = float(row['upper_straddle'])
    em = float(row['em_straddle'])

    # 조건: 현재가가 상단 밴드 - 버퍼(EM * ENTRY_BAND_BUFFER_PCT) 이상
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
        print("\n[매도 시점] 없음 (상단 밴드 근처 종목 없음)")
        return
    print("\n[매도 시점] (EM 상단 밴드 근처)")
    for s in sigs:
        skew_disp = ""
        try:
            skew_val = float(df.loc[df['symbol']==s['symbol'], 'skew_rr5'].iloc[0])
            if not np.isnan(skew_val):
                skew_disp = f", skew={round(skew_val,2)}"
        except Exception:
            pass
        print(f"{s['symbol']} ({s['expiry']}, DTE={s['dte']}) : 매도 {s['sell_price']}  "
              f"(현재가={s['price']}, 상단={s['upper']}, 기준가={s['basis']}, EM=${s['em$']}{skew_disp})")


# ===== 메인 파이프라인 =====

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

    # 보기 좋은 포맷팅용 컬럼 추가
    df['dte'] = df['dte'].round(1)
    df['price'] = df['price'].round(4)
    df['price_ref'] = df['price_ref'].round(4)
    df['price_live'] = df['price_live'].round(4)
    df['atm_strike'] = df['atm_strike'].round(2)
    df['call_mid'] = df['call_mid'].round(3)
    df['put_mid'] = df['put_mid'].round(3)
    df['straddle_mid'] = df['straddle_mid'].round(3)
    df['em_straddle'] = df['em_straddle'].round(3)
    df['iv_atm_pct'] = (df['iv_atm'] * 100)
    df['iv_atm_pct'] = df['iv_atm_pct'].where(~df['iv_atm_pct'].isna(), other=np.nan)
    df['iv_atm_pct'] = df['iv_atm_pct'].round(2)
    df['em_pct_straddle'] = (df['em_pct_straddle'] * 100).round(2)
    # Removed rounding for em_iv, em_pct_iv, lower_iv, upper_iv as requested
    df['lower_straddle'] = df['lower_straddle'].round(3)
    df['upper_straddle'] = df['upper_straddle'].round(3)
    df['skew_rr5'] = df['skew_rr5'].round(2)

    # 거래대금 순 대신 시총 상위로 뽑았으므로 알파벳순 정렬
    return df.sort_values('symbol').reset_index(drop=True)

def print_table(df: pd.DataFrame, top_n: Optional[int] = None):
    if df.empty:
        print("⚠️ 결과가 비었습니다. 옵션/시세 조회 실패 또는 만기체인 부재 가능.")
        return
    base_cols = {
        'symbol': df['symbol'],
        'expiry': df['expiry'],
        'dte': df['dte'].round(1),
        'price': df['price_ref'].round(2),
        'price_live': df['price_live'].round(2),
        'Expected Move ($)': df['em_straddle'].round(2),
        'Expected Move (%)': df['em_pct_straddle'].round(2),
        'Lower Price': df['lower_straddle'].round(2),
        'Upper Price': df['upper_straddle'].round(2),
        'Implied Volatility (%)': df['iv_atm_pct'],
    }
    if SKEW_MODE:
        base_cols['Skew RR5 (volp)'] = df['skew_rr5']
    out = pd.DataFrame(base_cols)
    # 선택적으로 상위 N개만 보여주기
    if top_n:
        out = out.head(top_n)
    # 정렬은 심볼 오름차순 유지
    out = out.sort_values(['symbol','expiry']).reset_index(drop=True)
    print("\n[Abridged Expected Move Table] (Straddle 기준, price = {} )".format(PRICE_BASIS))
    print(out.to_string(index=False, justify='left'))

if __name__ == "__main__":
    try:
        time.sleep(0.2)
        df = scan_expected_moves()
        print_table(df)
        print_long_signals(df)
        print_upper_signals(df)

        # 앱용 JSON 저장 (현재 파일 기준 상위 폴더에 output/data.json 생성)
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "data.json")
        write_json_for_app(df, out_path)

        # (선택) 환경변수 PUSH=1 이면 GitHub로 자동 커밋/푸시
        git_push_optional(commit_msg=f"auto: EM update @ {datetime.utcnow().isoformat()}Z")

    except KeyboardInterrupt:
        print("\n⏹️ 중단됨 (사용자)")
    except Exception as e:
        print(f"❌ 치명적 오류: {e}")
        