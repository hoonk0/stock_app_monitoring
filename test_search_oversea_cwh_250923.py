#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cup & Handle Multi-Symbol Scanner (with Yahoo Finance)
======================================================
이 스크립트는 Yahoo Finance 데이터를 이용해 다수의 종목(주식/코인)을 대상으로
컵앤핸들 패턴을 자동으로 탐지합니다.

주요 기능:
- 유니버스 선택: S&P500, NASDAQ100, Crypto, 또는 직접 지정
- 상위 N개 종목만 스캔 (거래대금/시가총액 기준 정렬)
- 봉 간격 선택: 일봉(1d) / 주봉(1wk) / 시간봉(1h)
- 컵앤핸들 패턴 검출 (형성 중 or 돌파 완료)
- 옵션으로 형성 중만 표시하거나, 돌파+형성 중 모두 표시 가능
- CSV로 결과 저장 가능

사용 예시:
  $ python test_search_oversea_cwh_250923.py --universe crypto --topn 30 --timeframe 1h --years 1
  $ python test_search_oversea_cwh_250923.py --symbols AAPL,MSFT,NVDA --include-forming
  $ python test_search_oversea_cwh_250923.py --forming-only --export signals.csv
  $ python test_search_oversea_cwh_250923.py --universe sp500 --topn 50 --rank-by dollarvol --timeframe 1wk
  $ python test_search_oversea_cwh_250923.py --universe crypto --topn 30 --timeframe 1h --years 1

기본값(Params.default()):
- Universe: crypto
- Top N: 30
- Timeframe: 1d
- 모드: 아래 default() 내부의 `mode` 값을 한 줄로 선택
    - "breakout_only"         : 돌파만
    - "breakout_plus_forming" : 돌파 + 형성 중
    - "forming_only"          : 형성 중만
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True

import traceback

# === Suppress FutureWarning for float(single-row Series) ===
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Calling float on a single element Series is deprecated")

try:
    import yfinance as yf
except Exception as e:
    print("yfinance 가 필요합니다: pip install yfinance", file=sys.stderr)
    raise

# =====================
# 파라미터 세트 (기본값)
# =====================
@dataclass
class Params:
    years: int = 2
    ph_lookback: int = 3
    pl_lookback: int = 3
    min_cup_bars: int = 30
    max_cup_bars: int = 250
    rim_tolerance_pct: float = 0.08
    min_depth_pct: float = 0.12
    max_depth_pct: float = 0.55
    percent_max_breaks: float = 0.015

    min_handle_bars: int = 3
    max_handle_bars: int = 30
    handle_min_ratio: float = 0.33
    handle_max_ratio: float = 0.50

    breakout_buffer_pct: float = 0.001
    use_volume_filter: bool = False
    volume_boost: float = 1.2
    trigger: str = "close"

    # === Default scan universe / ranking / timeframe ===
    universe: str = "nasdaq"      # 기본값: crypto sp500 nasdaq
    topn: int = 2000                # 기본값: 상위 30개만
    rank_by: str = "dollarvol"
    interval: str = "1d리스ㅡ텅ㄹ"

    # === 패턴 모드 ===
    include_forming: bool = False   # 돌파 전(형성 중)도 포함할지
    forming_only: bool = False      # 형성 중만 볼지
    forming_max_age_bars: int = 60  # 우측림 이후 최근성(최대 봉 수)

    export_path: Optional[str] = None
    display_split_status: bool = False  # 터미널 표 분할 표시 여부

    # === 컵림 우선/허용치 옵션 ===
    prefer_earliest_right_rim: bool = True  # 최근보다 바닥 직후의 '가장 이른' 우측 컵림을 우선할지
    rim_epsilon_up: float = 0.02             # 우측 컵림이 좌측 컵림보다 약간 높은 경우 허용치(+2%)

    # === 근접 breakout 필터 ===
    near_breakout_pct: Optional[float] = None  # buy_point 대비 현재가가 이 값(%) 이내로 근접한 '형성중'만 필터

    @classmethod
    def default(cls) -> "Params":
        """한 곳에서 기본값 관리 (모드 선택 포함).
        mode 선택:
          - "breakout_only"         : 돌파만
          - "breakout_plus_forming" : 돌파 + 형성중
          - "forming_only"          : 형성중만
        """
        base = cls()
        # === 기본 모드 선택 (여기 한 줄만 바꾸면 전체 기본 동작 변경) ===
        mode = "forming_only"   # <-- "breakout_only" / "breakout_plus_forming" / "forming_only"
        if mode == "breakout_only":
            base.include_forming = False
            base.forming_only = False
        elif mode == "breakout_plus_forming":
            base.include_forming = True
            base.forming_only = False
        elif mode == "forming_only":
            base.include_forming = False
            base.forming_only = True
        # 컵림 우선/허용치 기본값
        base.prefer_earliest_right_rim = False
        base.rim_epsilon_up = 0.02
        # === 기본 근접 브레이크아웃 필터 ===
        # 아래 한 줄로 기본값을 쉽게 바꿀 수 있습니다.
        # - None : 필터 끔 (기본)
        # - 예) 1.0 : '형성중' 중에서 buy_point 대비 1% 이내로 근접한 종목만
        base.near_breakout_pct = None
        # === 기본 출력 형식 ===
        # True 로 두면 터미널에 '형성중' / '돌파완료' 표를 따로 보여줍니다.
        base.display_split_status = False
        return base


# =====================
# 유틸 함수
# =====================

def _yf_period_for(interval: str, years: int) -> str:
    """yfinance period 문자열 계산. 1h는 API 제한으로 최대 730d 근처만 허용."""
    if interval == "1h":
        days = min(int(years * 365), 730)
        days = max(days, 60)  # 최소 60일 보장
        return f"{days}d"
    elif interval == "1wk":
        return f"{years}y"
    else:  # 1d
        return f"{years}y"


def download_ohlcv(ticker: str, years: int, interval: str = "1d") -> pd.DataFrame:
    """Yahoo Finance에서 OHLCV 로드 및 표준화.
    컬럼: [open, high, low, close, volume] (소문자)
    interval: 1d | 1wk | 1h
    """
    period = _yf_period_for(interval, years)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance 기본 컬럼 표준화
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    keep = ["open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            raise ValueError(f"{ticker}: OHLCV 컬럼 누락: {c}")
    df = df[keep].copy().dropna()
    return df

# =====================
# 유니버스 & 랭킹 유틸
# =====================

def _default_crypto_watchlist() -> List[str]:
    # yfinance 심볼 표기. 대표 코인 위주(원하면 symbols-file로 대체 가능)
    return [
        "BTC-USD","ETH-USD","USDT-USD","BNB-USD","SOL-USD","XRP-USD","USDC-USD","DOGE-USD",
        "ADA-USD","TRX-USD","TON-USD","AVAX-USD","SHIB-USD","DOT-USD","WBTC-USD","LINK-USD",
        "BCH-USD","NEAR-USD","UNI-USD","LTC-USD","APT-USD","ATOM-USD","XMR-USD","ETC-USD",
        "XLM-USD","ICP-USD","FIL-USD","IMX-USD","OKB-USD","HBAR-USD"
    ]


def _try_read_sp500() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df[df.columns[0]].astype(str).str.replace("\n", "").str.strip().tolist()
        return syms
    except Exception:
        # 최소 대형주 백업 리스트
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","BRK-B","LLY","JPM","UNH","XOM",
            "V","TSLA","PG","JNJ","MA","HD","COST","MRK"
        ]


def _try_read_nasdaq100() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        # 구성 종목 표 추출(첫 테이블에 없는 경우도 있어 약간 관대하게 합침)
        syms = []
        for t in tables:
            cols = [c for c in t.columns if str(c).lower().startswith("ticker") or str(c).lower().startswith("symbol")]
            if cols:
                syms += t[cols[0]].astype(str).str.replace("\n", "").str.strip().tolist()
        syms = [s for s in syms if s and s != "nan"]
        return list(dict.fromkeys(syms))
    except Exception:
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","PEP","COST","ADBE",
            "NFLX","AMD","INTC","CSCO","QCOM","HON","AMAT","CMCSA"
        ]


def build_universe(universe: str) -> List[str]:
    u = universe.lower()
    if u == "sp500":
        return _try_read_sp500()
    if u == "nasdaq100":
        return _try_read_nasdaq100()
    if u == "crypto":
        return _default_crypto_watchlist()
    return []  # custom


def rank_topn(symbols: List[str], topn: int, rank_by: str, years: int, interval: str) -> List[str]:
    if topn is None or topn <= 0 or topn >= len(symbols):
        return symbols

    rank_by = (rank_by or "dollarvol").lower()

    scores: Dict[str, float] = {}
    # 최근 60~90개 바 기준으로 거래대금 평균 계산(빠르고 범용)
    for s in symbols:
        try:
            df = download_ohlcv(s, years=max(1, min(years, 2)), interval=interval)
            if df.empty:
                continue
            tail = df.tail(90)
            if tail.empty:
                continue
            if rank_by == "marketcap":
                # yfinance fast_info의 market_cap 시도, 실패시 dollarvol로 대체
                try:
                    info = yf.Ticker(s).fast_info
                    mc = float(getattr(info, "market_cap", None) or info.get("market_cap", None) or 0)
                    if mc and mc > 0:
                        scores[s] = mc
                        continue
                except Exception:
                    pass
            # fallback: 평균 거래대금(= close*volume)
            dv = float((tail["close"] * tail["volume"]).mean())
            scores[s] = dv
        except Exception:
            continue

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in ranked[:topn]] if ranked else symbols[:topn]


def pivot_points(series: pd.Series, lookback: int, mode: str) -> pd.Series:
    """간단한 피벗 검출 (국소 고점/저점)."""
    arr = series.values
    n = len(arr)
    is_pivot = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        window = arr[i - lookback : i + lookback + 1]
        center = arr[i]
        if mode == "H":
            if center == window.max() and (window == center).sum() == 1:
                is_pivot[i] = True
        else:  # mode == "L"
            if center == window.min() and (window == center).sum() == 1:
                is_pivot[i] = True
    return pd.Series(is_pivot, index=series.index)


def line_y(x0, y0, x1, y1, x):
    """x0~x1 직선상 x 에 대응하는 y (선형보간). 인덱스는 정수형 바 인덱스 사용."""
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class CupHandlePattern:
    left_rim_date: pd.Timestamp
    bottom_date: pd.Timestamp
    right_rim_date: pd.Timestamp
    handle_low_date: pd.Timestamp
    breakout_date: Optional[pd.Timestamp]  # 형성 중이면 None

    left_rim_price: float
    bottom_price: float
    right_rim_price: float
    handle_low_price: float
    breakout_close: Optional[float]        # 형성 중이면 None

    depth_pct: float
    handle_drop_pct: float
    buy_point: float
    stop_loss: float

    status: str                             # "breakout" | "forming"


# =====================
# 핵심 로직: 컵앤핸들 탐지
# =====================

def detect_cup_handle(df: pd.DataFrame, p: Params) -> Optional[CupHandlePattern]:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # Use NumPy arrays to ensure scalar indexing always returns Python floats
    closev = close.to_numpy()
    highv  = high.to_numpy()
    lowv   = low.to_numpy()
    volv   = vol.to_numpy()

    # --- helpers to avoid NumPy deprecation on array-to-scalar conversion ---
    def _scalar(x) -> float:
        """Return a Python float from any NumPy/Pandas scalar/0-d/1-d element."""
        return float(np.asarray(x).reshape(-1)[0])

    def _at(arr, idx) -> float:
        """Safe float(arr[int(idx)]) with deprecation-proof handling."""
        return _scalar(arr[int(idx)])

    # 트리거 기준 시리즈 선택
    trigger_series = closev if (p.trigger == "close") else highv

    ph = pivot_points(high, p.ph_lookback, "H")  # Pivot High
    pl = pivot_points(low, p.pl_lookback, "L")   # Pivot Low

    idxs = np.arange(len(df))

    # 모든 좌림/우림 조합 탐색 (최근 패턴 우선)
    ph_idx = np.where(ph.values)[0]
    pl_idx = np.where(pl.values)[0]

    r2_iter = ph_idx if getattr(p, "prefer_earliest_right_rim", False) else reversed(ph_idx)
    for r2_raw in r2_iter:  # 우측 컵림 후보 순회(초기/최신 우선 선택 가능)
        r2 = int(r2_raw)
        # 좌측림 후보는 min_cup_bars 이전까지
        r1_candidates = ph_idx[ph_idx < r2 - p.min_cup_bars]
        if len(r1_candidates) == 0:
            continue
        # 컵 길이 제한
        r1_candidates = r1_candidates[r1_candidates >= r2 - p.max_cup_bars]
        if len(r1_candidates) == 0:
            continue
        for r1_raw in reversed(r1_candidates):
            r1 = int(r1_raw)
            # --- Left Rim re-anchoring ---
            # 좌/우 림(r1~r2) 사이에 r1보다 더 높은 Pivot High가 있으면, 그 피벗을 새로운 좌측림으로 재정의
            mid_ph = ph_idx[(ph_idx > r1) & (ph_idx < r2)]
            if len(mid_ph) > 0:
                best_mid_idx = int(mid_ph[np.argmax(highv[mid_ph])])
                if _at(highv, best_mid_idx) > _at(highv, r1):
                    r1 = best_mid_idx
                    # 재정의 후 컵 길이 제약 재검사
                    if (r2 - r1) < p.min_cup_bars or (r2 - r1) > p.max_cup_bars:
                        continue

            Lp = _at(highv, r1)
            Rp = _at(highv, r2)
            # 우측 컵림이 좌측 컵림보다 약간 높은 경우도 허용 (예: +2% 이내)
            if Rp > Lp * (1.0 + float(getattr(p, "rim_epsilon_up", 0.0))):
                continue
            # 림 가격 유사성 검사
            if abs(Rp - Lp) / Lp > p.rim_tolerance_pct:
                continue

            # 바닥(최저 pivot low)은 r1~r2 사이
            mids = pl_idx[(pl_idx > r1) & (pl_idx < r2)]
            if len(mids) == 0:
                continue
            b = mids[np.argmin(lowv[mids])]
            b = int(b)
            Bp = _at(lowv, b)

            cup_height = Lp - Bp
            if cup_height <= 0:
                continue
            depth_pct = cup_height / Lp
            if not (p.min_depth_pct <= depth_pct <= p.max_depth_pct):
                continue

            # 테스트라인(좌림~우림): 중간 구간에서 상방 과도 돌파 여부 체크
            violated = False
            for x in range(r1 + 1, r2):
                y_hat = line_y(r1, Lp, r2, Rp, x)
                if _scalar(closev[x]) > _scalar(y_hat * (1 + p.percent_max_breaks)):
                    violated = True
                    break
            if violated:
                continue

            # 핸들 구간: 우측림 이후 min~max bar 사이에서 최저점 탐색
            h_start = r2 + 1
            h_end = min(r2 + p.max_handle_bars, len(df) - 1)
            if h_end - h_start + 1 < p.min_handle_bars:
                continue
            # use NumPy for robust scalar extraction
            local_argmin = int(np.argmin(lowv[h_start:h_end + 1]))
            h_i = int(h_start + local_argmin)
            handle_low = _at(lowv, h_i)

            handle_bars = h_i - r2
            if handle_bars < p.min_handle_bars:
                continue

            # 핸들 되돌림 비율
            handle_ratio = (Rp - handle_low) / cup_height
            if not (p.handle_min_ratio <= handle_ratio <= p.handle_max_ratio):
                continue
            # 핸들 저점은 컵 상단 절반 이상 권장
            if _scalar(handle_low) < _scalar(Bp + 0.5 * cup_height):
                continue

            # 핸들 상단(돌파 레벨): 핸들 시작~핸들 저점 이후의 최고 종가
            # 보수적으로: 우측림 이후 ~ 핸들 저점 사이의 최고 종가를 상단으로 사용
            # 핸들 상단은 일반적으로 고가 기준이 더 보수적인 저항선이므로 high 기반으로 계산
            handle_high = _scalar(np.max(highv[r2:h_i + 1]))
            buy_point = _scalar(handle_high * (1 + float(p.breakout_buffer_pct)))

            # 오늘(마지막 봉) 돌파 여부 체크
            last_i = len(df) - 1

            # breakout search from handle_low_date+1 to last_i for earliest breakout day
            search_start = h_i + 1
            if search_start > last_i:
                continue

            hits_rel = np.where(trigger_series[search_start:] >= buy_point)[0]
            if len(hits_rel) == 0:
                # 돌파 전(형성 중) 후보 반환 (옵션)
                if getattr(p, "include_forming", False) or getattr(p, "forming_only", False):
                    # 우측림 이후 최근성 제한
                    last_i = len(df) - 1
                    if (last_i - r2) <= int(getattr(p, "forming_max_age_bars", 60)):
                        stop_loss = _scalar(handle_low * 0.995)
                        return CupHandlePattern(
                            left_rim_date=df.index[r1],
                            bottom_date=df.index[b],
                            right_rim_date=df.index[r2],
                            handle_low_date=df.index[h_i],
                            breakout_date=None,
                            left_rim_price=_scalar(Lp),
                            bottom_price=_scalar(Bp),
                            right_rim_price=_scalar(Rp),
                            handle_low_price=_scalar(handle_low),
                            breakout_close=None,
                            depth_pct=_scalar(depth_pct * 100),
                            handle_drop_pct=_scalar((Rp - handle_low) / Rp * 100),
                            buy_point=_scalar(buy_point),
                            stop_loss=_scalar(stop_loss),
                            status="forming",
                        )
                continue
            breakout_rel_i = hits_rel[0]
            breakout_i = search_start + breakout_rel_i

            # volume filter 적용 시
            if p.use_volume_filter:
                vol20 = float(vol.rolling(20).mean().iloc[breakout_i])
                if not np.isnan(vol20) and vol20 > 0:
                    vol_ok = bool(float(volv[breakout_i]) >= vol20 * float(p.volume_boost))
                    if not vol_ok:
                        continue
            stop_loss = _scalar(handle_low * 0.995)  # 핸들 저점 살짝 하회 (0.5% 버퍼)

            return CupHandlePattern(
                left_rim_date=df.index[r1],
                bottom_date=df.index[b],
                right_rim_date=df.index[r2],
                handle_low_date=df.index[h_i],
                breakout_date=df.index[breakout_i],
                left_rim_price=_scalar(Lp),
                bottom_price=_scalar(Bp),
                right_rim_price=_scalar(Rp),
                handle_low_price=_scalar(handle_low),
                breakout_close=_scalar(closev[breakout_i]),
                depth_pct=_scalar(depth_pct * 100),
                handle_drop_pct=_scalar((Rp - handle_low) / Rp * 100),
                buy_point=_scalar(buy_point),
                stop_loss=_scalar(stop_loss),
                status="breakout",
            )

    return None


# =====================
# 스캔 러너
# =====================

def scan_symbols(symbols: List[str], p: Params) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            df = download_ohlcv(sym, p.years, p.interval)
            if df.empty:
                continue
            pat = detect_cup_handle(df, p)
            if pat is None:
                continue
            if getattr(p, "forming_only", False) and pat.status != "forming":
                continue
            if not getattr(p, "include_forming", False) and not getattr(p, "forming_only", False):
                # 기본은 breakout만 포함
                if pat.status != "breakout":
                    continue
            # 근접 breakout 필터 및 거리 계산
            # robust scalar extraction
            last_price = float(df["close"].iloc[-1]) if (p.trigger == "close") else float(df["high"].iloc[-1])
            dist_pct = None
            if pat.status == "forming":
                dist_pct = float((pat.buy_point - last_price) / pat.buy_point * 100.0)
                # dist_pct >= 0: 아직 buy_point 아래, 0보다 작으면 이미 돌파 또는 초과
                if p.near_breakout_pct is not None:
                    # 근접 필터: 0 이상이면서 near 이내인 것만 통과
                    if not (dist_pct >= 0 and dist_pct <= float(p.near_breakout_pct)):
                        continue
            rows.append({
                "symbol": sym,
                "status": pat.status,
                "cup_rim_left_date": pat.left_rim_date.date(),
                "bottom_date": pat.bottom_date.date(),
                "cup_rim_right_date": pat.right_rim_date.date(),
                "handle_low_date": pat.handle_low_date.date(),
                "breakout_date": (pat.breakout_date.date() if pat.breakout_date is not None else ""),
                "cup_rim_left_price": round(pat.left_rim_price, 4),
                "bottom_price": round(pat.bottom_price, 4),
                "cup_rim_right_price": round(pat.right_rim_price, 4),
                "handle_low_price": round(pat.handle_low_price, 4),
                "breakout_close": (round(pat.breakout_close, 4) if pat.breakout_close is not None else np.nan),
                "last_price": round(last_price, 4),
                # Safe scalar extraction even if index has duplicates
                "breakout_price": (
                    round(
                        float(
                            df.loc[[pat.breakout_date], ("high" if p.trigger == "high" else "close")].iloc[0]
                        ),
                        4
                    ) if pat.breakout_date is not None else np.nan
                ),
                "trigger_basis": p.trigger,
                "depth_pct": round(pat.depth_pct, 3),
                "handle_drop_pct": round(pat.handle_drop_pct, 3),
                "buy_point": round(pat.buy_point, 4),
                "stop_loss": round(pat.stop_loss, 4),
                "distance_to_buy_pct": (round(dist_pct, 3) if dist_pct is not None else np.nan),
            })
        except Exception as e:
            tb = traceback.format_exc(limit=5)
            print(f"[WARN] {sym} 처리 중 오류: {e}\n       ↳ {tb.strip()}")
            continue

    cols = [
        "symbol","status","cup_rim_left_date","bottom_date","cup_rim_right_date","handle_low_date","breakout_date",
        "cup_rim_left_price","bottom_price","cup_rim_right_price","handle_low_price","breakout_close",
        "last_price","breakout_price","trigger_basis",
        "depth_pct","handle_drop_pct","buy_point","stop_loss","distance_to_buy_pct"
    ]
    out = pd.DataFrame(rows, columns=cols)
    out = out.sort_values(["breakout_date","symbol"]).reset_index(drop=True)
    return out


# =====================
# 터미널 표 렌더러 (가독성 향상)
# =====================

# === Helper: column renaming for display/export ===
def _rename_columns_for_display(df: pd.DataFrame, compact: bool = True) -> pd.DataFrame:
    """
    Rename columns for display or export:
      - Remove 'rim' from cup columns, abbreviate as requested.
      - If compact, also abbreviate handle columns to 'h_*'.
      - Abbreviate distance_to_buy_pct to dist_pct.
    """
    if df is None or df.empty:
        return df
    rename_map = {
        "cup_rim_left_date": "c_left_date",
        "cup_rim_right_date": "c_right_date",
        "cup_rim_left_price": "c_left_price",
        "cup_rim_right_price": "c_right_price",
        "distance_to_buy_pct": "dist_pct",
    }
    if compact:
        rename_map.update({
            "handle_low_date": "h_date",
            "handle_low_price": "h_price",
            "handle_drop_pct": "h_drop_pct",
        })
    # else: keep handle_* columns as-is (do not abbreviate)
    return df.rename(columns=rename_map)


def render_terminal_table(df: pd.DataFrame, compact: bool = True) -> None:
    """
    터미널에서 보기 좋은 표 형태로 출력.
    - compact=True: handle 관련 컬럼을 약어로 바꿔 폭을 줄임 (요청: handle -> 'h')
    - tabulate가 설치되어 있으면 깔끔한 표(github 스타일), 없으면 pandas.to_string 사용
    """
    if df is None or df.empty:
        print("⚠️ 조건에 맞는 컵앤핸들 패턴이 없습니다.")
        return

    use_df = _rename_columns_for_display(df.copy(), compact=compact)

    preferred_order = [
        "symbol","status",
        "c_left_date","bottom_date","c_right_date","h_date","breakout_date",
        "c_left_price","bottom_price","c_right_price","h_price",
        "breakout_price","breakout_close","last_price","buy_point","stop_loss",
        "depth_pct","h_drop_pct","dist_pct","trigger_basis",
    ]
    cols = [c for c in preferred_order if c in use_df.columns] + [c for c in use_df.columns if c not in preferred_order]
    use_df = use_df[cols]

    def _fmt_num(x):
        try:
            if pd.isna(x):
                return ""
            # 백분율 후보
            if isinstance(x, (int, float)) and abs(x) < 1000:
                return f"{x:.3f}"
            # 큰 수는 천단위 구분자
            if isinstance(x, (int, float)):
                return f"{x:,.4f}"
        except Exception:
            pass
        return x

    for c in ["c_left_price","bottom_price","c_right_price","h_price",
              "breakout_price","breakout_close","last_price","buy_point","stop_loss",
              "depth_pct","h_drop_pct","dist_pct"]:
        if c in use_df.columns:
            use_df[c] = use_df[c].apply(_fmt_num)

    try:
        from tabulate import tabulate
        print(tabulate(use_df, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        with pd.option_context("display.max_rows", None,
                               "display.max_columns", None,
                               "display.width", 160,
                               "display.colheader_justify", "center"):
            print(use_df.to_string(index=False))


# =====================
# CLI
# =====================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cup&Handle 멀티심볼 스캐너")
    ap.add_argument("--symbols", type=str, default="",
                    help="콤마구분 심볼들 예: AAPL,MSFT,NVDA or 005930.KS")
    ap.add_argument("--symbols-file", type=str, default="",
                    help="심볼 목록 텍스트 파일 경로 (줄바꿈 구분)")
    ap.add_argument("--years", type=int, default=None, help="과거 조회 연수")
    ap.add_argument("--export", type=str, default="", help="CSV 저장 경로")

    ap.add_argument("--use-volume", action="store_true", help="거래량 필터 사용(20MA 대비 1.2배)")
    ap.add_argument("--min-cup", type=int, default=30, help="최소 컵 길이(봉) 기본=30")
    ap.add_argument("--max-cup", type=int, default=250, help="최대 컵 길이(봉) 기본=250")

    ap.add_argument("--trigger", type=str, choices=["close","high"], default="close",
                    help="브레이크아웃 판정 기준: close(종가) 또는 high(고가)")

    # === New: universe / timeframe / ranking ===
    ap.add_argument("--universe", type=str, choices=["custom","sp500","nasdaq100","crypto"], default=None,
                    help="심볼 소스 선택")
    ap.add_argument("--topn", type=int, default=None,
                    help="상위 N")
    ap.add_argument("--rank-by", type=str, choices=["dollarvol","marketcap"], default=None,
                    help="산정 기준")
    ap.add_argument("--timeframe", type=str, choices=["1d","1wk","1h"], default=None,
                    help="봉 간격")

    ap.add_argument("--include-forming", action="store_true", help="돌파 전(형성 중) 패턴도 함께 표시")
    ap.add_argument("--forming-only", action="store_true", help="형성 중 패턴만 표시")
    ap.add_argument("--forming-window", type=int, default=60, help="형성 중으로 인정할 우측림 이후 최대 봉 수(기본=60)")

    ap.add_argument("--prefer-earliest-rim", action="store_true", help="가장 이른 우측 컵림을 우선(형성 초기 판단)")
    ap.add_argument("--rim-eps-up", type=float, default=None, help="우측 컵림이 좌측보다 높아도 허용하는 비율(기본=0.02)")

    ap.add_argument("--debug", action="store_true", help="디버그 로그 출력")
    ap.add_argument("--no-compact", action="store_true", help="약어 컬럼 없이 전체 컬럼명 사용")

    ap.add_argument("--near", type=float, default=None,
                    help="형성중 패턴 중에서, buy_point 대비 현재가가 이 값(%) 이내로 근접한 종목만 표시. 예) --near 1.0")
    ap.add_argument("--split-status", action="store_true",
                    help="터미널 출력 시 '형성중'과 '돌파완료'를 표로 분리해 보여줍니다.")
    return ap.parse_args()


def load_symbols(args: argparse.Namespace, p: Params) -> List[str]:
    # 1) universe에서 가져오기 (Params 우선)
    base: List[str] = []
    universe = (p.universe or "custom").lower()
    if universe != "custom":
        base = build_universe(universe)

    # 2) 직접 입력/파일 병합
    direct: List[str] = []
    if args.symbols:
        direct += [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    direct.append(s)

    symbols: List[str] = base + direct

    # 3) 기본 워치리스트 (universe별 기본값 적용)
    if not symbols:
        if universe == "crypto":
            symbols = _default_crypto_watchlist()
        else:
            symbols = [
                "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","NFLX",
                "PLTR","CRM","COST","ADBE","JPM","WMT","ORCL","DIS","INTC","PYPL"
            ]

    # 중복 제거, 순서 유지
    symbols = list(dict.fromkeys(symbols))

    # 4) 상위 N 추리기(Params 기준)
    if getattr(p, "topn", 0):
        symbols = rank_topn(symbols, p.topn, p.rank_by, years=p.years, interval=p.interval)

    return symbols


def render_by_status(df: pd.DataFrame, compact: bool = True) -> None:
    if df is None or df.empty:
        print("⚠️ 조건에 맞는 컵앤핸들 패턴이 없습니다.")
        return
    forming = df[df["status"] == "forming"].copy()
    breakout = df[df["status"] == "breakout"].copy()

    if not forming.empty:
        print("\n[형성중(돌파 전)]")
        render_terminal_table(forming, compact=compact)
    else:
        print("\n[형성중(돌파 전)] 없음")

    if not breakout.empty:
        print("\n[돌파 완료]")
        render_terminal_table(breakout, compact=compact)
    else:
        print("\n[돌파 완료] 없음")


def main():
    args = parse_args()
    base = Params.default()
    p = Params(
        years=args.years if args.years is not None else base.years,
        use_volume_filter=bool(args.use_volume) if args.use_volume else base.use_volume_filter,
        min_cup_bars=args.min_cup or base.min_cup_bars,
        max_cup_bars=args.max_cup or base.max_cup_bars,
        trigger=args.trigger or base.trigger,
        universe=args.universe or base.universe,
        topn=args.topn if args.topn is not None else base.topn,
        rank_by=args.rank_by or base.rank_by,
        interval=args.timeframe or base.interval,
        include_forming=bool(args.include_forming or args.forming_only) if (args.include_forming or args.forming_only) else base.include_forming,
        forming_only=bool(args.forming_only) if args.forming_only else base.forming_only,
        forming_max_age_bars=args.forming_window or base.forming_max_age_bars,
        prefer_earliest_right_rim=bool(args.prefer_earliest_rim) if args.prefer_earliest_rim else base.prefer_earliest_right_rim,
        rim_epsilon_up=(args.rim_eps_up if args.rim_eps_up is not None else base.rim_epsilon_up),
        near_breakout_pct=(args.near if args.near is not None else base.near_breakout_pct),
        display_split_status=bool(args.split_status) if args.split_status else base.display_split_status,
    )

    # === Enforce mutually-exclusive viewing as requested ===
    if p.forming_only:
        # 형성중만 보여주기
        p.include_forming = False
        p.display_split_status = False
    else:
        # 돌파완료만 보여주기 (기본)
        p.include_forming = False
        p.display_split_status = False

    symbols = load_symbols(args, p)
    if getattr(args, "debug", False):
        print("[DEBUG] Params:", p)
        print("[DEBUG] Resolved universe:", p.universe)
        print("[DEBUG] First 10 symbols:", symbols[:10])
    mode_txt = "forming-only" if p.forming_only else ("breakout+forming" if p.include_forming else "breakout-only")
    print(f"[SCAN] mode={mode_txt} | symbols={len(symbols)}개, years={p.years}, timeframe={p.interval}, volume_filter={p.use_volume_filter}, universe={p.universe}, topn={p.topn or 'ALL'}({p.rank_by})")
    if p.near_breakout_pct is not None:
        print(f"[FILTER] near breakout: buy_point 대비 ≤ {p.near_breakout_pct:.3f}% 근접한 '형성중'만 표시")
    df = scan_symbols(symbols, p)

    if df.empty:
        print("⚠️ 조건에 맞는 컵앤핸들 패턴이 없습니다.")
    else:
        # 단일 모드 출력: 형성중-only 또는 돌파-only
        if p.forming_only:
            df_out = df[df["status"] == "forming"].copy()
        else:
            df_out = df[df["status"] == "breakout"].copy()
        if df_out.empty:
            print("⚠️ 조건에 맞는 컵앤핸들 패턴이 없습니다.")
        else:
            render_terminal_table(df_out, compact=(not args.no_compact))

    export_path = args.export.strip()
    if export_path:
        df_to_save = _rename_columns_for_display(df.copy(), compact=(not args.no_compact)) if not df.empty else df
        df_to_save.to_csv(export_path, index=False)
        print(f"[SAVE] CSV 저장: {export_path}")


if __name__ == "__main__":
    main()