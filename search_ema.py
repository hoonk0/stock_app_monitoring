from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import argparse
import time

# 외부 라이브러리
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message="Calling float on a single element Series is deprecated",
    category=FutureWarning,
)

# ccxt, yfinance는 환경에 따라 미설치일 수 있음 → 안전 import
try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None

try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

# =========================
# 옵션 (맨 위에서 수정 가능)
# =========================
CONFIG: Dict[str, object] = {
    "MARKET_TYPE": "nasdaq",      # 'coin' | 'nasdaq' | 'snp'
    "MARKETCAP_TOPN": 500,        # 상위 몇 개 스캔
    "USE_LAST_CANDLE": True,     # 진행 중 캔들 포함할지 여부
}

EMA_PERIODS: List[int] = [5, 10, 20, 60, 120]

# 나스닥/S&P 프리셋(대표 대형주, 필요시 자유롭게 추가)
PRESET_NASDAQ_100: List[str] = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","COST","TSLA","PEP",
    "ADBE","AMD","NFLX","AMAT","QCOM","INTC","LIN","CSCO","TXN","PDD",
    "HON","INTU","TMUS","ISRG","SBUX","AMGN","MU","MDLZ","BKNG","REGN",
    "PANW","ADP","CHTR","GILD","VRTX","LRCX","ABNB","KDP","ADSK","PYPL",
]

PRESET_SNP_50: List[str] = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","JPM",
    "XOM","V","WMT","JNJ","PG","MA","COST","UNH","HD","CVX",
    "ABBV","MRK","PEP","KO","BAC","ORCL","CRM","ADBE","TMO","ACN",
    "CSCO","LIN","MCD","INTC","WFC","DHR","AMD","TXN","PM","IBM",
    "AMGN","CAT","HON","MS","GE","BKNG","RTX","PFE","COP","LOW",
]

# =========================
# 유틸
# =========================

def now_kst_str() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(tz=kst).strftime("%Y-%m-%d %H:%M:%S")


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def detect_price_cross_ema(df: pd.DataFrame, period: int, use_last: bool = True) -> Optional[str]:
    """df: columns=['close'] (필수)
    return: 'bullish' | 'bearish' | None (교차 없음)
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    if len(df) < period + 2:
        return None

    close = df["close"].astype(float).reset_index(drop=True)
    ema = _ema(close, period)

    # 평가 인덱스 선택 (마지막 or 직전)
    idx = len(close) - 1 if use_last else len(close) - 2
    if idx < 1:
        return None

    prev_idx = idx - 1
    prev_close, curr_close = close.iloc[prev_idx], close.iloc[idx]
    prev_ema, curr_ema = ema.iloc[prev_idx], ema.iloc[idx]

    if np.isnan(prev_ema) or np.isnan(curr_ema):
        return None

    # 상승돌파: 아래→위 교차
    if (prev_close <= prev_ema) and (curr_close > curr_ema):
        return "bullish"
    # 하락돌파: 위→아래 교차
    if (prev_close >= prev_ema) and (curr_close < curr_ema):
        return "bearish"

    return None


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def current_ma_state_5_20(df: pd.DataFrame, use_last: bool = True) -> Optional[str]:
    """현재 5MA vs 20MA 상태를 반환
    return: 'golden' | 'dead' | None
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = df["close"].astype(float).reset_index(drop=True)
    if len(close) < 25:
        return None
    ma5 = _sma(close, 5)
    ma20 = _sma(close, 20)
    idx = len(close) - 1 if use_last else len(close) - 2
    if idx < 0 or np.isnan(ma5.iloc[idx]) or np.isnan(ma20.iloc[idx]):
        return None
    if ma5.iloc[idx] > ma20.iloc[idx]:
        return "golden"
    if ma5.iloc[idx] < ma20.iloc[idx]:
        return "dead"
    return None


# =========================
# 심볼 로딩
# =========================

def load_symbols_from_config(market: str, topn: int) -> List[str]:
    market = market.lower()
    if market == "coin":
        if ccxt is None:
            print("[WARN] ccxt 미설치로 코인 심볼을 불러올 수 없습니다. pip install ccxt")
            return []
        try:
            ex = ccxt.binance({"enableRateLimit": True})
            # 현물 마켓 로딩
            ex.load_markets()
            # 유동성 판단 위해 tickers 조회 (조금 느릴 수 있음)
            tickers = ex.fetch_tickers()
            pairs = []
            for sym, t in tickers.items():
                # USDT 현물 위주, 선물/레버리지/마진 제외(간단 필터)
                if not sym.endswith("/USDT"):
                    continue
                # 거래대금 근사: quoteVolume 또는 baseVolume*last
                quote_vol = None
                if isinstance(t, dict):
                    quote_vol = t.get("quoteVolume") or (t.get("baseVolume") or 0) * (t.get("last") or 0)
                if quote_vol is None:
                    continue
                pairs.append((sym, float(quote_vol)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [s for s, _ in pairs[:topn]]
        except Exception as e:
            print(f"[ERR] 심볼 로딩 실패(coin): {e}")
            return []

    elif market == "nasdaq":
        # 우선 프리셋 사용 후, 부족하면 yfinance의 tickers_nasdaq()로 보충
        syms = PRESET_NASDAQ_100.copy()
        if topn > len(syms):
            try:
                if yf is not None and hasattr(yf, "tickers_nasdaq"):
                    big = yf.tickers_nasdaq()  # 수천 개 반환(알파벳 정렬)
                    # 파생/우선주/권리락 티커 간단 필터: '.' 포함/긴 티커 제외
                    clean = [s for s in big if ("." not in s) and (len(s) <= 5)]
                    # 중복 제거 + 프리셋 우선 유지
                    seen = set(syms)
                    for s in clean:
                        if s not in seen:
                            syms.append(s)
                            seen.add(s)
                            if len(syms) >= topn:
                                break
            except Exception as e:
                print(f"[WARN] 나스닥 확장 목록 로딩 실패: {e}")
        return syms[:topn]
    elif market == "snp":
        try:
            if yf is not None and hasattr(yf, "tickers_sp500"):
                sp = yf.tickers_sp500()
                # 권리락/우선주 등 간단 필터
                sp = [s for s in sp if ("." not in s)]
                return sp[:topn]
        except Exception as e:
            print(f"[WARN] S&P500 목록 로딩 실패: {e}")
        return PRESET_SNP_50[:topn]
    else:
        return []


# =========================
# 데이터 로딩
# =========================

def fetch_ohlcv_coin(symbol: str, timeframe: str = "1d", limit: int = 500) -> List[List[float]]:
    if ccxt is None:
        raise RuntimeError("ccxt가 설치되어 있지 않습니다.")
    ex = ccxt.binance({"enableRateLimit": True})
    # 일부 심볼은 바로 호출 시 에러 → 재시도 간단 구현
    for i in range(3):
        try:
            o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return o
        except Exception as e:
            if i == 2:
                raise
            time.sleep(0.8)
    return []


def fetch_ohlcv_stock(symbol: str, period_days: int = 500) -> List[List[float]]:
    if yf is None:
        return []
    # 최대 3회 재시도, 개별 실패는 조용히 스킵
    last_err = None
    for _ in range(3):
        try:
            df = yf.download(symbol, period="2y", interval="1d", progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                return []
            df = df.dropna()
            ohlcv = []

            def _scalar(x):
                # yfinance가 가끔 단일 원소 Series를 반환하는 경우가 있어 안전 변환
                if isinstance(x, pd.Series):
                    if not x.empty:
                        x = x.iloc[0]
                    else:
                        return np.nan
                if isinstance(x, np.generic):  # numpy scalar
                    x = x.item()
                return x

            for ts, row in df.iterrows():
                ts_ms = int(pd.Timestamp(ts).timestamp() * 1000)
                open_v = _scalar(row.get("Open", np.nan))
                high_v = _scalar(row.get("High", np.nan))
                low_v  = _scalar(row.get("Low", np.nan))
                close_v= _scalar(row.get("Close", np.nan))
                vol_v  = _scalar(row.get("Volume", 0))
                ohlcv.append([
                    ts_ms,
                    float(open_v) if pd.notna(open_v) else np.nan,
                    float(high_v) if pd.notna(high_v) else np.nan,
                    float(low_v)  if pd.notna(low_v)  else np.nan,
                    float(close_v)if pd.notna(close_v)else np.nan,
                    float(vol_v)  if pd.notna(vol_v)  else 0.0,
                ])
            return ohlcv[-period_days:]
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    # 최종 실패 시 경고만 출력하고 빈 리스트 반환
    print(f"[WARN] 주식 데이터 로딩 실패: {symbol} ({last_err})")
    return []


def fetch_ohlcv_any(market: str, symbol: str, timeframe: str = "1d", limit: int = 500) -> List[List[float]]:
    if market == "coin":
        return fetch_ohlcv_coin(symbol, timeframe=timeframe, limit=limit)
    else:  # nasdaq / snp → 주식
        return fetch_ohlcv_stock(symbol, period_days=limit)


# =========================
# 메인 스캔
# =========================

def scan_crosses(market: str, symbols: List[str], use_last_candle: bool = True) -> Dict[int, Dict[str, List[str]]]:
    # 결과 구조: {period: {"bullish": [...], "bearish": [...]}}
    result: Dict[int, Dict[str, List[str]]] = {p: {"bullish": [], "bearish": []} for p in EMA_PERIODS}

    for sym in symbols:
        try:
            # 간단 페이싱: 50개마다 잠시 대기(무료 API 안정성)
            i = symbols.index(sym)
            if i > 0 and (i % 50 == 0):
                time.sleep(1.5)
            ohlcv = fetch_ohlcv_any(market, sym, timeframe="1d", limit=500)
            if not ohlcv or len(ohlcv) < 130:  # 120EMA 안정화 여유
                continue
            df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"]).dropna()

            state = current_ma_state_5_20(df, use_last=use_last_candle)
            def _with_state(name: str) -> str:
                if state == "golden":
                    return f"{name} (golden cross)"
                if state == "dead":
                    return f"{name} (dead cross)"
                return name

            for p in EMA_PERIODS:
                sig = detect_price_cross_ema(df, p, use_last=use_last_candle)
                if sig == "bullish":
                    result[p]["bullish"].append(_with_state(sym))
                elif sig == "bearish":
                    result[p]["bearish"].append(_with_state(sym))
        except Exception:
            print(f"[WARN] 스킵: {sym}")
            continue

    return result


def print_result_table(market: str, topn: int, symbols: List[str], res: Dict[int, Dict[str, List[str]]]):
    print("="*64)
    print(f"[EMA 돌파 스캔 결과] 기준(KST): {now_kst_str()}  |  시장={market}  |  종목수={len(symbols)} (topN={topn})")
    print("="*64)
    for p in EMA_PERIODS:
        bull = res[p]["bullish"]
        bear = res[p]["bearish"]
        print(f"\n▶ {p}일 EMA")
        print("  [상승돌파]", ", ".join(bull) if bull else "없음")
        print("  [하락돌파]", ", ".join(bear) if bear else "없음")
    print()


# =========================
# 엔트리 포인트
# =========================

def main():
    parser = argparse.ArgumentParser(description="EMA(5/10/20/60/120) 돌파 스캐너")
    parser.add_argument("--market", choices=["coin","nasdaq","snp"], default=str(CONFIG["MARKET_TYPE"]))
    parser.add_argument("--topn", type=int, default=int(CONFIG["MARKETCAP_TOPN"]))
    parser.add_argument("--use-last-candle", dest="use_last_candle", action="store_true")
    parser.add_argument("--no-last-candle", dest="use_last_candle", action="store_false")
    parser.add_argument("--symbols", type=str, default="", help=",로 구분 (예: BTC/USDT,ETH/USDT | AAPL,MSFT)")
    parser.set_defaults(use_last_candle=bool(CONFIG["USE_LAST_CANDLE"]))
    args = parser.parse_args()

    market = args.market.lower()
    topn = max(1, args.topn)
    use_last = bool(args.use_last_candle)

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols_from_config(market, topn)

    if not symbols:
        print(f"[결과] 스캔 대상 심볼이 없습니다. (market={market}, topN={topn})")
        return

    res = scan_crosses(market, symbols, use_last_candle=use_last)
    print_result_table(market, topn, symbols, res)


if __name__ == "__main__":
    main()
