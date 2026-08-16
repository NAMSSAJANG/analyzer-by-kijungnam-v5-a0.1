from __future__ import annotations

from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


FALLBACK_NASDAQ = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "AVGO": "Broadcom", "TSLA": "Tesla",
    "COST": "Costco", "NFLX": "Netflix", "AMD": "AMD", "PLTR": "Palantir",
    "CSCO": "Cisco", "TMUS": "T-Mobile", "LIN": "Linde", "PEP": "PepsiCo",
    "INTU": "Intuit", "AMGN": "Amgen", "TXN": "Texas Instruments", "QCOM": "Qualcomm",
}
FALLBACK_SP500 = {
    **FALLBACK_NASDAQ, "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
    "V": "Visa", "MA": "Mastercard", "LLY": "Eli Lilly", "WMT": "Walmart",
    "XOM": "Exxon Mobil", "UNH": "UnitedHealth", "JNJ": "Johnson & Johnson",
    "HD": "Home Depot", "PG": "Procter & Gamble", "BAC": "Bank of America",
    "ABBV": "AbbVie", "KO": "Coca-Cola", "CRM": "Salesforce", "ORCL": "Oracle",
}
FALLBACK_KOSPI = {
    "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스", "005380.KS": "현대차", "000270.KS": "기아",
    "068270.KS": "셀트리온", "105560.KS": "KB금융", "035420.KS": "NAVER",
    "055550.KS": "신한지주", "005490.KS": "POSCO홀딩스", "012330.KS": "현대모비스",
    "028260.KS": "삼성물산", "066570.KS": "LG전자", "035720.KS": "카카오",
}


def _yahoo_symbol(symbol: str) -> str:
    return str(symbol).strip().replace(".", "-")


@st.cache_data(ttl=86400, show_spinner=False)
def index_universe(market: str) -> pd.DataFrame:
    """Return Symbol/Name pairs. Remote lists have small built-in fallbacks."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        if market == "NASDAQ 100":
            nasdaq_headers = {
                **headers, "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/",
            }
            payload = requests.get(
                "https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                headers=nasdaq_headers, timeout=20,
            ).json()
            rows = payload["data"]["data"]["rows"]
            return pd.DataFrame({
                "Symbol": [_yahoo_symbol(x["symbol"]) for x in rows],
                "Name": [x["companyName"] for x in rows],
            })
        if market == "S&P 500":
            html = requests.get(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=15
            ).text
            table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
            return pd.DataFrame({"Symbol": table["Symbol"].map(_yahoo_symbol), "Name": table["Security"]})
        import FinanceDataReader as fdr

        listing = fdr.StockListing("KRX")
        code_col = next(x for x in ("Code", "Symbol") if x in listing.columns)
        kospi = listing[listing["Market"].astype(str).str.upper().eq("KOSPI")].copy()
        # Very illiquid micro-caps make a free-data daily scan unstable. Use the
        # largest 500 KOSPI companies when market-cap data is available.
        if "Marcap" in kospi.columns:
            kospi = kospi.sort_values("Marcap", ascending=False).head(500)
        return pd.DataFrame({
            "Symbol": kospi[code_col].astype(str).str.zfill(6) + ".KS",
            "Name": kospi["Name"].astype(str),
        })
    except Exception:
        fallback = (
            FALLBACK_NASDAQ if market == "NASDAQ 100"
            else FALLBACK_SP500 if market == "S&P 500"
            else FALLBACK_KOSPI
        )
        return pd.DataFrame([{"Symbol": k, "Name": v} for k, v in fallback.items()])


def _scale(value, low, high):
    if not np.isfinite(value):
        return np.nan
    return float(np.clip((value - low) / (high - low) * 100, 0, 100))


def _weighted(parts):
    valid = [(v, w) for v, w in parts if np.isfinite(v)]
    return float(sum(v * w for v, w in valid) / sum(w for _, w in valid)) if valid else 50.0


def _one_score(d: pd.DataFrame, market_health: float):
    d = d.dropna(subset=["Close"]).copy()
    if len(d) < 210 or "Volume" not in d:
        return None
    c = d["Close"].astype(float)
    v = d["Volume"].fillna(0).astype(float)
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    ret = lambda n: (c.iloc[-1] / c.iloc[-min(n, len(c))] - 1) * 100
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + up / down.replace(0, np.nan))).iloc[-1])
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat(
        [d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1
    ).max(axis=1)
    atr_pct = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1] / c.iloc[-1] * 100
    typical = (d["High"] + d["Low"] + c) / 3
    vwap = (typical * v).cumsum().iloc[-1] / max(v.cumsum().iloc[-1], 1)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    vol_ratio = v.iloc[-1] / max(v.tail(20).mean(), 1)
    z = (c.iloc[-1] - c.tail(60).mean()) / max(c.tail(60).std(), 1e-9)
    trend = _weighted([
        (_scale(c.iloc[-1] / ema200.iloc[-1] - 1, -.15, .25), .35),
        (_scale(ema20.iloc[-1] / ema50.iloc[-1] - 1, -.08, .10), .30),
        (_scale(ret(252), -30, 60), .35),
    ])
    momentum = _weighted([
        (_scale(rsi, 35, 75), .30),
        (_scale(ret(63), -20, 30), .35),
        (_scale(macd.iloc[-1] - signal.iloc[-1], -c.iloc[-1] * .015, c.iloc[-1] * .015), .35),
    ])
    volatility = float(np.clip(100 - (atr_pct - 1) * 18, 0, 100))
    supply = _weighted([
        (_scale(vol_ratio, .5, 2), .45),
        (_scale(obv.iloc[-1] / max(abs(obv.tail(60)).max(), 1), -.5, 1), .35),
        (_scale(c.iloc[-1] / vwap - 1, -.08, .12), .20),
    ])
    overheat = max(0, (rsi - 68) * 2.4) + max(0, z - 1.5) * 12
    score = float(np.clip(
        _weighted([(trend, .34), (momentum, .22), (volatility, .14), (supply, .20), (market_health, .10)]) - overheat,
        0, 100,
    ))
    day_change = float((c.iloc[-1] / c.iloc[-2] - 1) * 100)
    return score, day_change, c.index[-1]


def _history_for(raw: pd.DataFrame, symbol: str, single: bool) -> pd.DataFrame:
    if single:
        return raw
    try:
        if symbol in raw.columns.get_level_values(0):
            return raw[symbol]
        return raw.xs(symbol, axis=1, level=1)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def daily_top10(market: str, market_health: float) -> tuple[pd.DataFrame, str]:
    universe = index_universe(market).drop_duplicates("Symbol")
    names = dict(zip(universe["Symbol"], universe["Name"]))
    rows = []
    symbols = universe["Symbol"].tolist()
    for start in range(0, len(symbols), 80):
        chunk = symbols[start:start + 80]
        try:
            raw = yf.download(
                chunk, period="18mo", interval="1d", auto_adjust=True,
                progress=False, threads=True, group_by="ticker", timeout=25,
            )
        except Exception:
            continue
        single = len(chunk) == 1 and not isinstance(raw.columns, pd.MultiIndex)
        for symbol in chunk:
            result = _one_score(_history_for(raw, symbol, single), market_health)
            if result is None:
                continue
            score, change, data_date = result
            rows.append({"Symbol": symbol, "Name": names.get(symbol, symbol), "Score": score, "Change": change, "Date": data_date})
    if not rows:
        return pd.DataFrame(), "-"
    ranked = pd.DataFrame(rows).sort_values(["Score", "Change"], ascending=False).head(10).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1))
    date_value = pd.to_datetime(ranked["Date"], errors="coerce").max()
    as_of = date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else datetime.now().strftime("%Y-%m-%d")
    return ranked, as_of


def entry_label(score: float) -> str:
    if score >= 80:
        return "진입 우위"
    if score >= 65:
        return "진입 검토"
    if score >= 45:
        return "관망·조건 확인"
    if score >= 30:
        return "진입 보류"
    return "진입 회피"


def render_top10(market_health: float):
    st.subheader("오늘의 퀀트 TOP 10")
    market = st.radio(
        "시장 선택", ["NASDAQ 100", "S&P 500", "KOSPI"],
        horizontal=True, label_visibility="collapsed", key="top10_market",
    )
    with st.spinner(f"{market} 퀀트 순위를 계산하는 중입니다. 최초 조회는 시간이 걸릴 수 있습니다..."):
        ranked, as_of = daily_top10(market, round(float(market_health), 1))
    if ranked.empty:
        st.info("현재 순위 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")
        return
    shown = ranked.assign(
        종목=ranked["Name"] + " · " + ranked["Symbol"],
        **{"퀀트 점수": ranked["Score"].map(lambda x: f"{x:.1f}"),
           "진입 적합도": ranked["Score"].map(entry_label),
           "당일 등락": ranked["Change"].map(lambda x: f"{x:+.2f}%")},
    ).rename(columns={"Rank": "순위"})[["순위", "종목", "퀀트 점수", "진입 적합도", "당일 등락"]]
    st.caption(f"{as_of} 종가 기준 · 매일 1회 갱신 · 행을 선택하면 해당 종목을 분석합니다.")
    event = st.dataframe(
        shown, hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row", key=f"top10_table_{market}",
        column_config={
            "순위": st.column_config.NumberColumn(width="small"),
            "종목": st.column_config.TextColumn(width="large"),
            "퀀트 점수": st.column_config.TextColumn(width="small"),
            "진입 적합도": st.column_config.TextColumn(width="medium"),
            "당일 등락": st.column_config.TextColumn(width="small"),
        },
    )
    selected = event.selection.rows if event and hasattr(event, "selection") else []
    if selected:
        row = ranked.iloc[selected[0]]
        st.session_state["symbol"] = row["Symbol"]
        st.caption(f"선택 종목 · {row['Name']} · {row['Symbol']}")
    if market == "KOSPI":
        st.caption("KOSPI는 안정적인 무료 데이터 조회를 위해 시가총액 상위 500개 종목을 대상으로 산정합니다.")


