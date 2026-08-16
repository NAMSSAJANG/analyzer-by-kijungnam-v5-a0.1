from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from consensus_engine import Lens, build_consensus, confidence_interpretation
from entry_engine import build_entry_snapshot
from korean_stock_search import contains_hangul, load_krx_listing, search_krx_listing
from score_history import JsonScoreHistory, format_trend

st.set_page_config(page_title="Stock Analyzer V5-a0.1", page_icon="📈", layout="wide")

GRADE = [(80, "Strong"), (65, "Good"), (45, "Neutral"), (30, "Weak"), (-1, "Very Weak")]
PULSE = {
    "S&P 500 (미국 대형주)": "^GSPC", "Nasdaq 100 (미국 기술주)": "^NDX",
    "SOX (반도체 지수)": "^SOX", "VIX (공포지수)": "^VIX",
    "Gold (금)": "GC=F", "Silver (은)": "SI=F", "WTI (미국 원유)": "CL=F", "Copper (경기민감 구리)": "HG=F",
    "USD/KRW (원·달러)": "KRW=X", "DXY (달러지수)": "DX-Y.NYB", "Bitcoin (비트코인)": "BTC-USD", "Ethereum (이더리움)": "ETH-USD",
}
RATE_GUIDE = {
    "US 2Y": "연준 정책 기대에 민감한 단기 국채금리",
    "US 5Y": "중기 성장·물가 기대를 반영하는 국채금리",
    "US 10Y": "장기 성장과 주식 할인율의 기준 금리",
    "US 30Y": "초장기 물가·재정 부담을 반영하는 국채금리",
    "HYG": "미국 하이일드 회사채 ETF · 위험선호 참고",
    "LQD": "미국 투자등급 회사채 ETF · 우량 신용시장 참고",
    "10Y-2Y": "장단기 금리차 · 경기 사이클 참고",
    "Credit Spread proxy": "HYG와 LQD의 20일 상대성과 · 신용 위험선호 대용값",
}
HISTORY_FILE = Path(os.getenv("ANALYZER_HISTORY_FILE", ".data/score_history.json"))
HISTORY_STORE = JsonScoreHistory(HISTORY_FILE)

st.markdown("""
<style>
.stApp{background:linear-gradient(180deg,#07111f 0%,#081525 100%)}
.block-container{padding-top:1.2rem;max-width:1440px}.small{color:#94a3b8;font-size:.86rem}.blue{color:#38bdf8;font-size:.9rem}
.card{border:1px solid #29415e;border-radius:14px;padding:14px;background:#0d1b2d;min-height:132px}
.badge{display:inline-block;padding:3px 9px;border-radius:99px;background:#102c46;color:#7dd3fc;font-weight:700}
.brief-card{box-sizing:border-box;border:1px solid #29415e;border-radius:15px;padding:18px;background:#0d1b2d;height:calc(100% - 14px);min-height:190px;color:#dbeafe;margin-bottom:14px}
.brief-card.wide{min-height:168px}.brief-card.matched{height:220px;min-height:220px}.brief-kicker{color:#38bdf8;font-size:.72rem;font-weight:800;letter-spacing:.16em;margin-bottom:8px}
.brief-card h3{color:#f8fafc;margin:.1rem 0 1rem;font-size:1.35rem}.brief-card p{line-height:1.8;margin:0;color:#dbeafe}
.scenario{box-sizing:border-box;border-radius:12px;padding:16px;min-height:145px;height:145px;overflow:auto}.scenario h4{margin:0 0 16px}.scenario p{margin:0;line-height:1.75}
.up{background:#103c30;color:#6ee7b7}.mid{background:#102d4d;color:#7dd3fc}.down{background:#451d28;color:#fda4af}
.score-card{box-sizing:border-box;border:1px solid #29415e;border-radius:16px;padding:18px 18px 16px;background:#0d1b2d;min-height:150px;height:100%}
.score-label{color:#94a3b8;font-size:.78rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;margin-bottom:8px}
.score-value{color:#f8fafc;font-size:clamp(1.65rem,2.4vw,2.25rem);font-weight:800;line-height:1.15;white-space:nowrap}.score-denom{color:#94a3b8;font-size:.9rem;font-weight:700}
.score-track{height:9px;border-radius:99px;background:#223149;margin:18px 0 13px;overflow:hidden}.score-fill{height:100%;border-radius:99px;min-width:2px}
.score-grade{font-size:.92rem;font-weight:750;display:flex;align-items:center;gap:7px}.score-dot{width:11px;height:11px;border-radius:50%;display:inline-block;flex:none}
.consensus{border:1px solid #315272;border-radius:18px;padding:18px;background:linear-gradient(135deg,#0d1b2d,#10243a);margin:14px 0}.consensus-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.lens{background:#0a1728;border:1px solid #29415e;border-radius:12px;padding:11px}.lens small{color:#94a3b8;display:block}.lens b{color:#f8fafc}.lens-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px;box-shadow:0 0 9px currentColor}.consensus-summary{margin-top:13px;color:#dbeafe;line-height:1.65}.consensus-lines{margin-top:12px;border-top:1px solid #29415e;padding-top:10px;display:grid;gap:6px}.consensus-line{display:flex;gap:10px;align-items:center}.consensus-line span:first-child{width:48px;color:#94a3b8}.consensus-line b{color:#f8fafc}
[data-testid="stMetricValue"]{font-size:clamp(1.55rem,3vw,2.35rem);white-space:normal;overflow-wrap:anywhere}
[data-testid="stMetricLabel"]{color:#cbd5e1} [data-testid="stExpander"]{border-color:#29415e;background:#0a1728}
[data-testid="stAlert"]{border-radius:12px} div[data-testid="stPlotlyChart"]{overflow:hidden}
@media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}.brief-card,.brief-card.wide,.brief-card.matched{height:auto;min-height:0}.scenario{height:auto;min-height:130px}}
@media(max-width:700px){
html,body,#root{
  height:auto!important;min-height:100%!important;overflow:visible!important;background:#07111f!important
}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{
  height:auto!important;max-height:none!important;min-height:100svh!important;
  overflow:visible!important;background:#07111f!important
}
[data-testid="stHorizontalBlock"]{flex-wrap:wrap;gap:.75rem}
[data-testid="column"]{min-width:45%!important;flex:1 1 45%!important}
div[role="radiogroup"]{display:grid!important;grid-template-columns:repeat(4,minmax(0,1fr));gap:.25rem;width:100%}
div[role="radiogroup"] label{min-width:0!important;padding:.42rem .15rem!important;justify-content:center}
div[role="radiogroup"] label p{font-size:.72rem!important;white-space:nowrap}
.consensus-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.consensus{padding:13px}.lens{padding:9px}
.score-card{height:auto!important;min-height:0!important;margin-bottom:14px!important}
.scenario{height:auto!important;min-height:0!important;margin:0 0 14px!important;position:relative}
}
</style>""", unsafe_allow_html=True)

def grade(v: float) -> str:
    return next(label for floor, label in GRADE if v >= floor)

def score_color(v: float) -> str:
    if v >= 80: return "#22c55e"
    if v >= 65: return "#34d399"
    if v >= 45: return "#94a3b8"
    if v >= 30: return "#f97316"
    return "#ef4444"

def clamp(v): return float(max(0, min(100, v)))
def safe(v, default=0): return default if v is None or (isinstance(v, float) and np.isnan(v)) else v
def pct(v): return f"{v:+.2f}%"
def money(v):
    if not np.isfinite(v): return "-"
    return f"{v:,.2f}" if abs(v) < 10000 else f"{v:,.0f}"

@st.cache_data(ttl=86400, show_spinner=False)
def krx_listing():
    """한국거래소 목록을 단계별 원격 소스와 내장 안전망에서 불러옵니다."""
    return load_krx_listing()

@st.cache_data(ttl=300, show_spinner=False)
def search_yahoo(q: str):
    q=q.strip()
    if not q: return []
    merged=[]
    # 한글 회사명과 6자리 종목코드는 KRX 목록을 우선 검색합니다.
    is_hangul=contains_hangul(q)
    if is_hangul or re.fullmatch(r"\d{1,6}",q):
        try:
            merged.extend(search_krx_listing(q,krx_listing()))
        except Exception: pass
    try:
        data=requests.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":q,"quotesCount":10,"newsCount":0},headers={"User-Agent":"Mozilla/5.0"},timeout=8).json()
        merged.extend({"symbol":x.get("symbol"),"name":x.get("longname") or x.get("shortname") or x.get("symbol"),"exchange":x.get("exchDisp",""),"type":x.get("quoteType","")} for x in data.get("quotes",[]) if x.get("symbol") and not contains_hangul(x.get("symbol")))
    except Exception: pass
    if not merged and re.fullmatch(r"\d{6}",q):
        merged=[{"symbol":f"{q}.KS","name":f"한국 종목 {q}","exchange":"KOSPI 후보","type":"Equity"},{"symbol":f"{q}.KQ","name":f"한국 종목 {q}","exchange":"KOSDAQ 후보","type":"Equity"}]
    if not merged and not is_hangul: merged=[{"symbol":q.upper(),"name":q.upper(),"exchange":"직접 입력","type":""}]
    seen=set(); unique=[]
    for row in merged:
        if row["symbol"] not in seen: seen.add(row["symbol"]); unique.append(row)
    return unique[:10]

@st.cache_data(ttl=900, show_spinner=False)
def prices(symbol: str, period="2y", interval="1d"):
    d = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return d.dropna(how="all")

@st.cache_data(ttl=21600, show_spinner=False)
def fred_yields() -> pd.DataFrame:
    """Fetch all Treasury maturities in one short FRED request.

    Streamlit Cloud can occasionally delay or block FRED.  A single request keeps
    that failure bounded instead of waiting once for every maturity.
    """
    series_ids=("DGS2","DGS5","DGS10","DGS30")
    url=f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={','.join(series_ids)}"
    response=requests.get(url,timeout=5,headers={"User-Agent":"StockAnalyzer/5"}); response.raise_for_status()
    frame=pd.read_csv(StringIO(response.text))
    date_column="observation_date" if "observation_date" in frame.columns else "DATE"
    if date_column not in frame.columns: raise ValueError("FRED date column is missing")
    frame[date_column]=pd.to_datetime(frame[date_column],errors="coerce")
    for series_id in series_ids:
        if series_id not in frame.columns: frame[series_id]=np.nan
        frame[series_id]=pd.to_numeric(frame[series_id],errors="coerce")
    return frame.set_index(date_column)[list(series_ids)].dropna(how="all")

@st.cache_data(ttl=21600, show_spinner=False)
def treasury_yields() -> pd.DataFrame:
    """Official U.S. Treasury par yields, including the actual 2-year yield."""
    year=datetime.now(timezone.utc).year
    url="https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    response=requests.get(url,params={"data":"daily_treasury_yield_curve","field_tdr_date_value":year},
                          timeout=5,headers={"User-Agent":"StockAnalyzer/5"}); response.raise_for_status()
    rows=[]
    field_map={"BC_2YEAR":"DGS2","BC_5YEAR":"DGS5","BC_10YEAR":"DGS10","BC_30YEAR":"DGS30"}
    for properties in ET.fromstring(response.content).iter():
        if properties.tag.rsplit("}",1)[-1] != "properties": continue
        values={child.tag.rsplit("}",1)[-1]: child.text for child in properties}
        date=values.get("NEW_DATE") or values.get("Date")
        if not date: continue
        row={"date":pd.to_datetime(date,errors="coerce")}
        row.update({target:pd.to_numeric(values.get(source),errors="coerce") for source,target in field_map.items()})
        rows.append(row)
    if not rows: raise ValueError("Treasury yield rows are missing")
    return pd.DataFrame(rows).set_index("date").sort_index().dropna(how="all")

@st.cache_data(ttl=3600, show_spinner=False)
def info(symbol):
    try: return yf.Ticker(symbol).get_info()
    except Exception: return {}

@st.cache_data(ttl=1800, show_spinner=False)
def news(symbol):
    try:
        rows = yf.Ticker(symbol).news[:8]
        out=[]
        for x in rows:
            c=x.get("content", x); title=c.get("title") or x.get("title")
            if title: out.append((title, c.get("summary", ""), (c.get("canonicalUrl") or {}).get("url") or x.get("link", "")))
        return out
    except Exception: return []

def spark(symbol):
    try:
        d=prices(symbol,"5d","15m"); s=d["Close"].dropna(); day=s.index[-1].date(); s=s[s.index.date==day]
        prev=prices(symbol,"5d","1d")["Close"].dropna(); base=float(prev.iloc[-2] if len(prev)>1 else s.iloc[0])
        return s, base
    except Exception: return pd.Series(dtype=float), np.nan

def calc(symbol, as_of=None, info_override=None):
    d=prices(symbol)
    if as_of is not None: d=d.loc[pd.to_datetime(d.index).date<=pd.Timestamp(as_of).date()]
    c=d["Close"].dropna(); vol=d.get("Volume", pd.Series(index=c.index,dtype=float)).reindex(c.index)
    if len(c)<210: raise ValueError("최소 200거래일 이상의 가격 데이터가 필요합니다.")
    now=float(c.iloc[-1]); ma20=c.rolling(20).mean(); ma50=c.rolling(50).mean(); ma200=c.rolling(200).mean()
    ret=lambda n: float((now/c.iloc[-min(n,len(c))]-1)*100)
    slope50=(ma50.iloc[-1]/ma50.iloc[-21]-1)*100; slope200=(ma200.iloc[-1]/ma200.iloc[-21]-1)*100
    long=clamp(50+1.1*(now/ma200.iloc[-1]-1)*100+.8*(ma50.iloc[-1]/ma200.iloc[-1]-1)*100+.8*slope50+.5*slope200+.18*ret(126)+.12*ret(252))
    rsi_delta=c.diff(); gain=rsi_delta.clip(lower=0).rolling(14).mean(); loss=-rsi_delta.clip(upper=0).rolling(14).mean(); rsi=float(100-100/(1+gain.iloc[-1]/max(loss.iloc[-1],1e-9)))
    short=clamp(50+1.2*(now/ma20.iloc[-1]-1)*100+.7*(ma20.iloc[-1]/ma50.iloc[-1]-1)*100+(rsi-50)*.35+.18*ret(20))
    inf=info_override if info_override is not None else info(symbol); pe=safe(inf.get("trailingPE"),25); growth=safe(inf.get("revenueGrowth"),0)*100; margin=safe(inf.get("profitMargins"),0)*100; roe=safe(inf.get("returnOnEquity"),0)*100; debt=safe(inf.get("debtToEquity"),100)
    fundamental=clamp(50+(growth-8)*.7+(margin-10)*.45+(roe-12)*.35-(max(pe-30,0))*.25-(max(debt-100,0))*.05)
    technical=clamp(.55*long+.45*short)
    market=market_score(pd.Timestamp(as_of).date().isoformat() if as_of is not None else None)
    total=clamp(.38*fundamental+.42*technical+.20*market)
    atr=float(pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1).rolling(14).mean().iloc[-1])
    lows=[]; highs=[]
    for w in (20,50,100): lows.append(float(d.Low.tail(w).min())); highs.append(float(d.High.tail(w).max()))
    supports=sorted(set(round(x,2) for x in lows), reverse=True); resist=sorted(set(round(x,2) for x in highs))
    pullback=(now-ma20.iloc[-1])/max(atr,1e-9); entry=clamp(72-abs(pullback)*12+(technical-50)*.35)
    lo=max(supports[0], float(ma20.iloc[-1]-atr*.5)); hi=min(now, float(ma20.iloc[-1]+atr*.3));
    if lo>hi: lo,hi=min(lo,hi),max(lo,hi)
    return dict(data=d,now=now,ma20=float(ma20.iloc[-1]),ma50=float(ma50.iloc[-1]),ma200=float(ma200.iloc[-1]),rsi=rsi,
      returns={"7일":ret(6),"1개월":ret(22),"3개월":ret(66),"6개월":ret(126),"1년":ret(252)}, long=long,short=short,fundamental=fundamental,technical=technical,market=market,total=total,entry=entry,atr=atr,supports=supports,resist=resist,entry_range=(lo,hi),inf=inf,
      long_reason=f"현재가/200일선 {(now/ma200.iloc[-1]-1)*100:+.1f}% · 50/200일선 {(ma50.iloc[-1]/ma200.iloc[-1]-1)*100:+.1f}% · 50일선 기울기 {slope50:+.1f}% · 200일선 기울기 {slope200:+.1f}%",
      short_reason=f"현재가/20일선 {(now/ma20.iloc[-1]-1)*100:+.1f}% · 20/50일선 {(ma20.iloc[-1]/ma50.iloc[-1]-1)*100:+.1f}% · RSI(14) {rsi:.1f}")

@st.cache_data(ttl=600, show_spinner=False)
def market_score(as_of=None):
    vals=[]
    for s in ("^GSPC","^NDX","^SOX","HYG"):
        try:
            c=prices(s,"6mo")["Close"].dropna()
            if as_of is not None: c=c.loc[pd.to_datetime(c.index).date<=pd.Timestamp(as_of).date()]
            vals.append(clamp(50+(c.iloc[-1]/c.rolling(50).mean().iloc[-1]-1)*250))
        except Exception: pass
    try:
        vc=prices("^VIX","1mo")["Close"].dropna()
        if as_of is not None: vc=vc.loc[pd.to_datetime(vc.index).date<=pd.Timestamp(as_of).date()]
        v=vc.iloc[-1]; vals.append(clamp(80-(v-12)*3))
    except Exception: pass
    return float(np.mean(vals)) if vals else 50

def gauge(label, value):
    color=score_color(value); label_text=grade(value)
    st.markdown(f"""<div class='score-card'>
      <div class='score-label'>{label}</div>
      <div class='score-value'>{value:.1f}<span class='score-denom'> / 100</span></div>
      <div class='score-track'><div class='score-fill' style='width:{clamp(value):.1f}%;background:{color}'></div></div>
      <div class='score-grade' style='color:{color}'><span class='score-dot' style='background:{color}'></span>{label_text}</div>
    </div>""",unsafe_allow_html=True)

def trend_sparkline(trend, key):
    if len(trend.values) < 2:
        return
    color = "#34d399" if trend.label == "Improving" else "#fb7185" if trend.label == "Weakening" else "#fbbf24"
    fill = "rgba(52,211,153,.10)" if trend.label == "Improving" else "rgba(251,113,133,.10)" if trend.label == "Weakening" else "rgba(251,191,36,.10)"
    # Categorical positions keep every trading-day observation equally spaced;
    # weekends and holidays should not create visual gaps in a score trend.
    x = list(range(len(trend.values)))
    ticktext = [pd.Timestamp(date).strftime("%m.%d") for date in trend.dates] if len(trend.dates) == len(trend.values) else [str(value + 1) for value in x]
    fig = go.Figure(go.Scatter(
        x=x, y=list(trend.values), mode="lines+markers+text",
        line=dict(color=color, width=3, shape="spline"),
        marker=dict(color=color, size=8, line=dict(color="#07111f", width=1.5)),
        text=[f"{value:.0f}" for value in trend.values], textposition="top center",
        textfont=dict(color="#cbd5e1", size=12), cliponaxis=False,
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{y:.1f}점<extra></extra>",
    ))
    low, high = min(trend.values), max(trend.values)
    padding = max((high - low) * .45, 2)
    fig.update_layout(
        height=155, margin=dict(l=18, r=18, t=28, b=28), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=True, fixedrange=True, showgrid=True, gridcolor="rgba(148,163,184,.22)",
                   tickfont=dict(color="#94a3b8", size=11), tickmode="array", tickvals=x,
                   ticktext=ticktext,
                   ticks="outside", ticklen=4,
                   range=[x[0]-.18, x[-1]+.18]),
        yaxis=dict(visible=False, fixedrange=True, range=[low-padding, high+padding]),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)

def render_entry_engine(snapshot, key):
    st.markdown(f"### ENTRY ENGINE V2 · {snapshot.score:.1f} / 100 · {grade(snapshot.score)}")
    labels={"Trend":"추세","Price Position":"가격 위치","Momentum":"모멘텀","Volume / OBV":"거래량 / OBV","Volatility":"변동성","Market":"시장환경"}
    rows=[]
    for name,value in snapshot.factors.items():
        icon="🟢" if value>=65 else "🟡" if value>=45 else "🔴"
        rows.append({"요소":labels[name] if name != "Volatility" else "변동성 안정성","점수":round(value,1),"상태":f"{icon} {grade(value)}","해석":snapshot.details[name]})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,
                 column_config={"점수":st.column_config.ProgressColumn("점수",min_value=0,max_value=100,format="%.1f")},key=key)
    st.info(snapshot.interpretation)

def briefing(title: str, body: str, kicker="AI BRIEF", wide=False, matched=False):
    cls="brief-card" + (" wide" if wide else "") + (" matched" if matched else "")
    st.markdown(f"<div class='{cls}'><div class='brief-kicker'>{kicker}</div><h3>{title}</h3><p>{body}</p></div>",unsafe_allow_html=True)

def load_history():
    try: return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception: return {}
def save_score(symbol, a):
    h=load_history(); arr=h.setdefault(symbol,[]); today=datetime.now(timezone.utc).date().isoformat()
    row={"date":today,"score":round(a["total"],1),"fundamental":round(a["fundamental"],1),"technical":round(a["technical"],1),"market":round(a["market"],1)}
    if arr and arr[-1].get("date")==today: arr[-1]=row
    else: arr.append(row)
    try: HISTORY_FILE.parent.mkdir(parents=True,exist_ok=True); HISTORY_FILE.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding="utf-8")
    except Exception: pass
    return h

def pulse_card(name,symbol):
    s,base=spark(symbol)
    if s.empty or not np.isfinite(base): st.markdown(f"<div class='card'><b>{name}</b><br>데이터 없음</div>",unsafe_allow_html=True); return
    change=(float(s.iloc[-1])/base-1)*100; y=(s/base-1)*100
    fig=go.Figure(); fig.add_hline(y=0,line_color="#94a3b8",line_width=1)
    fig.add_trace(go.Scatter(x=s.index,y=y,mode="lines",line=dict(color="#2563eb" if change>=0 else "#dc2626",width=2),fill="tozeroy",fillcolor="rgba(37,99,235,.08)" if change>=0 else "rgba(220,38,38,.08)"))
    fig.update_layout(height=80,margin=dict(l=0,r=0,t=2,b=0),showlegend=False,xaxis_visible=False,yaxis_visible=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.markdown(f"<b>{name}</b>　{money(float(s.iloc[-1]))}　<span style='color:{'#2563eb' if change>=0 else '#dc2626'}'>{pct(change)}</span>",unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

def render_market_dashboard():
    st.subheader("Market Dashboard")
    mh=market_score(); c1,c2,c3=st.columns(3)
    with c1: gauge("Market Health",mh)
    with c2: st.metric("Market Grade",grade(mh)); st.metric("AI Confidence",f"{min(92,round(60+abs(mh-50)*.65))}%")
    with c3:
        brief = "위험자산 흐름이 우호적입니다. 추세 확인 후 분할 접근이 유리합니다." if mh>=65 else "시장 방향성이 혼재합니다. 종목별 지지 확인과 비중 관리가 중요합니다." if mh>=45 else "시장 위험 선호가 약합니다. 현금 비중과 무효화 기준을 보수적으로 관리하세요."
        st.markdown("**AI Market Brief**"); st.info(brief)
    st.subheader("MARKET HEALTH · 최근 10영업일 차트")
    market_history_trend=HISTORY_STORE.recent("__MARKET__", "market")
    market_chart_trend=HISTORY_STORE.recent("__MARKET__", "market", count=10)
    st.markdown(format_trend(market_history_trend))
    trend_sparkline(market_chart_trend, "market_10d_sparkline")
    with st.expander("Market Pulse 12",expanded=False):
        st.caption("작은 선 그래프는 마지막 거래일의 15분 단위 장중 흐름입니다. Market Health 차트는 최근 10영업일, 변화량과 Trend 판단은 최근 5영업일 기준입니다.")
        items=list(PULSE.items())
        for i in range(0,12,4):
            cols=st.columns(4)
            for col,(n,s) in zip(cols,items[i:i+4]):
                with col: pulse_card(n,s)
    with st.expander("금리 · 신용시장 보조 패널"):
        st.caption("2년물은 미 재무부 공식 자료, 5·10·30년물은 Yahoo를 우선 사용합니다. 연결 실패 시 미 재무부·FRED 자료로 자동 전환합니다.")
        cols=st.columns(4); last={}; sources={}
        try:
            official=treasury_yields()
        except Exception:
            official=pd.DataFrame()
        try:
            fred=fred_yields() if official.empty else pd.DataFrame()
        except Exception:
            fred=pd.DataFrame()
        yahoo_rates={"US 5Y":"^FVX","US 10Y":"^TNX","US 30Y":"^TYX"}
        for name,series_id,col in zip(("US 2Y","US 5Y","US 10Y","US 30Y"),("DGS2","DGS5","DGS10","DGS30"),cols):
            try:
                d=pd.Series(dtype=float)
                if name in yahoo_rates:
                    try:
                        d=prices(yahoo_rates[name],"1mo")["Close"].dropna(); sources[name]="Yahoo"
                    except Exception: pass
                if d.empty and not official.empty and series_id in official:
                    d=official[series_id].dropna(); sources[name]="미 재무부"
                if d.empty and not fred.empty and series_id in fred:
                    d=fred[series_id].dropna(); sources[name]="FRED"
                if d.empty: raise ValueError("rate series is empty")
                v=float(d.iloc[-1]); change=v-float(d.iloc[-2]); last[name]=v
                col.metric(name,f"{v:.2f}%",f"{change:+.2f}%p"); col.caption(f"{RATE_GUIDE[name]} · {sources[name]}")
            except Exception: col.metric(name,"N/A"); col.caption(RATE_GUIDE[name])
        missing=[name for name in ("US 2Y","US 5Y","US 10Y","US 30Y") if name not in last]
        if missing: st.caption(f"※ 현재 연결되지 않은 금리: {', '.join(missing)}. 나머지 신용시장 자료는 정상적으로 계속 표시합니다.")
        cols=st.columns(4)
        for name,symbol_name,col in (("HYG","HYG",cols[0]),("LQD","LQD",cols[1])):
            try:
                d=prices(symbol_name,"1mo")["Close"].dropna(); v=float(d.iloc[-1]); ch=(v/d.iloc[-2]-1)*100
                col.metric(name,money(v),pct(ch)); col.caption(RATE_GUIDE[name])
            except Exception: col.metric(name,"N/A"); col.caption(RATE_GUIDE[name])
        spread=last.get('US 10Y',np.nan)-last.get('US 2Y',np.nan)
        cols[2].metric("10Y-2Y",f"{spread:+.2f}%p" if np.isfinite(spread) else "N/A"); cols[2].caption(RATE_GUIDE["10Y-2Y"])
        try:
            hyg=prices("HYG","6mo")["Close"].squeeze(); lqd=prices("LQD","6mo")["Close"].squeeze()
            cols[3].metric("Credit Spread proxy",f"{(hyg.iloc[-1]/hyg.iloc[-20]-lqd.iloc[-1]/lqd.iloc[-20])*100:+.2f}%")
        except Exception: cols[3].metric("Credit Spread proxy","N/A")
        cols[3].caption(RATE_GUIDE["Credit Spread proxy"])

st.title("Stock Analyzer by Kijungnam")
st.caption("V5-a0.1 · MULTI-LENS DECISION SYSTEM")
try:
    from top10_ranking import render_top10
    render_top10(market_score())
except Exception as exc:
    st.subheader("오늘의 퀀트 TOP 10")
    st.info("현재 순위 데이터를 불러오지 못했습니다. 기존 종목 분석은 정상적으로 사용할 수 있습니다.")
st.subheader("종목 검색")
q=st.text_input("티커 또는 회사명",placeholder="예: NVDA, NVIDIA, 삼성전자, 005930")
results=search_yahoo(q) if q else []
if results:
    labels=[f"{x['name']} · {x['symbol']} · {x['exchange']} · {x['type']}" for x in results]
    selected=st.selectbox("Yahoo 검색 후보",labels); symbol=results[labels.index(selected)]["symbol"]
else: symbol=""

if symbol and st.button("분석 시작",type="primary",use_container_width=True): st.session_state["symbol"]=symbol
symbol=st.session_state.get("symbol","")
if symbol:
    selected_name=info(symbol).get("longName",symbol)
    st.caption(f"현재 분석 종목 · {selected_name} · {symbol}")

analysis = None
quant_view = None
option_view = None
entry_view = None
analysis_history = HISTORY_STORE.load()
if symbol:
    try:
        from advanced_analyzer import quant_snapshot
        from options_analyzer import bias_style, get_option_snapshot, option_bias
        with st.spinner(f"{symbol}의 분석 관점을 통합하는 중입니다..."):
            analysis = calc(symbol)
            quant_view = quant_snapshot(analysis)
            entry_view = build_entry_snapshot(analysis, quant_view)
            try:
                option_view, _, _, _ = get_option_snapshot(symbol, analysis["now"])
            except Exception:
                option_view = None
            # 5D 판단은 유지하면서 차트에는 10영업일을 보여줄 수 있도록 누락 데이터를 역산합니다.
            valid_trading_dates = {pd.Timestamp(x).date().isoformat() for x in analysis["data"].index}
            HISTORY_STORE.retain_valid_dates(symbol, valid_trading_dates)
            try:
                market_valid_dates = {pd.Timestamp(x).date().isoformat() for x in prices("^GSPC", "6mo").index}
                HISTORY_STORE.retain_valid_dates("__MARKET__", market_valid_dates)
            except Exception:
                pass
            existing_dates = HISTORY_STORE.dates(symbol)
            recent_dates = [pd.Timestamp(x).date() for x in analysis["data"].index[-10:]]
            for historical_date in recent_dates[:-1]:
                if historical_date.isoformat() in existing_dates:
                    continue
                try:
                    past = calc(symbol, historical_date, analysis["inf"])
                    past_quant = quant_snapshot(past)
                    HISTORY_STORE.record(symbol, {"overall": past["total"], "quant": past_quant["score"], "market": past["market"]}, historical_date, {"source": "reconstructed"})
                    HISTORY_STORE.record("__MARKET__", {"market": past["market"]}, historical_date, {"source": "reconstructed"})
                except Exception:
                    pass
            data_date = pd.Timestamp(analysis["data"].index[-1]).date()
            analysis_history = HISTORY_STORE.record(symbol, {"overall": analysis["total"], "quant": quant_view["score"], "market": analysis["market"]}, data_date, {"source": "recorded"})
            HISTORY_STORE.record("__MARKET__", {"market": analysis["market"]}, data_date, {"source": "recorded"})
        overall_trend = HISTORY_STORE.recent(symbol, "overall")
        quant_trend = HISTORY_STORE.recent(symbol, "quant")
        market_trend = HISTORY_STORE.recent("__MARKET__", "market")
        reconstructed = any(row.get("source") == "reconstructed" for row in HISTORY_STORE.recent_rows(symbol))
        history_quality = .82 if reconstructed else 1.0
        option_label = option_bias(option_view) if option_view else "N/A"
        lenses = {
            "overall": Lens("종합", analysis["total"], change=overall_trend.change, data_quality=history_quality),
            "quant": Lens("퀀트", quant_view["score"], change=quant_trend.change, data_quality=history_quality),
            "options": Lens("옵션", label=option_label, available=option_view is not None, data_quality=option_view.data_quality if option_view else 0),
            "market": Lens("시장", analysis["market"], change=market_trend.change, data_quality=history_quality),
        }
        consensus = build_consensus(lenses)
        confidence_note = confidence_interpretation(consensus.confidence)
        def lens_tone(lens):
            if not lens.available: return ("#94a3b8", "N/A")
            if lens.name == "옵션":
                _, color = bias_style(option_label); return (color, option_label)
            color = "#34d399" if lens.direction > 0 else "#fb7185" if lens.direction < 0 else "#fbbf24"
            return (color, grade(lens.score))
        def lens_card(label, value, lens, trend=None):
            delta = "" if trend is None or trend.change is None else f" · {'▲' if trend.change >= 0 else '▼'}{abs(trend.change):.1f} / 5D"
            color, _ = lens_tone(lens)
            return f"<div class='lens'><small>{label}</small><b><i class='lens-dot' style='color:{color};background:{color}'></i>{value}{delta}</b></div>"
        def lens_line(lens, value):
            color, status = lens_tone(lens)
            return f"<div class='consensus-line'><span>{lens.name}</span><i class='lens-dot' style='color:{color};background:{color}'></i><b>{value if value else status}</b></div>"
        st.markdown("<div class='consensus'><div class='brief-kicker'>ANALYSIS CONSENSUS</div><div class='consensus-grid'>"+
            lens_card("종합", f"{analysis['total']:.0f} {grade(analysis['total'])}", lenses['overall'], overall_trend)+
            lens_card("퀀트", f"{quant_view['score']:.0f} {grade(quant_view['score'])}", lenses['quant'], quant_trend)+
            lens_card("옵션", option_label, lenses['options'])+lens_card("시장", f"{analysis['market']:.0f} {grade(analysis['market'])}", lenses['market'], market_trend)+
            "</div><div class='consensus-lines'>"+lens_line(lenses['overall'], grade(analysis['total']))+lens_line(lenses['quant'], grade(quant_view['score']))+lens_line(lenses['options'], option_label)+lens_line(lenses['market'], grade(analysis['market']))+"</div>"+
            f"<div class='consensus-summary'><b>{consensus.headline}</b><br>Consensus: <b>{consensus.pattern}</b> · Confidence: <b>{consensus.confidence}%</b><br><span style='color:#94a3b8'>Confidence 해석: {confidence_note}</span><br><br><b>해석</b><br>{consensus.interpretation}</div></div>", unsafe_allow_html=True)
        st.caption("Confidence는 상승 확률이 아니라 분석 방향의 일관성과 데이터 품질에 대한 신뢰도입니다.")
        if reconstructed:
            st.caption("최근 5D 중 `가격 기반 역산` 값은 현재 펀더멘털을 고정하고 각 거래일의 가격·거래량·시장환경을 재계산한 참고값입니다. 이후 실제 저장값으로 순차 교체됩니다.")
    except Exception as exc:
        st.warning(f"Analysis Consensus를 계산하지 못했습니다. 개별 분석 메뉴는 계속 사용할 수 있습니다: {exc}")

mode=st.radio("분석 메뉴",["📊 종합분석","🎯 퀀트분석","🧩 옵션분석","🌎 시장환경"],horizontal=True,label_visibility="collapsed")
st.divider()

if mode=="🌎 시장환경":
    render_market_dashboard()

if symbol and mode=="🧩 옵션분석":
    try:
        from options_analyzer import render_options
        with st.spinner(f"{symbol}의 옵션 시장 데이터를 분석하는 중입니다..."):
            d=prices(symbol,"5d"); spot=float(d["Close"].dropna().iloc[-1])
        base = analysis or calc(symbol)
        render_options(symbol,spot,money,base["supports"][0],base["resist"][0])
    except Exception:
        st.warning("옵션분석을 현재 표시할 수 없습니다. 데이터 제공 상태를 확인한 뒤 다시 시도해 주세요. 기존 분석 탭은 정상적으로 사용할 수 있습니다.")
elif mode=="🧩 옵션분석":
    st.info("먼저 위 검색창에서 분석할 종목을 선택해 주세요.")

if symbol and mode=="🎯 퀀트분석":
    try:
        from advanced_analyzer import render_advanced
        with st.spinner(f"{symbol}의 퀀트 데이터를 분석하는 중입니다..."): advanced_base=analysis or calc(symbol)
        if quant_view:
            st.subheader("QUANT SCORE · 최근 10영업일 차트")
            quant_history_trend=HISTORY_STORE.recent(symbol, "quant")
            quant_chart_trend=HISTORY_STORE.recent(symbol, "quant", count=10)
            st.markdown(format_trend(quant_history_trend))
            trend_sparkline(quant_chart_trend, f"quant_10d_sparkline_{symbol}")
            st.caption(f"현재 {quant_view['score']:.1f} {grade(quant_view['score'])} · 추세 {quant_view['trend']:.1f} · 모멘텀 {quant_view['momentum']:.1f} · 수급 {quant_view['supply']:.1f} · 기업품질 {quant_view['quality']:.1f}")
        if entry_view: render_entry_engine(entry_view, f"quant_entry_v2_{symbol}")
        render_advanced(symbol,advanced_base,prices,news,money,pct,clamp,grade,score_color,entry_view)
    except Exception as e: st.error(f"퀀트분석을 표시할 수 없습니다: {e}")
elif mode=="🎯 퀀트분석":
    st.info("먼저 위 검색창에서 분석할 종목을 선택해 주세요.")

if symbol and mode=="📊 종합분석":
  try:
    with st.spinner(f"{symbol} 데이터를 분석하는 중입니다..."): a=analysis or calc(symbol); hist=analysis_history
    name=a["inf"].get("longName",symbol); st.divider(); st.header(f"{name} ({symbol})"); st.caption(f"현재가 {money(a['now'])} · 데이터 기준 {a['data'].index[-1].date()}")
    cols=st.columns(4)
    for col,(label,key) in zip(cols,[("종합점수","total"),("펀더멘털","fundamental"),("테크니컬","technical"),("시장환경","market")]):
        with col:gauge(label,a[key])
    st.subheader("종합점수 · 최근 10영업일 차트")
    overall_history_trend=HISTORY_STORE.recent(symbol, "overall")
    overall_chart_trend=HISTORY_STORE.recent(symbol, "overall", count=10)
    st.markdown(format_trend(overall_history_trend))
    trend_sparkline(overall_chart_trend, f"overall_10d_sparkline_{symbol}")
    if any(row.get("source") == "reconstructed" for row in HISTORY_STORE.recent_rows(symbol)):
        st.caption("가격 기반 역산 포함 · 현재 펀더멘털 고정, 과거 기술·시장 데이터 재계산")
    positives=[]; cautions=[]
    (positives if a['long']>=65 else cautions).append(f"장기 추세 {grade(a['long'])} ({a['long']:.0f})")
    (positives if a['short']>=65 else cautions).append(f"단기 추세 {grade(a['short'])} ({a['short']:.0f})")
    (positives if a['fundamental']>=65 else cautions).append(f"펀더멘털 {grade(a['fundamental'])}")
    x,y=st.columns(2)
    with x: st.success("**긍정 요인**\n\n"+"\n\n".join(f"• {z}" for z in positives) if positives else "뚜렷한 우위 요인을 추가 확인하세요.")
    with y: st.warning("**주의 요인**\n\n"+"\n\n".join(f"• {z}" for z in cautions) if cautions else "현재 모델상 두드러진 약점이 적습니다.")
    st.info(f"종합점수는 펀더멘털 38%, 테크니컬 42%, 시장환경 20%를 반영합니다. 현재 {a['total']:.1f}점({grade(a['total'])})입니다.")

    st.subheader("AI 종합 브리핑")
    total_view = "우호적 흐름이지만 추천 가격대의 지지를 확인하며 분할 접근하는 편이 유리합니다." if a['total']>=65 else "방향성이 혼재하므로 추천 가격대와 무효화선을 함께 확인하는 전략이 적절합니다." if a['total']>=45 else "방어적 접근이 필요한 구간입니다. 추세 회복과 지지 확인 전까지 비중을 보수적으로 관리하세요."
    briefing("종합 AI 브리핑",f"종합점수는 {a['total']:.0f}점({grade(a['total'])})입니다. {total_view} 공개 데이터의 시차와 장중 변동성을 함께 고려해야 합니다.","DECISION BRIEF",True)
    b1,b2=st.columns(2)
    with b1:
        f_view="성장성과 수익성 지표가 상대적으로 우호적입니다." if a['fundamental']>=65 else "공개 재무자료만으로 강한 우위를 단정하기 어려워 다음 실적과 현금흐름 확인이 중요합니다."
        briefing("펀더멘털 브리핑",f"펀더멘털 점수는 {a['fundamental']:.0f}점입니다. {f_view}","FUNDAMENTAL")
    with b2:
        t_view="이동평균 구조가 가격에 우호적입니다." if a['technical']>=65 else "단기와 장기 신호가 엇갈리므로 돌파 또는 지지 확인이 필요합니다."
        briefing("테크니컬 브리핑",f"테크니컬 점수는 {a['technical']:.0f}점입니다. 현재가 {money(a['now'])}, 20일선 {money(a['ma20'])}, 50일선 {money(a['ma50'])}, 200일선 {money(a['ma200'])}, RSI {a['rsi']:.1f}를 반영했습니다. {t_view}","TECHNICAL")
    b3,b4=st.columns(2)
    with b3:
        m_view="시장 신호가 종목 선택을 뒷받침합니다." if a['market']>=65 else "시장 신호가 혼재해 종목 자체의 지지 확인이 더 중요합니다." if a['market']>=45 else "시장 위험 선호가 약해 개별 종목 신호보다 비중 관리가 우선입니다."
        briefing("시장환경 브리핑",f"시장환경 점수는 {a['market']:.0f}점입니다. S&P 500·Nasdaq 100·SOX 방향, VIX, 위험자산과 신용시장을 종합했습니다. {m_view}","MARKET",matched=True)
    with b4:
        briefing("추세·진입 브리핑",f"장기 추세 {a['long']:.0f}점, 단기 추세 {a['short']:.0f}점, 진입 적합도 {a['entry']:.0f}점입니다. 추천 진입 참고가격대와 가까운 지지에서 확인되는 반응을 보고 분할 접근하며 손절·무효화선을 지키는 것이 핵심입니다.","TREND & ENTRY",matched=True)

    st.subheader("장기 · 단기 추세 분석")
    st.write("50일선·200일선의 이격과 기울기, 6개월·12개월 수익률을 연속형으로 반영합니다.")
    c1,c2=st.columns(2)
    with c1: gauge("장기 추세",a["long"]); st.caption(a["long_reason"])
    with c2: gauge("단기 추세",a["short"]); st.caption(a["short_reason"])
    st.markdown("<p class='blue'>점수가 50에 가까우면 방향이 불분명하며, 65 이상은 상승 우위, 35 이하는 하락 우위로 해석합니다.</p>",unsafe_allow_html=True)

    st.subheader("진입 적합도 · AI 눌림목 전략")
    if entry_view: render_entry_engine(entry_view, f"overall_entry_v2_{symbol}")
    c1,c2,c3=st.columns(3)
    entry_score=entry_view.score if entry_view else a["entry"]
    with c1:gauge("진입 적합도",entry_score)
    lo,hi=a["entry_range"]; stop=max(a["supports"][-1],lo-a["atr"]*1.3); target=max(a["resist"][-1],a["now"]+a["atr"]*2)
    c2.metric("추천 진입가격대",f"{money(lo)} ~ {money(hi)}"); c2.metric("현재 위치",f"추천구간 대비 {(a['now']/((lo+hi)/2)-1)*100:+.1f}%")
    weight=30 if entry_score>=65 else 20 if entry_score>=45 else 10
    c3.metric("추천비중",f"최대 {weight}%"); c3.caption("AI 눌림목: 20일선·ATR·가까운 지지를 결합해 과도한 추격 여부를 평가합니다.")
    plans=pd.DataFrame({"단계":["1차","2차","3차"],"가격":[hi,(lo+hi)/2,lo],"해당 계획 내 비중":["30%","30%","40%"]})
    st.dataframe(plans.style.format({"가격":"{:,.2f}"}),hide_index=True,use_container_width=True)
    st.write(f"손절/무효화 참고선 **{money(stop)}** · 목표 참고선 **{money(target)}**")
    st.caption(f"진입 적합도 근거: 추천구간과 현재가 거리, ATR 변동성, 테크니컬 점수를 결합했습니다. 현재 RSI {a['rsi']:.1f}, ATR {a['atr']:.2f}.")

    st.subheader("대응 시나리오")
    c1,c2,c3=st.columns(3)
    first_target=max(a['resist'][0],a['now']+a['atr']); second_target=target
    with c1: st.markdown(f"<div class='scenario up'><h4>🟢 Bull · 돌파</h4><p><b>조건</b> {money(a['resist'][0])} 돌파 + 거래량 증가<br><b>대응</b> 돌파 지지 확인 후 접근<br><b>목표</b> {money(first_target)} / {money(second_target)}</p></div>",unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='scenario mid'><h4>🟡 Base · 박스/지지</h4><p><b>조건</b> {money(lo)}~{money(hi)} 범위 유지<br><b>대응</b> {money((lo+hi)/2)} 이하 지지 시 분할 접근<br><b>주의</b> 박스 중앙 추격 자제</p></div>",unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='scenario down'><h4>🔴 Bear · 무효화</h4><p><b>조건</b> {money(stop)} 종가 이탈<br><b>대응</b> 추가 진입 중단<br><b>재평가</b> 다음 지지 {money(a['supports'][-1])}</p></div>",unsafe_allow_html=True)

    st.subheader("가격 차트 · 지지와 저항")
    d=a['data'].tail(252); fig=go.Figure(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name="가격"))
    for v in a['supports']: fig.add_hline(y=v,line_dash="dot",line_color="#2563eb",annotation_text=f"지지 {v}")
    for v in a['resist']: fig.add_hline(y=v,line_dash="dot",line_color="#dc2626",annotation_text=f"저항 {v}")
    fig.update_layout(height=520,xaxis_rangeslider_visible=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0a1728",font=dict(color="#cbd5e1"),xaxis=dict(gridcolor="#20344d"),yaxis=dict(gridcolor="#20344d")); st.plotly_chart(fig,use_container_width=True)
    zones=[]
    for i,v in enumerate(a['supports']): zones.append(["지지",v,"강" if i==0 else "중","최근 20/50/100일 저점"])
    for i,v in enumerate(a['resist']): zones.append(["저항",v,"강" if i==0 else "중","최근 20/50/100일 고점"])
    st.dataframe(pd.DataFrame(zones,columns=["구분","가격","강도","산출근거"]),hide_index=True,use_container_width=True)

    st.subheader("투자기간별 참고점수 · 수익률")
    horizon=pd.DataFrame({"기간":["단기(1개월)","스윙(3개월)","중기(6개월)","장기(1년)"],"참고점수":[round(a['short']),round(.55*a['short']+.45*a['long']),round(.35*a['short']+.65*a['long']),round(a['long'])]})
    st.dataframe(horizon,hide_index=True,use_container_width=True); st.write("　".join(f"**{k}** {pct(v)}" for k,v in a['returns'].items() if k in ("7일","1개월","3개월","1년")))

    st.subheader("AI Score History")
    rows=hist.get(symbol,[]); hf=pd.DataFrame(rows)
    if len(hf)<=1: st.info("첫 분석입니다. 같은 종목을 다른 영업일에 다시 분석하면 종합·퀀트·시장 점수 모멘텀을 비교합니다.")
    if not hf.empty:
        available=[x for x in ("overall","quant","market") if x in hf]
        st.line_chart(hf.set_index("date")[available].tail(20))
    st.download_button("점수 이력 JSON 내보내기",json.dumps(hist,ensure_ascii=False,indent=2),"analyzer_score_history.json","application/json")
    uploaded=st.file_uploader("이전 이력 JSON 가져오기",type="json")
    if uploaded and st.button("가져온 이력 적용"):
        try: HISTORY_FILE.parent.mkdir(parents=True,exist_ok=True); HISTORY_FILE.write_bytes(uploaded.getvalue()); st.success("이력을 적용했습니다. 페이지를 새로고침하세요.")
        except Exception as e: st.error(f"이력 적용 실패: {e}")
    st.caption("Streamlit Community Cloud의 로컬 파일은 재배포·재시작 시 초기화될 수 있습니다. 장기 보관은 JSON을 내려받아 보관하고 다음 배포에서 다시 가져오세요. 외부 DB 연결 시 ANALYZER_HISTORY_FILE 대신 영구 저장소 어댑터를 사용할 수 있습니다.")

    st.subheader("최근 뉴스 요약")
    ns=news(symbol)
    if not ns: st.info("현재 불러온 뉴스가 없습니다.")
    for title,summary,url in ns:
        st.markdown(f"**[{title}]({url})**" if url else f"**{title}**"); st.caption(summary[:300] if summary else "제목 기반 참고 뉴스")
    st.warning("본 결과는 공개 시세 기반의 정량 참고자료이며 투자 권유가 아닙니다. 지연·누락 데이터와 모델 오차가 있을 수 있습니다.")
  except Exception as e: st.error(f"분석할 수 없습니다: {e}")
