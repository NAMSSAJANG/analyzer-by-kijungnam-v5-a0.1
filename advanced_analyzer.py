from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


def _safe(v, default=np.nan):
    try:
        x=float(v)
        return x if np.isfinite(x) else default
    except Exception: return default


def _score(v, low, high):
    if not np.isfinite(v): return np.nan
    return float(np.clip((v-low)/(high-low)*100,0,100))


def _fmt(v, suffix="", digits=1):
    return "—" if not np.isfinite(v) else f"{v:,.{digits}f}{suffix}"


def _weighted(parts):
    valid=[(v,w) for v,w in parts if np.isfinite(v)]
    return float(sum(v*w for v,w in valid)/sum(w for _,w in valid)) if valid else 50.0


def _rsi(c, n=14):
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); down=-delta.clip(upper=0).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/down.replace(0,np.nan))


def _adx(d, n=14):
    up=d.High.diff(); dn=-d.Low.diff(); plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean(); p=100*pd.Series(plus,index=d.index).ewm(alpha=1/n,adjust=False).mean()/atr; m=100*pd.Series(minus,index=d.index).ewm(alpha=1/n,adjust=False).mean()/atr
    return (100*(p-m).abs()/(p+m).replace(0,np.nan)).ewm(alpha=1/n,adjust=False).mean(),atr


def _metrics(a):
    d=a["data"].copy(); c=d.Close.astype(float); v=d.Volume.astype(float)
    ema20=c.ewm(span=20,adjust=False).mean(); ema50=c.ewm(span=50,adjust=False).mean(); ema200=c.ewm(span=200,adjust=False).mean()
    mid=c.rolling(20).mean(); std=c.rolling(20).std(); upper=mid+2*std; lower=mid-2*std
    rs=_rsi(c); adx,atr=_adx(d); typical=(d.High+d.Low+d.Close)/3; vwap=(typical*v).cumsum()/v.cumsum().replace(0,np.nan)
    obv=(np.sign(c.diff()).fillna(0)*v).cumsum(); macd=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); signal=macd.ewm(span=9,adjust=False).mean()
    ret=lambda n:(c.iloc[-1]/c.iloc[-min(n,len(c))]-1)*100
    high52=c.tail(252).max(); low52=c.tail(252).min(); pos=(c.iloc[-1]-low52)/max(high52-low52,1e-9)*100
    vol_ratio=v.iloc[-1]/max(v.tail(20).mean(),1); atr_pct=atr.iloc[-1]/c.iloc[-1]*100; z=(c.iloc[-1]-c.tail(60).mean())/max(c.tail(60).std(),1e-9)
    trend=_weighted([(_score(c.iloc[-1]/ema200.iloc[-1]-1,-.15,.25),.35),(_score(ema20.iloc[-1]/ema50.iloc[-1]-1,-.08,.10),.3),(_score(ret(252),-30,60),.35)])
    momentum=_weighted([(_score(rs.iloc[-1],35,75),.3),(_score(ret(63),-20,30),.35),(_score(macd.iloc[-1]-signal.iloc[-1],-c.iloc[-1]*.015,c.iloc[-1]*.015),.35)])
    volatility=float(np.clip(100-(atr_pct-1)*18,0,100)); supply=_weighted([(_score(vol_ratio,.5,2),.45),(_score(obv.iloc[-1]/max(abs(obv.tail(60)).max(),1),-.5,1),.35),(_score(c.iloc[-1]/vwap.iloc[-1]-1,-.08,.12),.2)])
    overheat=max(0,(rs.iloc[-1]-68)*2.4)+max(0,z-1.5)*12
    timing=float(np.clip(_weighted([(trend,.34),(momentum,.22),(volatility,.14),(supply,.2),(a["market"],.1)])-overheat,0,100))
    return dict(d=d,c=c,v=v,ema20=ema20,ema50=ema50,ema200=ema200,upper=upper,lower=lower,obv=obv,rsi=float(rs.iloc[-1]),adx=float(adx.iloc[-1]),atr_pct=float(atr_pct),vwap=float(vwap.iloc[-1]),vol_ratio=float(vol_ratio),macd=float(macd.iloc[-1]),signal=float(signal.iloc[-1]),ret3=float(ret(63)),ret12=float(ret(252)),high52=float(high52),low52=float(low52),position=float(pos),z=float(z),trend=trend,momentum=momentum,volatility=volatility,supply=supply,timing=timing)


def quant_snapshot(a):
    m = _metrics(a)
    inf = a["inf"]
    quality = _weighted([(a["fundamental"], .65), (_score(_safe(inf.get("revenueGrowth")) * 100, -5, 25), .2), (_score(_safe(inf.get("returnOnEquity")) * 100, 0, 30), .15)])
    score = _weighted([(quality, .55), (m["timing"], .45)])
    return {"score": score, "quality": quality, "timing": m["timing"], "trend": m["trend"], "momentum": m["momentum"], "supply": m["supply"], "volatility": m["volatility"]}


def _bar_card(label,value,description,color_fn,tag="FACTOR"):
    color=color_fn(value)
    st.markdown(f"""<div class='v4-factor'><div><span class='v4-tag'>{tag}</span><b>{label}</b></div>
    <div class='v4-num' style='color:{color}'>{value:.1f}</div><div class='v4-track'><div style='width:{value:.1f}%;background:{color}'></div></div>
    <p>{description}</p></div>""",unsafe_allow_html=True)


def _status_row(label,value,guide,tone="neutral"):
    st.markdown(f"<div class='v4-row {tone}'><div><b>{label}</b><small>{guide}</small></div><strong>{value}</strong></div>",unsafe_allow_html=True)


@st.cache_data(ttl=3600,show_spinner=False)
def _peer_info(symbols):
    out=[]
    for sym in symbols[:6]:
        try:
            x=yf.Ticker(sym).get_info(); h=yf.download(sym,period="1y",auto_adjust=True,progress=False,threads=False)
            if isinstance(h.columns,pd.MultiIndex): h.columns=h.columns.get_level_values(0)
            c=h.Close.dropna(); ret=(c.iloc[-1]/c.iloc[0]-1)*100 if len(c)>1 else np.nan
            out.append({"종목":x.get("shortName",sym),"티커":sym,"시가총액":_safe(x.get("marketCap"))/1e9,"PER":_safe(x.get("trailingPE")),"PBR":_safe(x.get("priceToBook")),"ROE":_safe(x.get("returnOnEquity"))*100,"영업이익률":_safe(x.get("operatingMargins"))*100,"12M":ret})
        except Exception: pass
    return pd.DataFrame(out)


def _peer_defaults(symbol,inf):
    sector=(inf.get("sector") or "").lower()
    pools={"technology":["AAPL","MSFT","NVDA","AVGO","ORCL"],"financial":["JPM","BAC","WFC","GS","MS"],"healthcare":["LLY","JNJ","UNH","ABBV","MRK"],"consumer cyclical":["AMZN","TSLA","HD","MCD","NKE"],"communication":["META","GOOGL","NFLX","TMUS","DIS"],"energy":["XOM","CVX","COP","SLB","EOG"]}
    if symbol.endswith((".KS",".KQ")): return [symbol,"005930.KS","000660.KS","035420.KS","051910.KS"]
    pool=next((v for k,v in pools.items() if k in sector),[symbol,"SPY","QQQ"])
    return list(dict.fromkeys([symbol]+pool))[:5]


@st.cache_data(ttl=3600,show_spinner=False)
def _calendar_data(symbol):
    try:
        cal=yf.Ticker(symbol).calendar
        days=None
        if isinstance(cal,dict):
            raw=cal.get("Earnings Date") or cal.get("EarningsDate")
            if isinstance(raw,(list,tuple)) and raw: raw=raw[0]
            if raw is not None:
                dt=pd.Timestamp(raw)
                if dt.tzinfo is not None: dt=dt.tz_localize(None)
                days=(dt.normalize()-pd.Timestamp.now().normalize()).days
        return cal,days
    except Exception: return None,None


def _entry_label(score):
    if score>=80: return "🟢 진입 우위",0
    if score>=65: return "🟢 진입 검토",1
    if score>=45: return "🟡 관망·조건 확인",2
    if score>=30: return "🟠 진입 보류",3
    return "🔴 진입 회피",4


def _entry_decision(symbol,a,m,entry_score=None):
    base,level=_entry_label(m["timing"] if entry_score is None else entry_score); risks=[]; releases=[]; cal,earnings_days=_calendar_data(symbol)
    if earnings_days is not None and 0<=earnings_days<=7:
        risks.append(f"실적 발표가 D-{earnings_days}로 임박했습니다")
        releases.append("실적 발표 후 가격 방향과 거래량 확인")
    if m["rsi"]>=72:
        risks.append(f"RSI {m['rsi']:.1f}로 단기 과열 구간입니다")
        releases.append("RSI 과열 완화 또는 20일선 부근 눌림 확인")
    hi=a["entry_range"][1]; distance=(a["now"]/hi-1)*100
    if distance>5:
        risks.append(f"현재가가 추천 진입구간 상단보다 {distance:.1f}% 높습니다")
        releases.append("추천 진입가격대 재진입 또는 새 지지 형성")
    if m["vol_ratio"]<.8:
        risks.append(f"거래량이 20일 평균의 {m['vol_ratio']:.2f}배로 확인 강도가 낮습니다")
        releases.append("평균 이상의 거래량을 동반한 반등·돌파 확인")
    if a["market"]<45:
        risks.append(f"시장환경이 {a['market']:.1f}점으로 Weak 구간입니다")
        releases.append("시장환경 45점 이상 회복")
    stop=max(a["supports"][-1],a["entry_range"][0]-a["atr"]*1.3); stop_distance=(a["now"]/stop-1)*100
    if stop_distance>12:
        risks.append(f"무효화선까지 거리가 {stop_distance:.1f}%로 손실 허용폭이 큽니다")
        releases.append("가까운 지지 형성으로 손절 거리 축소")
    final_level=min(4,level+1) if risks else level
    labels=["🟢 진입 우위","🟢 진입 검토","🟡 관망·조건 확인","🟠 진입 보류","🔴 진입 회피"]
    final=labels[final_level]
    if earnings_days is not None and 0<=earnings_days<=3: final="🟠 실적 확인 전 보류"
    return dict(base=base,final=final,risks=risks,releases=list(dict.fromkeys(releases)),calendar=cal)


def render_advanced(symbol,a,prices_fn,news_fn,money,pct,clamp,grade,color_fn,entry_snapshot=None):
    st.markdown("""<style>
    .v4-hero{border:1px solid #29415e;border-radius:16px;padding:20px;background:#0d1b2d;margin-bottom:14px}.v4-hero h2{margin:.2rem 0 .6rem}.v4-hero p{color:#cbd5e1;line-height:1.75;margin:0}
    .v4-factor{border:1px solid #29415e;border-radius:14px;padding:16px;background:#0d1b2d;min-height:180px;margin-bottom:12px}.v4-tag{display:inline-block;color:#7dd3fc;background:#123252;border-radius:7px;padding:4px 7px;margin-right:8px;font-size:.7rem}.v4-num{font-size:2rem;font-weight:850;text-align:center;margin:12px}.v4-track{height:7px;background:#223149;border-radius:99px;overflow:hidden}.v4-track div{height:100%;border-radius:99px}.v4-factor p{color:#aebdd0;font-size:.86rem;line-height:1.55}
    .v4-row{border:1px solid #29415e;border-left:3px solid #64748b;border-radius:11px;padding:13px 15px;margin:8px 0;background:#0d1b2d;display:flex;justify-content:space-between;gap:18px;align-items:center}.v4-row small{display:block;color:#8292a8;margin-top:4px}.v4-row strong{text-align:right;white-space:nowrap}.v4-row.good{border-left-color:#10b981}.v4-row.good strong{color:#34d399}.v4-row.bad{border-left-color:#ef4444}.v4-row.bad strong{color:#fb7185}
    </style>""",unsafe_allow_html=True)
    m=_metrics(a); inf=a["inf"]; name=inf.get("longName",symbol); entry_score=entry_snapshot.score if entry_snapshot else m["timing"]; decision=_entry_decision(symbol,a,m,entry_score)
    quality=_weighted([(a["fundamental"],.65),(_score(_safe(inf.get("revenueGrowth"))*100,-5,25),.2),(_score(_safe(inf.get("returnOnEquity"))*100,0,30),.15)])
    verdict=entry_snapshot.status if entry_snapshot else decision["final"]
    st.header(f"퀀트분석 · {name} ({symbol})")
    trend_strength=entry_snapshot.trend_strength if entry_snapshot else m["trend"]
    st.markdown(f"<div class='v4-hero'><div class='brief-kicker'>ENTRY ENGINE V2 DECISION</div><h2>{verdict} · {entry_score:.1f}점</h2><p>Trend Strength {trend_strength:.1f}점과 Entry Timing {entry_score:.1f}점을 분리했습니다. 기업 종합점수 {quality:.1f}점은 기존 산식을 유지합니다. RSI {m['rsi']:.1f}는 단독 보류 조건이 아닙니다.</p></div>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.subheader("기업 종합점수"); _bar_card("기업 품질",quality,"성장성·수익성·재무안정성과 밸류에이션을 종합합니다.",color_fn,"QUALITY")
    with c2:
        st.subheader("진입 타이밍"); _bar_card("현재 진입 여건",entry_score,"Entry Engine v2의 6개 공통 요소를 반영합니다.",color_fn,"ENTRY V2")
        st.markdown(f"**기본 판정:** {decision['base']}  \n**최종 판정:** {decision['final']}")
    if decision["risks"]:
        left,right=st.columns(2)
        with left: st.warning("**판정을 낮춘 위험 조건**\n\n"+"\n\n".join(f"• {x}" for x in decision["risks"]))
        with right: st.info("**판정 상향을 위한 확인 조건**\n\n"+"\n\n".join(f"• {x}" for x in decision["releases"]))
    else:
        st.success("현재 확인된 주요 위험 조건으로 인한 추가 하향은 없습니다. 무효화선과 권장 비중은 계속 지켜주세요.")
    cols=st.columns(4)
    for col,(label,key,desc) in zip(cols,[("추세","trend","이동평균·12개월"),("모멘텀","momentum","RSI·MACD·3개월"),("변동성","volatility","ATR 기반 안정성"),("수급","supply","거래량·VWAP·OBV")]):
        with col:_bar_card(label,m[key],desc,color_fn,"5-FACTOR")

    st.subheader("가격·추세·거래량")
    d=m["d"].tail(130); idx=d.index
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.72,.28],vertical_spacing=.04,specs=[[{}],[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=idx,y=d.Close,line=dict(color="#f8fafc",width=2.5),name="Close"),row=1,col=1)
    for srs,n,c,ds in [(m['ema20'],"EMA20","#3b82f6",None),(m['ema50'],"EMA50","#f59e0b",None),(m['ema200'],"EMA200","#a855f7","dash")]: fig.add_trace(go.Scatter(x=idx,y=srs.reindex(idx),line=dict(color=c,width=1.6,dash=ds),name=n),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=m['upper'].reindex(idx),line=dict(color="#64748b",width=1,dash="dot"),name="BB Upper"),row=1,col=1)
    fig.add_trace(go.Scatter(x=idx,y=m['lower'].reindex(idx),line=dict(color="#64748b",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(100,116,139,.08)",name="BB Lower"),row=1,col=1)
    colors=np.where(d.Close>=d.Open,"#38bdf8","#fb7185"); fig.add_trace(go.Bar(x=idx,y=d.Volume,marker_color=colors,name="Volume"),row=2,col=1)
    fig.add_trace(go.Scatter(x=idx,y=m['obv'].reindex(idx),line=dict(color="#f59e0b",width=1.5),name="OBV"),row=2,col=1,secondary_y=True)
    fig.update_layout(height=590,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0a1728",font=dict(color="#cbd5e1"),legend=dict(orientation="h"),margin=dict(l=30,r=25,t=55,b=25)); fig.update_xaxes(gridcolor="#20344d"); fig.update_yaxes(gridcolor="#20344d")
    st.plotly_chart(fig,use_container_width=True)
    st.info(f"핵심 관찰 · 현재 주가는 52주 범위의 {m['position']:.1f}% 위치입니다. 장기 추세 {m['trend']:.1f}점, RSI {m['rsi']:.1f}, 거래량은 20일 평균의 {m['vol_ratio']:.2f}배입니다.")

    eps_growth=_safe(inf.get("earningsGrowth"))*100; rev_growth=_safe(inf.get("revenueGrowth"))*100; roe=_safe(inf.get("returnOnEquity"))*100
    can={"C":_score(eps_growth,-10,30),"A":_score(roe,0,25),"N":_weighted([(m['trend'],.6),(_score(m['position'],40,95),.4)]),"S":_score(m['vol_ratio'],.6,2),"L":m['trend'],"I":m['supply'],"M":a['market']}
    st.subheader("CAN SLIM 분석")
    st.caption("원형 C·A·N·S·L·I·M과 보조 퀀트 지표를 분리했습니다. 데이터가 없는 항목은 총점 계산에서 제외합니다.")
    cols=st.columns(2)
    can_desc={"C":"최근 EPS 성장","A":"연간 수익성·ROE","N":"신고가·새로운 모멘텀","S":"거래량 수급","L":"시장 주도력","I":"기관 수급 대용지표","M":"시장 방향"}
    for i,(k,v) in enumerate(can.items()):
        with cols[i%2]:
            if np.isfinite(v): _bar_card(f"{k} · {can_desc[k]}",v,f"{can_desc[k]} 관련 확인 가능한 공개 데이터를 반영했습니다.",color_fn,"CAN SLIM")
            else: st.info(f"{k} · {can_desc[k]}: 데이터 없음 — 총점에서 제외")

    with st.expander("보조 퀀트 지표",expanded=False):
        quant={"평균회귀":float(np.clip(50-m['z']*18,0,100)),"모멘텀":m['momentum'],"다중 시간대":m['trend'],"낙폭 위험도":float(np.clip(100-abs(min(0,(a['now']/m['high52']-1)*100))*1.2,0,100)),"스마트머니 흐름":m['supply'],"Target Price Factor":float(np.clip(50+(a['resist'][-1]/a['now']-1)*180,0,100)),"통계적 Z-Score":float(np.clip(50-m['z']*15,0,100)),"변동성 조정":m['volatility']}
        cols=st.columns(2)
        for i,(k,v) in enumerate(quant.items()):
            with cols[i%2]:_bar_card(k,v,"보조 판단 지표이며 단독 매수 신호로 사용하지 않습니다.",color_fn,"QUANT")

    st.subheader("기술 지표")
    tech=[("RSI (14)",_fmt(m['rsi']),"70↑ 과매수 · 30↓ 과매도","bad" if m['rsi']>70 else "good" if 40<=m['rsi']<=65 else "neutral"),("ADX",_fmt(m['adx']),"25↑ 추세 존재 · 40↑ 강한 추세","good" if m['adx']>=25 else "neutral"),("ATR%",_fmt(m['atr_pct'],"%",2),"높을수록 변동성 큼","bad" if m['atr_pct']>6 else "neutral"),("VWAP 거리",_fmt((a['now']/m['vwap']-1)*100,"%"),"양수는 누적 평균가격 위","good" if a['now']>=m['vwap'] else "bad"),("12M 수익률",_fmt(m['ret12'],"%"),"장기 상대강도 참고","good" if m['ret12']>0 else "bad"),("3M 수익률",_fmt(m['ret3'],"%"),"단기 추세 확인","good" if m['ret3']>0 else "bad"),("거래량 비율",_fmt(m['vol_ratio'],"x",2),"1↑ 평균보다 활발","good" if m['vol_ratio']>=1.2 else "neutral"),("MACD 방향","상승" if m['macd']>m['signal'] else "하락","MACD와 시그널 비교","good" if m['macd']>m['signal'] else "bad")]
    for x in tech:_status_row(*x)

    st.subheader("재무 지표")
    financial=[("PER",_fmt(_safe(inf.get('trailingPE'))),"업종과 성장률을 함께 비교"),("PBR",_fmt(_safe(inf.get('priceToBook'))),"자산 대비 가격"),("ROE",_fmt(roe,"%"),"17% 이상 우수 참고"),("EPS 성장률",_fmt(eps_growth,"%"),"최근 공개 성장률"),("매출 성장률",_fmt(rev_growth,"%"),"양수는 매출 확장"),("영업이익률",_fmt(_safe(inf.get('operatingMargins'))*100,"%"),"사업 수익성"),("부채비율",_fmt(_safe(inf.get('debtToEquity')),"%"),"업종별 적정 수준 상이"),("시가총액",_fmt(_safe(inf.get('marketCap'))/1e9,"B"),"통화 단위는 Yahoo 원자료 기준")]
    for label,val,guide in financial:
        tone="neutral" if val=="—" else "good" if label in ("ROE","EPS 성장률","매출 성장률") and not val.startswith("-") else "neutral"
        _status_row(label,val,guide,tone)

    st.subheader("동일 업종 경쟁사 비교")
    defaults=_peer_defaults(symbol,inf); raw=st.text_input("비교 티커 · 쉼표로 수정",value=", ".join(defaults),key=f"peers_{symbol}")
    peers=[x.strip().upper() for x in raw.split(",") if x.strip()]
    pf=_peer_info(peers)
    if pf.empty: st.info("비교 가능한 경쟁사 데이터가 없습니다.")
    else: st.dataframe(pf.style.format({"시가총액":"{:,.1f}B","PER":"{:.1f}","PBR":"{:.2f}","ROE":"{:.1f}%","영업이익률":"{:.1f}%","12M":"{:+.1f}%"},na_rep="—"),hide_index=True,use_container_width=True)
    st.caption("산업 분류 기반 자동 후보입니다. 사업 구조가 다른 종목은 직접 삭제하거나 티커를 추가하세요.")

    st.subheader("실적 일정 · 공시·뉴스")
    left,right=st.columns(2)
    with left:
        st.markdown("**종목 일정**")
        try:
            cal=decision["calendar"]
            if isinstance(cal,dict):
                for k,v in cal.items(): st.write(f"{k}: {v}")
            elif cal is not None: st.dataframe(pd.DataFrame(cal),use_container_width=True)
            else: st.info("Yahoo에서 제공된 예정 일정이 없습니다.")
        except Exception: st.info("예정 일정을 불러오지 못했습니다.")
        st.caption("실적 발표 전후에는 변동성이 확대될 수 있어 신규 진입 비중 축소가 유리할 수 있습니다.")
    with right:
        st.markdown("**최근 뉴스**")
        rows=news_fn(symbol)
        if not rows: st.info("현재 불러온 뉴스가 없습니다.")
        for title,summary,url in rows[:5]:
            st.markdown(f"[{title}]({url})" if url else title); st.caption(summary[:180] if summary else "제목 기반 참고 뉴스")
    st.warning("기관 수급·컨센서스·공매도·한국 공시는 무료 Yahoo 데이터에서 누락될 수 있습니다. 없는 데이터는 임의 추정하지 않으며, DART 정식 공시는 별도 API 키 연동이 필요합니다.")

