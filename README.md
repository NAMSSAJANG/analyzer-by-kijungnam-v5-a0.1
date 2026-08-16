# Stock Analyzer V5-a0.1 by Kijungnam — Multi-Lens Decision System

이 디렉터리와 `v5-a0.1` 브랜치는 기준 안정판 V5에서 분리한 파생 알파 버전입니다. 기존 종합점수 공식(펀더멘털 38%·테크니컬 42%·시장환경 20%)은 변경하지 않았으며 옵션 신호도 그 공식에 반영하지 않습니다.

- 검색 직후 종합·퀀트·옵션·시장환경을 비교하는 Analysis Consensus Hero Card
- 단순 평균점수가 아닌 방향 일치, 5D 변화, 결측 데이터와 데이터 품질 기반 Pattern/Confidence
- 종합·퀀트·Market Health의 날짜별 저장과 최근 5영업일 Score Momentum
- 옵션 Bias/Confirmation, Call·Put Volume/OI, IV, 만기별 Chain, Call/Put Wall, Max Pain, Expected Move
- 기술적 지지·저항과 옵션 Strike의 Confluence/Divergence
- 옵션 미지원 및 결측 종목은 N/A로 제외해 Consensus 분모 왜곡 방지
- 교체 가능한 `JsonScoreHistory` 저장 어댑터로 향후 PostgreSQL/Supabase 이전 가능
- 종합·퀀트가 공유하는 Entry Engine v2: Trend, Price Position, Momentum, Volume/OBV, Volatility, Market
- Option Entry Readiness: Direction, IV Efficiency, Liquidity, Risk/Reward, Time/DTE를 별도 평가하며 종합점수에는 미반영
- Bull/Base/Bear 시나리오에 발동 조건·대응 행동·목표·무효화 조건 표시

V4.1의 UI와 종합분석·퀀트분석·시장환경을 유지하면서 옵션분석을 네 번째 메뉴로 추가한 Streamlit 분석 앱입니다.

V4.1에서는 기존 화면과 분석 기능을 유지하면서 종목 검색 위에 `오늘의 퀀트 TOP 10`을 추가했습니다.

- `NASDAQ 100`, `S&P 500`, `KOSPI` 시장별 1~10위 전체 표시
- 추세·모멘텀·변동성·수급·시장환경 및 단기 과열 감점을 반영한 진입 타이밍 점수
- 순위, 종목명·티커, 퀀트 점수, 진입 적합도, 당일 등락률 표시
- 표의 종목을 선택하면 기존 상세 분석 종목으로 연결
- 전일 종가 기준 하루 1회 계산 및 캐시
- KOSPI는 무료 데이터의 안정성과 첫 로딩 속도를 위해 시가총액 상위 500개 종목 대상

한국거래소 종목 목록을 보완 검색하므로 `삼성전자`, `005930`, `005930.KS` 방식으로 한국 종목을 찾을 수 있습니다. 목록 조회가 일시적으로 실패하면 기존 Yahoo 검색으로 자동 전환됩니다.

- `종합분석`: V3.3의 상세 점수, 추세, 진입, 지지·저항, 대응 시나리오와 점수 이력
- `퀀트분석`: 기업 품질과 진입 시점 분리, CAN SLIM, 퀀트·기술·재무지표, 경쟁사, 실적·뉴스
- `옵션분석`: 미국 옵션 가능 종목의 만기·체인, Call/Put 거래량과 OI, Put/Call Ratio, IV, OI 집중 Strike, Expected Move, Option Confirmation
- `시장환경`: Market Health, Market Pulse 12, 금리·신용시장

옵션분석은 옵션 거래 추천이 아니라 현물 판단을 보완하는 정보 탭입니다. Option Confirmation은 기존 종합점수 공식에 포함되지 않습니다. 옵션이 없거나 무료 데이터 소스에서 체인을 제공하지 않는 KOSPI·KOSDAQ 종목은 안내 문구를 표시하고 나머지 기능은 정상 동작합니다.

## 실행

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

GitHub 저장소 루트에 이 폴더의 파일을 올린 뒤 Streamlit Community Cloud에서 `app.py`를 엔트리 파일로 지정합니다.

## 데이터와 점수

- Yahoo Finance 비공식 공개 엔드포인트와 `yfinance`를 사용하므로 지연, 누락, 심볼별 차이가 있을 수 있습니다.
- 기관 수급·컨센서스·공매도·한국 공시는 무료 데이터에서 누락될 수 있으며 없는 값은 임의 추정하지 않습니다.
- 점수는 0~100이며 Strong(80+), Good(65+), Neutral(45+), Weak(30+), Very Weak(<30)로 통일했습니다.
- 종합점수: 펀더멘털 38%, 테크니컬 42%, 시장환경 20%.
- 결과는 투자 권유가 아닌 참고 자료입니다.

## Score History 저장 주의

기본값은 `.data/score_history.json`입니다. Streamlit Cloud 로컬 디스크는 영구 저장소가 아니므로 재시작/재배포 때 사라질 수 있습니다. 화면의 JSON 내보내기/가져오기로 백업하십시오. 운영 환경에서는 외부 DB 연동을 권장합니다.
