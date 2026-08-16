# Stock Analyzer V5-a0.1

A Streamlit-based multi-lens stock analysis dashboard developed by Kijungnam.

The app brings together fundamental, technical, quantitative, options, and market-context analysis in one interface. Its Analysis Consensus layer highlights where these signals agree, strengthen, or diverge without reducing them to a simple average.

## Highlights

- Analysis Consensus with confidence and divergence interpretation
- Fundamental, technical, and quantitative stock analysis
- Five-trading-day score history and trend visualization
- U.S. options analysis, including market bias, option-chain statistics, expected move, max pain, and key strike levels
- Entry Engine V2 and practical Bull/Base/Bear scenarios
- Market dashboard with major assets, Treasury yields, and credit-market indicators
- Responsive layout for desktop and mobile

Missing or unsupported options data is shown as `N/A` and excluded from consensus calculations. Options signals remain separate from the original composite stock score.

## Run Locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Notes

The app uses public market-data sources, including Yahoo Finance, the U.S. Treasury, and FRED. Data may occasionally be delayed, unavailable, or incomplete. Local score history may be reset when a Streamlit Cloud app restarts or is redeployed.

## Disclaimer

This project is for informational and educational purposes only. It does not constitute investment advice, and all investment decisions remain the responsibility of the user.
