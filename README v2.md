# Investment and Finance App v2

This Streamlit app includes:

- Interactive assumption controls
- CAPM required return
- Relative P/E valuation
- Gordon Growth DDM for dividend-paying firms
- Risk metrics: annualized return, volatility, Sharpe ratio, VaR, CVaR, max drawdown
- Technical analysis rule:
  - Bullish if price > 5MA, 10MA, 20MA and 5MA > 10MA > 20MA
  - Bearish if price < 5MA, 10MA, 20MA and 5MA < 10MA < 20MA
  - Otherwise Not Trending
- Portfolio analysis tab with adjustable weights
- Watchlist scanner that applies the MA trend rule across multiple stocks
- CSV download for scanner results

## Files
- `investment_finance_app.py`
- `requirements.txt`
- `README.md`

## Run locally
```bash
pip install -r requirements.txt
streamlit run investment_finance_app.py
```

## Deploy to Streamlit Community Cloud
1. Create a public GitHub repository.
2. Upload `investment_finance_app.py`, `requirements.txt`, and `README.md`.
3. In Streamlit Community Cloud, choose Deploy App.
4. Select the GitHub repository and set the main file path to `investment_finance_app.py`.
5. Deploy.

## Notes
- The app uses Yahoo Finance data via yfinance.
- Some fundamental fields may be missing for certain tickers.
- DDM is best for dividend-paying firms.
- This is for educational use only.
