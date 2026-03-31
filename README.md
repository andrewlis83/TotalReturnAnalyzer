# Total Return Comparator

A small [Streamlit](https://streamlit.io/) app that compares hypothetical **$10,000** stock positions over a date range you choose. Data comes from **Yahoo Finance** (via `yfinance`).

For each ticker you can see:

- **Price only** — share price movement, ignoring dividends  
- **Total return (reinvested)** — uses adjusted close as a total-return proxy  
- **No reinvestment** — stock value plus cumulative dividend cash  

The UI shows a summary table, interactive charts (Plotly), and per-ticker dividend schedules.

## How to run

Requires **Python 3.10+** (or another version compatible with the packages in `requirements.txt`).

```bash
cd TotalReturnComparator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`). Pick a date range, enter tickers (comma- or newline-separated), and click **Run analysis**.

## Tests

```bash
pytest
```
