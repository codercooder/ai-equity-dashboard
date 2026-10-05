# AI Equity Dashboard

A Streamlit dashboard for 14 AI-industry-chain sectors. Each sector has a market-cap-weighted index and a second-level stock table.

## Features
- 14 AI sectors with mutually exclusive constituent membership
- Market-cap-weighted sector indices (base = 1000)
- Stock table with ticker, company, latest/close price, daily return, volume, volume / 3M average, Forward P/E, YTD, 1M, 3M, 6M, 1Y, 3Y, 5Y, market cap, and index weight
- Optional Twelve Data real-time quotes during U.S. regular trading hours
- Automatic regular-session close display after market close
- SK hynix uses Nasdaq ADR ticker `SKHY` only; unavailable pre-listing history is shown as `—`
- Caching to reduce API usage

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add TWELVE_DATA_API_KEY if desired
streamlit run app.py
```

## Data policy
- Historical daily prices: Yahoo Finance via `yfinance`
- Market cap: Yahoo Finance `fast_info`
- Forward P/E: Yahoo Finance fundamentals, loaded only for the selected sector
- Intraday latest price / volume: Twelve Data `/quote` when configured and market is open; Yahoo fallback otherwise
- Returns are calculated from each listed security's own adjusted-price history. No proxy history is backfilled.

## Index methodology
For each sector, the dashboard estimates a fixed share count from current market cap / current adjusted price, then reconstructs a historical market-value series using adjusted prices. The series is normalized to 1000 at the first date with sufficient constituent data. This gives a practical market-cap-weighted research index while avoiding false backfilling.

For a production index, add corporate-action divisor maintenance and point-in-time shares outstanding.
