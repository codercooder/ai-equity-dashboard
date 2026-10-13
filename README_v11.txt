AI Equity Dashboard v11

Changes:
- Fixed CSI 300 using canonical Shanghai code 000300 (displayed as 000300:SHA), with Eastmoney live/history fallback and Yahoo 000300.SS fallback.
- Added Nasdaq CTA Artificial Intelligence Index (NQINTEL).
- Added iShares Russell 1000 Value ETF (IWD).
- Added iShares Russell 1000 Growth ETF (IWF).
- Added NASDAQ Biotechnology Index (^NBI / NBI).
- Existing market overview columns and manual Twelve Data refresh logic retained.
- NQINTEL uses Twelve Data when available and Nasdaq GIW snapshot fallback; historical returns are shown when Twelve Data historical data is available, otherwise unavailable horizons remain blank.
