from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import streamlit as st
import yfinance as yf

from config import NAME_MAP

ET = ZoneInfo("America/New_York")
INDEX_BASE_DATE = pd.Timestamp("2021-01-04")
INDEX_BASE_VALUE = 1000.0


@dataclass
class MarketState:
    label: str
    is_open: bool
    now_et: datetime
    session_open: datetime | None
    session_close: datetime | None


def market_state() -> MarketState:
    now = datetime.now(ET)
    nyse = mcal.get_calendar("NYSE")
    start = (now.date() - timedelta(days=7)).isoformat()
    end = (now.date() + timedelta(days=1)).isoformat()
    schedule = nyse.schedule(start_date=start, end_date=end)

    if schedule.empty:
        return MarketState("CLOSED", False, now, None, None)

    today_rows = schedule[schedule.index.date == now.date()]
    if today_rows.empty:
        return MarketState("CLOSED", False, now, None, None)

    row = today_rows.iloc[0]
    open_et = row["market_open"].tz_convert(ET).to_pydatetime()
    close_et = row["market_close"].tz_convert(ET).to_pydatetime()
    is_open = open_et <= now <= close_et
    return MarketState("LIVE" if is_open else "CLOSED", is_open, now, open_et, close_et)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_history(tickers: tuple[str, ...], period: str = "10y") -> pd.DataFrame:
    """Adjusted daily close used for return and index calculations."""
    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"].copy() if "Adj Close" in data.columns.get_level_values(0) else data["Close"].copy()
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        px = data[[col]].rename(columns={col: tickers[0]})

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])

    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px.sort_index()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def download_volume_history(tickers: tuple[str, ...], period: str = "6mo") -> pd.DataFrame:
    data = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        vol = data["Volume"].copy()
    else:
        vol = data[["Volume"]].rename(columns={"Volume": tickers[0]})

    if isinstance(vol, pd.Series):
        vol = vol.to_frame(name=tickers[0])

    vol.index = pd.to_datetime(vol.index).tz_localize(None)
    return vol.sort_index()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def market_caps(tickers: tuple[str, ...]) -> dict[str, float]:
    """Market cap with fast_info first and info.marketCap as fallback."""
    out: dict[str, float] = {}

    for ticker in tickers:
        obj = yf.Ticker(ticker)
        value = None

        try:
            fi = obj.fast_info
            if hasattr(fi, "get"):
                value = fi.get("market_cap") or fi.get("marketCap")
            if value is None and hasattr(fi, "market_cap"):
                value = fi.market_cap
        except Exception:
            pass

        if value is None:
            try:
                value = obj.info.get("marketCap")
            except Exception:
                pass

        try:
            value = float(value)
            if np.isfinite(value) and value > 0:
                out[ticker] = value
        except Exception:
            pass

    return out


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fundamentals_for(tickers: tuple[str, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            out[ticker] = {
                "forwardPE": info.get("forwardPE"),
                "longName": info.get("longName") or info.get("shortName") or NAME_MAP.get(ticker, ticker),
            }
        except Exception:
            out[ticker] = {
                "forwardPE": None,
                "longName": NAME_MAP.get(ticker, ticker),
            }
    return out


def _twelve_data_key() -> str | None:
    """Read Twelve Data key from local .env/environment or Streamlit Cloud Secrets."""
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()

    if not key:
        try:
            key = str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
        except Exception:
            key = ""

    enabled = os.getenv("ENABLE_TWELVE_DATA", "true").lower() not in {"0", "false", "no"}
    return key if key and enabled else None


def twelve_data_enabled() -> bool:
    return bool(_twelve_data_key())


@st.cache_data(ttl=45, show_spinner=False)
def twelve_data_quotes(tickers: tuple[str, ...]) -> dict[str, dict]:
    key = _twelve_data_key()
    if not key:
        return {}

    try:
        r = requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": ",".join(tickers), "apikey": key},
            timeout=12,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return {}

    if len(tickers) == 1 and isinstance(payload, dict) and "symbol" in payload:
        payload = {tickers[0]: payload}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict] = {}
    for ticker in tickers:
        item = payload.get(ticker, {})
        if isinstance(item, dict) and item.get("status") != "error":
            out[ticker] = item
    return out


@st.cache_data(ttl=60, show_spinner=False)
def yahoo_latest(tickers: tuple[str, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ticker in tickers:
        try:
            obj = yf.Ticker(ticker)
            fi = obj.fast_info
            last_price = fi.get("last_price") if hasattr(fi, "get") else fi.last_price
            prev_close = fi.get("previous_close") if hasattr(fi, "get") else fi.previous_close
            hist = obj.history(period="5d", interval="1d", auto_adjust=False)

            volume = None
            close = None
            if not hist.empty:
                if pd.notna(hist["Volume"].iloc[-1]):
                    volume = float(hist["Volume"].iloc[-1])
                if pd.notna(hist["Close"].iloc[-1]):
                    close = float(hist["Close"].iloc[-1])

            out[ticker] = {
                "price": float(last_price) if last_price is not None else close,
                "previous_close": float(prev_close) if prev_close is not None else None,
                "volume": volume,
                "regular_close": close,
            }
        except Exception:
            continue
    return out


def latest_regular_close(prices: pd.DataFrame, ticker: str) -> tuple[float | None, float | None]:
    if ticker not in prices.columns:
        return None, None
    s = prices[ticker].dropna()
    if s.empty:
        return None, None
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    return last, prev


def price_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    eligible = s.loc[s.index <= target]
    return float(eligible.iloc[-1]) if not eligible.empty else None


def return_since(
    series: pd.Series,
    latest_price: float,
    months: int | None = None,
    years: int | None = None,
    ytd: bool = False,
) -> float | None:
    s = series.dropna()
    if s.empty or latest_price is None or not np.isfinite(latest_price):
        return None

    last_date = pd.Timestamp(s.index[-1])
    if ytd:
        target = pd.Timestamp(year=last_date.year - 1, month=12, day=31)
    elif months is not None:
        target = last_date - pd.DateOffset(months=months)
    elif years is not None:
        target = last_date - pd.DateOffset(years=years)
    else:
        return None

    # Do not backfill a requested horizon with a different security or a listing
    # that did not yet exist. A short tolerance handles weekends/holidays.
    first_date = pd.Timestamp(s.index[0])
    tolerance = pd.Timedelta(days=10)
    if first_date > target + tolerance:
        return None

    base_px = price_on_or_before(s, target)
    if base_px in (None, 0):
        return None
    return float(latest_price) / float(base_px) - 1.0


def avg_3m_volume(volumes: pd.DataFrame, ticker: str) -> float | None:
    if ticker not in volumes.columns:
        return None
    s = volumes[ticker].dropna().tail(63)
    return float(s.mean()) if not s.empty else None


def build_sector_index(
    prices: pd.DataFrame,
    tickers: list[str],
    caps: dict[str, float],
    base_date: pd.Timestamp | str = INDEX_BASE_DATE,
    base: float = INDEX_BASE_VALUE,
) -> pd.Series:
    """Research market-cap-weighted sector index, chain-linked from a fixed base.

    Methodology:
    - Base close: first available trading day on/after 2021-01-04 = 1000.
    - Current market cap / latest adjusted price is used to infer a fixed share count.
    - Daily sector returns are weighted by prior-day inferred market value.
    - A later-listed constituent enters only after it has a valid prior-day price,
      preventing an artificial index jump on listing day.
    - Missing constituent returns are excluded and the remaining prior-day weights
      are renormalized for that day.

    This is suitable as a research index. A production benchmark would maintain
    historical shares/free float and a corporate-action divisor.
    """
    base_date = pd.Timestamp(base_date).tz_localize(None)

    valid: list[str] = []
    shares: dict[str, float] = {}

    for ticker in tickers:
        if ticker not in prices.columns or ticker not in caps or caps[ticker] <= 0:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        last_px = float(s.iloc[-1])
        if not np.isfinite(last_px) or last_px <= 0:
            continue
        valid.append(ticker)
        shares[ticker] = float(caps[ticker]) / last_px

    if not valid:
        return pd.Series(dtype=float, name="Index")

    frame = prices[valid].copy().sort_index()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)

    # Asset daily returns; do not forward-fill through pre-IPO/missing periods.
    asset_returns = frame.pct_change(fill_method=None)

    # Prior-day market value determines today's weights.
    market_values = frame.mul(pd.Series(shares), axis=1)
    lagged_mv = market_values.shift(1)

    # A stock contributes only when it has both a daily return and lagged MV.
    eligible_mv = lagged_mv.where(asset_returns.notna())
    denom = eligible_mv.sum(axis=1, min_count=1)
    weights = eligible_mv.div(denom, axis=0)
    sector_return = (weights * asset_returns).sum(axis=1, min_count=1)

    # Base is the close of the first trading day on/after the chosen base date.
    candidate_dates = frame.index[frame.index >= base_date]
    if len(candidate_dates) == 0:
        return pd.Series(dtype=float, name="Index")

    start = candidate_dates[0]
    future_returns = sector_return.loc[sector_return.index > start].fillna(0.0)

    index = pd.Series(index=pd.Index([start]).append(future_returns.index), dtype=float, name="Index")
    index.loc[start] = float(base)

    if not future_returns.empty:
        index.loc[future_returns.index] = float(base) * (1.0 + future_returns).cumprod()

    return index.sort_index()


def pct_change_period(
    series: pd.Series,
    months: int | None = None,
    years: int | None = None,
    ytd: bool = False,
) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return return_since(s, float(s.iloc[-1]), months=months, years=years, ytd=ytd)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def stock_ohlcv(ticker: str) -> pd.DataFrame:
    """Full daily OHLCV history for the Level 3 stock charts."""
    try:
        obj = yf.Ticker(ticker)
        df = obj.history(
            period="max",
            interval="1d",
            auto_adjust=False,
            actions=False,
            repair=True,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except TypeError:
        df.index = df.index.tz_convert(None)
    return df.sort_index()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def valuation_history(ticker: str) -> pd.DataFrame:
    """Monthly historical Forward P/E and Price/Sales from Yahoo valuation measures.

    Requires yfinance >= 1.3.0. The function fails soft so the dashboard can
    continue to run if Yahoo does not provide valuation history for a symbol.
    """
    try:
        obj = yf.Ticker(ticker)
        table = obj.get_valuation_measures(freq="monthly", periods=None)
    except Exception:
        return pd.DataFrame(columns=["Forward P/E", "Price/Sales"])

    if table is None or table.empty:
        return pd.DataFrame(columns=["Forward P/E", "Price/Sales"])

    wanted = [x for x in ["Forward P/E", "Price/Sales"] if x in table.index]
    if not wanted:
        return pd.DataFrame(columns=["Forward P/E", "Price/Sales"])

    out_rows: list[dict] = []
    today = pd.Timestamp.today().normalize()

    for col in table.columns:
        if str(col).lower() == "current":
            dt = today
        else:
            dt = pd.to_datetime(str(col), errors="coerce")
            if pd.isna(dt):
                continue

        row = {"Date": dt}
        for measure in ["Forward P/E", "Price/Sales"]:
            value = table.loc[measure, col] if measure in table.index else np.nan
            try:
                row[measure] = float(value) if pd.notna(value) else np.nan
            except Exception:
                row[measure] = np.nan
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=["Forward P/E", "Price/Sales"])

    out = pd.DataFrame(out_rows).set_index("Date").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def eps_history(ticker: str) -> pd.DataFrame:
    """Historical reported quarterly EPS and a rolling four-quarter TTM EPS.

    Yahoo's earnings history provides the broadest free historical series for
    this dashboard. TTM EPS is the rolling sum of the latest four reported
    quarterly EPS observations. Missing history is left missing rather than
    backfilled from another security.
    """
    try:
        obj = yf.Ticker(ticker)
        earnings = obj.get_earnings_dates(limit=100)
    except Exception:
        earnings = None

    if earnings is None or earnings.empty:
        return pd.DataFrame(columns=["Quarterly EPS", "TTM EPS"])

    df = earnings.copy()
    df.index = pd.to_datetime(df.index)
    try:
        df.index = df.index.tz_localize(None)
    except TypeError:
        df.index = df.index.tz_convert(None)

    reported_col = None
    for candidate in ["Reported EPS", "reportedEPS", "epsActual"]:
        if candidate in df.columns:
            reported_col = candidate
            break

    if reported_col is None:
        return pd.DataFrame(columns=["Quarterly EPS", "TTM EPS"])

    eps = pd.to_numeric(df[reported_col], errors="coerce").dropna().sort_index()
    if eps.empty:
        return pd.DataFrame(columns=["Quarterly EPS", "TTM EPS"])

    # Some feeds can include duplicate timestamps around the same earnings event.
    eps = eps[~eps.index.duplicated(keep="last")]
    out = pd.DataFrame({"Quarterly EPS": eps})
    out["TTM EPS"] = out["Quarterly EPS"].rolling(4, min_periods=4).sum()
    return out

