from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal

from config import ALL_TICKERS, NAME_MAP, SECTORS

ET = ZoneInfo("America/New_York")


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
def download_history(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
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
        if "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"].copy()
        else:
            px = data["Close"].copy()
    else:
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        px = data[[col]].rename(columns={col: tickers[0]})

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0])
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
    return vol.sort_index()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def market_caps(tickers: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}

    for ticker in tickers:
        obj = yf.Ticker(ticker)
        value = None

        # 1. 优先 fast_info
        try:
            fi = obj.fast_info

            if hasattr(fi, "get"):
                value = (
                    fi.get("market_cap")
                    or fi.get("marketCap")
                )

            if value is None and hasattr(fi, "market_cap"):
                value = fi.market_cap

        except Exception:
            pass

        # 2. fast_info失败时回退 info
        if value is None:
            try:
                info = obj.info
                value = info.get("marketCap")
            except Exception:
                pass

        try:
            if value is not None:
                value = float(value)

                if np.isfinite(value) and value > 0:
                    out[ticker] = value

        except Exception:
            pass

    return out


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fundamentals_for(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Load slow-changing fundamentals only for the currently selected sector."""
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = {
                "forwardPE": info.get("forwardPE"),
                "longName": info.get("longName") or info.get("shortName") or NAME_MAP.get(t, t),
            }
        except Exception:
            out[t] = {"forwardPE": None, "longName": NAME_MAP.get(t, t)}
    return out


def _twelve_data_key() -> str | None:
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    enabled = os.getenv("ENABLE_TWELVE_DATA", "true").lower() not in {"0", "false", "no"}
    return key if key and enabled else None


@st.cache_data(ttl=45, show_spinner=False)
def twelve_data_quotes(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Fetch latest quotes for one selected sector.

    Twelve Data counts credits per symbol even in a batch request. Sector sizes in
    this dashboard are <= 6, deliberately keeping a refresh within the Basic
    plan's typical per-minute budget.
    """
    key = _twelve_data_key()
    if not key:
        return {}
    url = "https://api.twelvedata.com/quote"
    params = {"symbol": ",".join(tickers), "apikey": key}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return {}

    if len(tickers) == 1 and isinstance(payload, dict) and "symbol" in payload:
        payload = {tickers[0]: payload}
    if not isinstance(payload, dict):
        return {}

    out = {}
    for t in tickers:
        item = payload.get(t, {})
        if not isinstance(item, dict) or item.get("status") == "error":
            continue
        out[t] = item
    return out


@st.cache_data(ttl=60, show_spinner=False)
def yahoo_latest(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Fallback latest regular-session information via yfinance fast_info/history."""
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            obj = yf.Ticker(t)
            fi = obj.fast_info
            last_price = fi.get("last_price") if hasattr(fi, "get") else fi.last_price
            prev_close = fi.get("previous_close") if hasattr(fi, "get") else fi.previous_close
            hist = obj.history(period="5d", interval="1d", auto_adjust=False)
            volume = None
            close = None
            if not hist.empty:
                volume = float(hist["Volume"].iloc[-1]) if pd.notna(hist["Volume"].iloc[-1]) else None
                close = float(hist["Close"].iloc[-1]) if pd.notna(hist["Close"].iloc[-1]) else None
            out[t] = {
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
    s = s.loc[s.index <= target]
    return float(s.iloc[-1]) if not s.empty else None


def return_since(series: pd.Series, latest_price: float, months: int | None = None, years: int | None = None, ytd: bool = False) -> float | None:
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

    first_date = pd.Timestamp(s.index[0])
    # If the security did not yet exist around the requested horizon, do not proxy/backfill.
    tolerance = pd.Timedelta(days=10)
    if first_date > target + tolerance:
        return None
    base = price_on_or_before(s, target)
    if base in (None, 0):
        return None
    return latest_price / base - 1.0


def avg_3m_volume(volumes: pd.DataFrame, ticker: str) -> float | None:
    if ticker not in volumes.columns:
        return None
    s = volumes[ticker].dropna().tail(63)
    if s.empty:
        return None
    return float(s.mean())


def build_sector_index(prices: pd.DataFrame, tickers: list[str], caps: dict[str, float], base: float = 1000.0) -> pd.Series:
    valid = [t for t in tickers if t in prices.columns and t in caps and caps[t] > 0 and prices[t].dropna().size > 1]
    if not valid:
        return pd.Series(dtype=float)

    last_prices = {}
    for t in valid:
        s = prices[t].dropna()
        last_prices[t] = float(s.iloc[-1])

    # Implied current shares. Using adjusted prices yields a practical research index;
    # production methodology should maintain a corporate-action divisor.
    shares = {t: caps[t] / last_prices[t] for t in valid if last_prices[t] > 0}
    frame = prices[list(shares)].copy()
    market_value = frame.mul(pd.Series(shares), axis=1).sum(axis=1, min_count=max(1, len(shares) // 2))
    market_value = market_value.dropna()
    if market_value.empty or market_value.iloc[0] == 0:
        return pd.Series(dtype=float)
    return base * market_value / market_value.iloc[0]


def pct_change_period(series: pd.Series, months: int | None = None, years: int | None = None, ytd: bool = False) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    latest = float(s.iloc[-1])
    return return_since(s, latest, months=months, years=years, ytd=ytd)
