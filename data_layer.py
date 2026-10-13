from __future__ import annotations

import os
import re
import html as html_lib
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET_XML
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

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def twelve_data_history(symbol: str, start_date: str = "2010-01-01") -> pd.Series:
    """Daily close history from Twelve Data for instruments not carried by Yahoo."""
    key = _twelve_data_key()
    if not key or not symbol:
        return pd.Series(dtype=float)
    try:
        r = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": "1day",
                "start_date": start_date,
                "outputsize": 5000,
                "order": "ASC",
                "apikey": key,
            },
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        values = payload.get("values", []) if isinstance(payload, dict) else []
        rows = []
        for item in values:
            try:
                dt = pd.to_datetime(item.get("datetime"), errors="coerce")
                close = float(item.get("close"))
            except Exception:
                continue
            if pd.isna(dt) or not np.isfinite(close):
                continue
            rows.append((pd.Timestamp(dt).tz_localize(None), close))
        if not rows:
            return pd.Series(dtype=float)
        out = pd.Series(dict(rows), name=symbol, dtype=float).sort_index()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out[~out.index.duplicated(keep="last")]
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def csi300_history() -> pd.Series:
    """CSI 300 daily history using the canonical Shanghai index code 000300 (SHA)."""
    try:
        r = requests.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": "1.000300",
                "klt": "101",
                "fqt": "0",
                "beg": "20100101",
                "end": "20500101",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        r.raise_for_status()
        payload = r.json()
        klines = ((payload or {}).get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            parts = str(line).split(",")
            if len(parts) < 3:
                continue
            try:
                dt = pd.to_datetime(parts[0], errors="coerce")
                close = float(parts[2])
            except Exception:
                continue
            if pd.isna(dt) or not np.isfinite(close):
                continue
            rows.append((pd.Timestamp(dt).tz_localize(None), close))
        if not rows:
            return pd.Series(dtype=float)
        out = pd.Series(dict(rows), name="000300:SHA", dtype=float).sort_index()
        return out[~out.index.duplicated(keep="last")]
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=60, show_spinner=False)
def csi300_quote() -> dict:
    """Latest CSI 300 quote; regular close after the Shanghai session ends."""
    try:
        r = requests.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "secid": "1.000300",
                "fields": "f43,f58,f60,f170",
                "fltt": "2",
                "invt": "2",
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        r.raise_for_status()
        data = (r.json() or {}).get("data") or {}
        price = data.get("f43")
        prev = data.get("f60")
        price = float(price) if price not in (None, "-") else None
        prev = float(prev) if prev not in (None, "-") else None
        return {
            "price": price,
            "previous_close": prev,
            "name": data.get("f58") or "沪深300",
            "symbol": "000300:SHA",
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def nasdaq_index_snapshot(symbol: str) -> dict:
    """Best-effort current/close snapshot from Nasdaq Global Index Watch."""
    if not symbol:
        return {}
    try:
        r = requests.get(
            f"https://indexes.nasdaq.com/Index/Overview/{symbol}",
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            },
            timeout=12,
        )
        r.raise_for_status()
        txt = html_lib.unescape(re.sub(r"<[^>]+>", " ", r.text))
        txt = re.sub(r"\s+", " ", txt)
        m = re.search(
            r"Summary Details.*?Last\s+([0-9,]+(?:\.[0-9]+)?).*?Previous Close\s+([0-9,]+(?:\.[0-9]+)?)",
            txt,
            flags=re.I,
        )
        if not m:
            m = re.search(
                rf"{re.escape(symbol)}.*?DATA AS OF.*?([0-9,]+(?:\.[0-9]+)?).*?Previous Close\s+([0-9,]+(?:\.[0-9]+)?)",
                txt,
                flags=re.I,
            )
        if not m:
            return {}
        return {
            "price": float(m.group(1).replace(",", "")),
            "previous_close": float(m.group(2).replace(",", "")),
            "symbol": symbol,
        }
    except Exception:
        return {}


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
    """Daily OHLCV history for Level 3 charts.

    Use ``yf.download`` first because it is the same Yahoo path that already
    powers the dashboard's working historical-price tables.  Fall back through
    shorter periods rather than returning an empty chart when Yahoo rejects a
    ``period=max`` request for a symbol/session.
    """

    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        # yf.download can return a two-level column index even for one ticker.
        if isinstance(df.columns, pd.MultiIndex):
            # Usually level 0 is Price and level 1 is Ticker.
            if ticker in df.columns.get_level_values(-1):
                try:
                    df = df.xs(ticker, axis=1, level=-1, drop_level=True)
                except Exception:
                    pass
            if isinstance(df.columns, pd.MultiIndex):
                # If a single symbol still remains, collapse the redundant level.
                for level in range(df.columns.nlevels):
                    vals = df.columns.get_level_values(level)
                    if len(set(vals)) == 1:
                        try:
                            df.columns = df.columns.droplevel(level)
                            break
                        except Exception:
                            pass

        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(set(keep)):
            return pd.DataFrame()

        df = df[keep].copy()
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            df.index = df.index.tz_convert(None)

        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df.dropna(subset=["Open", "High", "Low", "Close"], how="any")

    # Try the same bulk-download route used elsewhere in the app first.
    for period in ("max", "10y", "5y"):
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            df = _normalise(raw)
            if not df.empty:
                return df
        except Exception:
            pass

    # Final fallback to Ticker.history.
    for period in ("10y", "5y", "2y"):
        try:
            raw = yf.Ticker(ticker).history(
                period=period,
                interval="1d",
                auto_adjust=False,
                actions=False,
            )
            df = _normalise(raw)
            if not df.empty:
                return df
        except Exception:
            pass

    return pd.DataFrame()


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def valuation_history(ticker: str) -> pd.DataFrame:
    """Reconstruct daily Forward P/E and Price/Sales series.

    Yahoo exposes valuation *snapshots* (normally monthly), not a genuine
    daily historical consensus series.  To make the chart respond every trading
    day to the stock price, each snapshot is converted into an implied
    denominator:

      implied forward EPS = stock price / Forward P/E
      implied sales/share  = stock price / Price/Sales

    The latest known denominator is then held constant until the next Yahoo
    valuation snapshot, while the daily close moves.  This produces a daily
    price-driven valuation series without pretending that Yahoo supplies daily
    analyst-estimate revisions.
    """
    empty = pd.DataFrame(columns=["Forward P/E", "Price/Sales"])

    try:
        obj = yf.Ticker(ticker)
        table = obj.get_valuation_measures(freq="monthly", periods=None)
    except Exception:
        return empty

    if table is None or table.empty:
        return empty

    wanted = [x for x in ["Forward P/E", "Price/Sales"] if x in table.index]
    if not wanted:
        return empty

    anchor_rows: list[dict] = []
    today = pd.Timestamp.today().normalize()
    for col in table.columns:
        if str(col).lower() == "current":
            dt = today
        else:
            dt = pd.to_datetime(str(col), errors="coerce")
            if pd.isna(dt):
                continue

        row = {"Date": pd.Timestamp(dt).tz_localize(None)}
        for measure in ["Forward P/E", "Price/Sales"]:
            value = table.loc[measure, col] if measure in table.index else np.nan
            try:
                value = float(value)
                row[measure] = value if np.isfinite(value) and value > 0 else np.nan
            except Exception:
                row[measure] = np.nan
        anchor_rows.append(row)

    if not anchor_rows:
        return empty

    anchors = pd.DataFrame(anchor_rows).set_index("Date").sort_index()
    anchors = anchors[~anchors.index.duplicated(keep="last")]

    ohlcv = stock_ohlcv(ticker)
    if ohlcv.empty or "Close" not in ohlcv.columns:
        return anchors[[c for c in ["Forward P/E", "Price/Sales"] if c in anchors.columns]]

    close = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna().sort_index()
    if close.empty:
        return empty

    result = pd.DataFrame(index=close.index, columns=["Forward P/E", "Price/Sales"], dtype=float)

    for measure in ["Forward P/E", "Price/Sales"]:
        if measure not in anchors.columns:
            continue
        vals = anchors[measure].dropna()
        if vals.empty:
            continue

        denominator_events = pd.Series(index=close.index, dtype=float)
        for dt, multiple in vals.items():
            eligible = close.loc[close.index <= pd.Timestamp(dt)]
            if eligible.empty or not np.isfinite(multiple) or multiple <= 0:
                continue
            trade_dt = eligible.index[-1]
            px = float(eligible.iloc[-1])
            if px > 0:
                denominator_events.loc[trade_dt] = px / float(multiple)

        denominator = denominator_events.ffill()
        daily_multiple = close / denominator
        result[measure] = daily_multiple.where((daily_multiple > 0) & np.isfinite(daily_multiple))

    result = result.dropna(how="all")
    return result


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


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def eps_revision_trend(ticker: str) -> pd.DataFrame:
    """Consensus annual EPS trend for the current and next fiscal year.

    Yahoo/yfinance exposes point-in-time analyst consensus at Current, 7, 30,
    60 and 90 days ago.  This is not a full historical time series, but it is
    a useful clean snapshot of recent estimate momentum.
    """
    columns = ["FY1 EPS", "FY2 EPS"]
    try:
        trend = yf.Ticker(ticker).get_eps_trend()
    except Exception:
        trend = None

    if trend is None or trend.empty:
        return pd.DataFrame(columns=columns)

    # yfinance row labels: 0q, +1q, 0y, +1y.  Annual rows are the most useful
    # for cross-company valuation and medium-term earnings revision analysis.
    annual_rows = {"0y": "FY1 EPS", "+1y": "FY2 EPS"}
    available = {k: v for k, v in annual_rows.items() if k in trend.index}
    if not available:
        return pd.DataFrame(columns=columns)

    offsets = [
        ("90daysAgo", 90),
        ("60daysAgo", 60),
        ("30daysAgo", 30),
        ("7daysAgo", 7),
        ("current", 0),
    ]
    today = pd.Timestamp.today().normalize()
    rows = []
    for field, days in offsets:
        row = {"Date": today - pd.Timedelta(days=days)}
        for src_row, label in available.items():
            value = trend.loc[src_row, field] if field in trend.columns else np.nan
            try:
                value = float(value)
                row[label] = value if np.isfinite(value) else np.nan
            except Exception:
                row[label] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows).set_index("Date").sort_index()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns].dropna(how="all")


@st.cache_data(ttl=60 * 30, show_spinner=False)
def company_event_snapshot(ticker: str) -> dict[str, list[dict]]:
    """Return recent company activity/news and publicly scheduled upcoming events.

    This intentionally avoids guessing. Recent items come from Yahoo Finance news,
    prioritising conference/event/earnings/product-related headlines when available.
    Upcoming items are drawn from announced future-event headlines, Yahoo's earnings
    dates and company calendar fields. Missing coverage is returned as an empty list.
    """
    obj = yf.Ticker(ticker)
    now = pd.Timestamp.now(tz=ET)
    recent: list[dict] = []
    upcoming: list[dict] = []

    # ----- Yahoo news: robust to old/new yfinance response shapes -----
    news_items = []
    try:
        news_items = obj.get_news(count=40, tab="news") or []
    except Exception:
        try:
            news_items = obj.news or []
        except Exception:
            news_items = []

    activity_kw = (
        "conference", "summit", "keynote", "fireside", "investor", "event",
        "presentation", "expo", "forum", "earnings", "launch", "unveil",
        "gtc", "computex", "ces", "analyst day", "capital markets day",
    )
    future_kw = (
        "will participate", "to participate", "will present", "to present",
        "will speak", "to speak", "upcoming", "scheduled", "conference",
        "investor day", "analyst day", "earnings call",
    )

    parsed_news: list[dict] = []
    for raw in news_items:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = content.get("title") or raw.get("title")
        if not title:
            continue

        source = None
        provider = content.get("provider")
        if isinstance(provider, dict):
            source = provider.get("displayName") or provider.get("name")
        source = source or content.get("publisher") or raw.get("publisher") or "Yahoo Finance"

        url = None
        for candidate in (content.get("canonicalUrl"), content.get("clickThroughUrl")):
            if isinstance(candidate, dict) and candidate.get("url"):
                url = candidate.get("url")
                break
        url = url or content.get("link") or raw.get("link")

        published = content.get("pubDate") or content.get("displayTime") or raw.get("providerPublishTime")
        dt = pd.NaT
        if isinstance(published, (int, float)) and published:
            try:
                dt = pd.to_datetime(published, unit="s", utc=True).tz_convert(ET)
            except Exception:
                dt = pd.NaT
        elif published:
            try:
                dt = pd.to_datetime(published, utc=True).tz_convert(ET)
            except Exception:
                dt = pd.NaT

        parsed_news.append({
            "title": str(title),
            "source": str(source),
            "url": url,
            "dt": dt,
        })

    parsed_news.sort(
        key=lambda x: x["dt"].timestamp() if pd.notna(x["dt"]) else 0,
        reverse=True,
    )

    # Prefer event-like items; fill remaining slots with latest company news.
    activity_items = [x for x in parsed_news if any(k in x["title"].lower() for k in activity_kw)]
    chosen = activity_items[:5]
    if len(chosen) < 5:
        seen = {x["title"] for x in chosen}
        chosen.extend([x for x in parsed_news if x["title"] not in seen][: 5 - len(chosen)])

    for item in chosen:
        dt = item["dt"]
        recent.append({
            "date": dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—",
            "title": item["title"],
            "source": item["source"],
            "url": item["url"],
        })

    # Announced future conference / presentation headlines.
    for item in parsed_news:
        title_lower = item["title"].lower()
        if any(k in title_lower for k in future_kw):
            upcoming.append({
                "date": "待定/见公告",
                "title": item["title"],
                "detail": item["source"],
                "url": item["url"],
            })
        if len(upcoming) >= 3:
            break

    # Yahoo earnings dates. Include only future dates.
    try:
        earnings = obj.get_earnings_dates(limit=24)
    except Exception:
        earnings = None

    if earnings is not None and not earnings.empty:
        dates = pd.to_datetime(earnings.index, utc=True, errors="coerce")
        for dt in dates:
            if pd.isna(dt):
                continue
            dt_et = dt.tz_convert(ET)
            if dt_et >= now - pd.Timedelta(hours=6):
                upcoming.append({
                    "date": dt_et.strftime("%Y-%m-%d"),
                    "title": "财报 / Earnings",
                    "detail": "Yahoo Finance earnings calendar",
                    "url": None,
                })
                break

    # Company calendar fields such as ex-dividend date or earnings date.
    try:
        cal = obj.calendar
    except Exception:
        cal = None

    def _calendar_items(calendar_obj):
        if calendar_obj is None:
            return []
        if isinstance(calendar_obj, dict):
            return list(calendar_obj.items())
        if isinstance(calendar_obj, pd.DataFrame) and not calendar_obj.empty:
            if calendar_obj.shape[1] == 1:
                return list(calendar_obj.iloc[:, 0].items())
            if calendar_obj.shape[0] == 1:
                return list(calendar_obj.iloc[0].items())
        return []

    label_map = {
        "Earnings Date": "财报 / Earnings",
        "Ex-Dividend Date": "除息日 / Ex-Dividend",
        "Dividend Date": "股息支付日 / Dividend",
    }

    for key, value in _calendar_items(cal):
        if str(key) not in label_map:
            continue
        vals = value if isinstance(value, (list, tuple)) else [value]
        for val in vals:
            try:
                dt = pd.to_datetime(val, utc=True, errors="coerce")
            except Exception:
                dt = pd.NaT
            if pd.isna(dt):
                continue
            dt_et = dt.tz_convert(ET)
            if dt_et >= now - pd.Timedelta(hours=6):
                upcoming.append({
                    "date": dt_et.strftime("%Y-%m-%d"),
                    "title": label_map[str(key)],
                    "detail": "Yahoo Finance company calendar",
                    "url": None,
                })

    # De-duplicate by (date, title), then order dated items ahead of undated announcements.
    dedup = {}
    for item in upcoming:
        dedup[(item.get("date"), item.get("title"))] = item
    upcoming = list(dedup.values())
    upcoming.sort(key=lambda x: (x.get("date") in {None, "待定/见公告"}, x.get("date") or "9999-12-31"))

    return {"recent": recent[:5], "upcoming": upcoming[:6]}


# -----------------------------------------------------------------------------
# Cross-company latest news feed (Google News RSS aggregator)
# -----------------------------------------------------------------------------

_NEWS_PRIORITY_SOURCES = {
    "reuters", "bloomberg", "the wall street journal", "wsj", "financial times",
    "cnbc", "marketwatch", "barron's", "associated press", "ap news", "the verge",
    "techcrunch", "fortune", "business insider", "yahoo finance", "investing.com",
}

_NEWS_IMPORTANT_KEYWORDS = (
    "earnings", "revenue", "profit", "guidance", "forecast", "outlook", "eps",
    "acquisition", "acquire", "merger", "deal", "partnership", "contract", "order",
    "launch", "unveil", "approval", "fda", "trial", "antitrust", "regulator",
    "investigation", "probe", "lawsuit", "settlement", "recall", "ban", "tariff",
    "export", "downgrade", "upgrade", "price target", "buyback", "dividend",
    "layoff", "cuts jobs", "ceo", "cfo", "resign", "appoint", "conference",
    "investor day", "capital markets day", "data center", "ai", "artificial intelligence",
)

_POSITIVE_NEWS_KEYWORDS = (
    "beat", "beats", "surge", "surges", "jump", "jumps", "rise", "rises", "gain", "gains",
    "raise guidance", "raises guidance", "upgrade", "upgraded", "approval", "approved",
    "record revenue", "record profit", "strong demand", "wins contract", "wins order",
    "partnership", "expands", "launches", "buyback", "dividend increase", "outperform",
)

_NEGATIVE_NEWS_KEYWORDS = (
    "miss", "misses", "fall", "falls", "drop", "drops", "slump", "slumps", "cut guidance",
    "cuts guidance", "downgrade", "downgraded", "warning", "weak demand", "lawsuit", "probe",
    "investigation", "recall", "delay", "delays", "ban", "antitrust", "layoff", "cuts jobs",
    "loss", "losses", "fraud", "breach", "outage", "reject", "rejected", "underperform",
)

_NEWS_ALIASES = {
    "NVDA": ("nvidia",),
    "AMD": ("advanced micro devices", "amd"),
    "AVGO": ("broadcom",),
    "MRVL": ("marvell",),
    "ARM": ("arm holdings", "arm ltd", "arm chip"),
    "INTC": ("intel",),
    "MU": ("micron",),
    "SNDK": ("sandisk", "san disk"),
    "WDC": ("western digital",),
    "SKHY": ("sk hynix", "skhynix"),
    "TSM": ("taiwan semiconductor", "tsmc"),
    "AMAT": ("applied materials",),
    "LRCX": ("lam research",),
    "KLAC": ("kla corp", "kla corporation"),
    "ANET": ("arista networks", "arista"),
    "COHR": ("coherent corp", "coherent"),
    "LITE": ("lumentum",),
    "DELL": ("dell technologies", "dell"),
    "HPE": ("hewlett packard enterprise",),
    "SMCI": ("super micro computer", "supermicro"),
    "VRT": ("vertiv",),
    "ETN": ("eaton",),
    "CEG": ("constellation energy",),
    "VST": ("vistra",),
    "GEV": ("ge vernova",),
    "NRG": ("nrg energy",),
    "BE": ("bloom energy",),
    "EQIX": ("equinix",),
    "DLR": ("digital realty",),
    "MSFT": ("microsoft",),
    "AMZN": ("amazon", "aws"),
    "GOOGL": ("alphabet", "google"),
    "ORCL": ("oracle",),
    "PLTR": ("palantir",),
    "NOW": ("servicenow",),
    "CRM": ("salesforce",),
    "SNOW": ("snowflake",),
    "DDOG": ("datadog",),
    "MDB": ("mongodb",),
    "ESTC": ("elastic nv", "elastic search", "elastic n.v."),
    "GTLB": ("gitlab",),
    "PANW": ("palo alto networks",),
    "CRWD": ("crowdstrike",),
    "ZS": ("zscaler",),
    "AAPL": ("apple",),
    "META": ("meta platforms", "meta ai", "facebook"),
    "QCOM": ("qualcomm",),
    "TSLA": ("tesla",),
    "MBLY": ("mobileye",),
    "TER": ("teradyne",),
    "ISRG": ("intuitive surgical",),
    "JPM": ("jpmorgan", "jp morgan"),
    "V": ("visa inc", "visa card", "visa payments"),
    "MA": ("mastercard",),
    "GE": ("ge aerospace", "general electric aerospace"),
    "LLY": ("eli lilly", "lilly"),
}


def _normalise_news_title(title: str) -> str:
    text = html_lib.unescape(str(title or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _headline_sentiment(title: str) -> str:
    text = _normalise_news_title(title).lower()
    pos = sum(1 for kw in _POSITIVE_NEWS_KEYWORDS if kw in text)
    neg = sum(1 for kw in _NEGATIVE_NEWS_KEYWORDS if kw in text)
    if pos > neg:
        return "正面"
    if neg > pos:
        return "负面"
    return "中性"


def _news_importance(title: str, source: str) -> int:
    text = _normalise_news_title(title).lower()
    src = str(source or "").lower()
    score = 0
    if any(s in src for s in _NEWS_PRIORITY_SOURCES):
        score += 2
    kw_hits = sum(1 for kw in _NEWS_IMPORTANT_KEYWORDS if kw in text)
    score += min(3, kw_hits)
    # Headlines carrying explicit market-moving language get an additional point.
    if any(kw in text for kw in _POSITIVE_NEWS_KEYWORDS + _NEGATIVE_NEWS_KEYWORDS):
        score += 1
    return score


def _news_importance_label(score: int) -> str:
    if score >= 4:
        return "高"
    if score >= 2:
        return "中"
    return "低"


def _news_event_type(title: str) -> str:
    text = _normalise_news_title(title).lower()
    buckets = [
        ("财报 / 指引", ("earnings", "revenue", "profit", "eps", "guidance", "forecast", "outlook")),
        ("产品 / 技术", ("launch", "unveil", "product", "chip", "gpu", "platform", "software", "model", "technology")),
        ("AI Capex / 数据中心", ("data center", "datacenter", "capex", "ai infrastructure", "gpu cluster", "cloud capacity")),
        ("M&A", ("acquisition", "acquire", "merger", "takeover", "buyout")),
        ("合作 / 订单", ("partnership", "partner", "contract", "order", "deal", "supplier", "customer")),
        ("监管 / 法律", ("regulator", "antitrust", "investigation", "probe", "lawsuit", "settlement", "ban", "recall", "fda", "approval", "trial")),
        ("分析师评级", ("upgrade", "downgrade", "price target", "outperform", "underperform", "overweight", "underweight")),
        ("管理层", (" ceo", " cfo", "chief executive", "chief financial", "resign", "appoint", "management")),
        ("资本回报 / 融资", ("buyback", "repurchase", "dividend", "offering", "bond", "debt", "financing")),
        ("宏观 / 贸易政策", ("tariff", "export", "sanction", "trade restriction", "china", "rate cut", "interest rate")),
        ("会议 / Investor Event", ("conference", "investor day", "capital markets day", "fireside", "keynote", "presentation")),
    ]
    for label, keywords in buckets:
        if any(keyword in text for keyword in keywords):
            return label
    return "其他"


def _aliases_for_ticker(ticker: str, name: str) -> tuple[str, ...]:
    aliases = list(_NEWS_ALIASES.get(ticker, ()))
    name_clean = re.sub(r"\b(inc\.?|corp\.?|corporation|plc|ltd\.?|holdings?|technologies)\b", "", str(name), flags=re.I)
    name_clean = re.sub(r"\s+", " ", name_clean).strip().lower()
    if len(name_clean) >= 4:
        aliases.append(name_clean)
    if len(ticker) >= 3:
        aliases.append(ticker.lower())
    # Preserve order while deduplicating.
    return tuple(dict.fromkeys(a.strip().lower() for a in aliases if a and a.strip()))


def _parse_google_news_sector(sector: str, members: tuple[tuple[str, str], ...], lookback_days: int) -> list[dict]:
    # One RSS request per sector keeps the dashboard responsive and avoids dozens
    # of individual company requests. Google News is used only as an aggregator;
    # the source field below preserves the original publisher shown by Google.
    terms = []
    for ticker, name in members:
        # Company names are more reliable than raw tickers in news search.
        # Short/generic tickers such as ARM/NOW/V/MA create noisy matches.
        terms.append(f'"{name}"')
    query = " OR ".join(terms) + f" when:{max(1, int(lookback_days))}d"

    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        r.raise_for_status()
        root = ET_XML.fromstring(r.content)
    except Exception:
        return []

    alias_map = {ticker: _aliases_for_ticker(ticker, name) for ticker, name in members}
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff = now_utc - pd.Timedelta(days=lookback_days)
    out: list[dict] = []

    for item in root.findall(".//item"):
        title = _normalise_news_title(item.findtext("title") or "")
        if not title:
            continue
        link = (item.findtext("link") or "").strip()
        description = _normalise_news_title(item.findtext("description") or "")
        source_node = item.find("source")
        source = _normalise_news_title(source_node.text if source_node is not None else "") or "Google News"
        pub_raw = item.findtext("pubDate")
        try:
            dt = pd.Timestamp(parsedate_to_datetime(pub_raw)).tz_convert("UTC")
        except Exception:
            dt = pd.NaT
        if pd.notna(dt) and dt < cutoff:
            continue

        haystack = f"{title} {description}".lower()
        related = []
        for ticker, aliases in alias_map.items():
            if any(alias in haystack for alias in aliases):
                related.append(ticker)

        # Google matched the sector query but the title/snippet may omit the company
        # name. In that case keep only headlines with obvious market-moving language;
        # they are still useful, but do not invent a company association.
        if not related:
            continue

        score = _news_importance(title, source)
        out.append({
            "published": dt,
            "source": source,
            "title": title,
            "sentiment": _headline_sentiment(title),
            "event_type": _news_event_type(title),
            "companies": related,
            "sector": sector,
            "url": link,
            "importance": score,
            "importance_label": _news_importance_label(score),
        })

    return out


@st.cache_data(ttl=60 * 15, show_spinner=False)
def company_news_feed(
    sectors: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Latest cross-company news from public publisher headlines via Google News RSS.

    Returns a deduplicated, newest-first table. Sentiment is a transparent
    headline-keyword classifier intended for triage only, not an investment signal.
    """
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(sectors)))) as pool:
        futures = {
            pool.submit(_parse_google_news_sector, sector, members, lookback_days): sector
            for sector, members in sectors
        }
        for future in as_completed(futures):
            try:
                rows.extend(future.result() or [])
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(
            columns=["published", "source", "title", "sentiment", "event_type", "companies", "sector", "url", "importance", "importance_label"]
        )

    # Merge duplicate syndicated headlines and union their company tags.
    merged: dict[str, dict] = {}
    for row in rows:
        key = re.sub(r"[^a-z0-9]+", " ", str(row.get("title", "")).lower()).strip()
        if not key:
            continue
        if key not in merged:
            merged[key] = row.copy()
            merged[key]["companies"] = set(row.get("companies", []))
        else:
            merged[key]["companies"].update(row.get("companies", []))
            # Keep the newest timestamp and the highest importance score.
            old_dt = merged[key].get("published")
            new_dt = row.get("published")
            if pd.notna(new_dt) and (pd.isna(old_dt) or new_dt > old_dt):
                merged[key]["published"] = new_dt
                if row.get("url"):
                    merged[key]["url"] = row.get("url")
                if row.get("source"):
                    merged[key]["source"] = row.get("source")
            merged[key]["importance"] = max(int(merged[key].get("importance", 0)), int(row.get("importance", 0)))

    clean_rows = []
    for row in merged.values():
        row = row.copy()
        row["companies"] = ", ".join(sorted(row.get("companies", set())))
        clean_rows.append(row)

    df = pd.DataFrame(clean_rows)
    if df.empty:
        return df
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.sort_values(["published", "importance"], ascending=[False, False], na_position="last")
    return df.reset_index(drop=True)
