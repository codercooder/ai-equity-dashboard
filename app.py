from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from config import ALL_TICKERS, SECTORS
from data_layer import (
    INDEX_BASE_DATE,
    INDEX_BASE_VALUE,
    avg_3m_volume,
    build_sector_index,
    download_history,
    download_volume_history,
    fundamentals_for,
    latest_regular_close,
    market_caps,
    market_state,
    pct_change_period,
    return_since,
    twelve_data_enabled,
    twelve_data_quotes,
    yahoo_latest,
    stock_ohlcv,
    valuation_history,
    eps_history,
)

load_dotenv()

st.set_page_config(
    page_title="AI Equity Dashboard",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(127,127,127,.06);
        border: 1px solid rgba(127,127,127,.18);
        padding: 12px;
        border-radius: 12px;
    }
    .small-note {font-size:.82rem; opacity:.72;}
    </style>
    """,
    unsafe_allow_html=True,
)

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
ZERO_COLOR = "#6b7280"
VOLUME_ALERT_COLOR = "#d97706"


def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:+.2%}"


def fmt_num(x, decimals=1):
    return "—" if x is None or pd.isna(x) else f"{x:,.{decimals}f}"


def fmt_price(x):
    return "—" if x is None or pd.isna(x) else f"${x:,.2f}"


def fmt_compact(x, dollar=False):
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    prefix = "$" if dollar else ""
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(x) >= div:
            return f"{prefix}{x / div:.2f}{unit}"
    return f"{prefix}{x:,.0f}"


def fmt_ratio(x):
    return "—" if x is None or pd.isna(x) else f"{x:.2f}x"


def fmt_pe(x):
    return "—" if x is None or pd.isna(x) else f"{x:.1f}x"


def return_css(value):
    if value is None or pd.isna(value):
        return f"color: {ZERO_COLOR};"
    if float(value) > 0:
        return f"color: {POSITIVE_COLOR}; font-weight: 600;"
    if float(value) < 0:
        return f"color: {NEGATIVE_COLOR}; font-weight: 600;"
    return f"color: {ZERO_COLOR};"


def volume_ratio_css(value):
    if value is None or pd.isna(value):
        return f"color: {ZERO_COLOR};"
    if float(value) >= 1.5:
        return f"color: {VOLUME_ALERT_COLOR}; font-weight: 700;"
    return ""


def style_numeric_table(
    df: pd.DataFrame,
    formatters: dict,
    return_cols: list[str],
    volume_col: str | None = None,
):
    """Keep numeric dtype so Streamlit right-aligns numeric columns."""
    styler = df.style.format(formatters, na_rep="—")

    cols = [c for c in return_cols if c in df.columns]
    if cols:
        styler = styler.map(return_css, subset=cols)

    if volume_col and volume_col in df.columns:
        styler = styler.map(volume_ratio_css, subset=[volume_col])

    return styler


# -----------------------------------------------------------------------------
# Market state / API state
# -----------------------------------------------------------------------------
state = market_state()
td_enabled = twelve_data_enabled()
status_color = "🟢" if state.is_open else "⚪"

if "live_quotes" not in st.session_state:
    st.session_state.live_quotes = {}
if "live_updated_at" not in st.session_state:
    st.session_state.live_updated_at = {}


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
header_l, header_r = st.columns([3, 2])
with header_l:
    st.title("AI Equity Dashboard")
    st.caption("14个AI产业链板块 · 市值加权研究指数 · 个股行情与估值")
with header_r:
    st.markdown(
        f"**{status_color} Market Status: {state.label}**  \n"
        f"ET {state.now_et.strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"Twelve Data: {'Ready' if td_enabled else 'Not configured'}"
    )


# -----------------------------------------------------------------------------
# Core data
# -----------------------------------------------------------------------------
with st.spinner("加载历史行情与市值数据…"):
    prices = download_history(tuple(ALL_TICKERS), period="10y")
    caps = market_caps(tuple(ALL_TICKERS))

if prices.empty:
    st.error("暂时无法取得历史行情。请检查网络或稍后重试。")
    st.stop()


# -----------------------------------------------------------------------------
# Sector selection + manual Twelve Data refresh
# -----------------------------------------------------------------------------
selected_sector = st.selectbox(
    "选择板块查看二级个股",
    list(SECTORS.keys()),
    index=0,
)
selected_members = SECTORS[selected_sector]
selected_tickers = [ticker for ticker, _ in selected_members]

btn_col, note_col = st.columns([1, 4])

with btn_col:
    refresh_live = st.button(
        "刷新实时行情",
        type="primary",
        disabled=(not td_enabled or not state.is_open),
        use_container_width=True,
    )

with note_col:
    last_refresh = st.session_state.live_updated_at.get(selected_sector)
    if not td_enabled:
        st.caption("Twelve Data API key 未配置；当前使用收盘数据。")
    elif not state.is_open:
        st.caption("美股当前已收盘：一级和二级均显示 regular-session close 的当日变动。")
    elif last_refresh:
        st.caption(f"当前板块最近一次 Twelve Data 手动刷新：{last_refresh} ET")
    else:
        st.caption("美股交易时段：点击“刷新实时行情”后，一级‘当日’和二级 Daily Return 同步使用 Twelve Data。")

if refresh_live:
    try:
        # Manual means manual: clear the short cache before each button-triggered request.
        twelve_data_quotes.clear()
    except Exception:
        pass

    with st.spinner("正在从 Twelve Data 刷新当前板块…"):
        quotes = twelve_data_quotes(tuple(selected_tickers))

    if quotes:
        st.session_state.live_quotes[selected_sector] = quotes
        st.session_state.live_updated_at[selected_sector] = datetime.now(state.now_et.tzinfo).strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"已刷新 {len(quotes)} / {len(selected_tickers)} 只股票。一级板块‘当日’与二级 Daily Return 已同步更新。")
    else:
        st.warning("Twelve Data 未返回有效行情；本页继续显示最近 regular-session close 数据。")

# Only use stored live data when the market is open.
td_q = (
    st.session_state.live_quotes.get(selected_sector, {})
    if state.is_open
    else {}
)


def live_sector_daily_return(
    tickers: list[str],
    quotes: dict[str, dict],
) -> float | None:
    """Calculate the sector's intraday move using the same stock returns shown in Level 2.

    Constituents are weighted by market cap. Only names with a valid Twelve Data
    price/previous close and market cap enter the calculation; weights are
    renormalized across the successfully refreshed names.
    """
    numerator = 0.0
    denominator = 0.0

    for ticker in tickers:
        td = quotes.get(ticker, {})
        cap = caps.get(ticker)
        if not td or not cap:
            continue

        hist_last, hist_prev = latest_regular_close(prices, ticker)

        try:
            latest_price = float(td.get("close")) if td.get("close") is not None else None
        except Exception:
            latest_price = None

        try:
            prev_close = (
                float(td.get("previous_close"))
                if td.get("previous_close") is not None
                else hist_prev
            )
        except Exception:
            prev_close = hist_prev

        if latest_price is None or prev_close in (None, 0):
            continue

        stock_return = latest_price / float(prev_close) - 1.0
        numerator += float(cap) * stock_return
        denominator += float(cap)

    return numerator / denominator if denominator > 0 else None


# -----------------------------------------------------------------------------
# Level 1 sector table
# -----------------------------------------------------------------------------
st.subheader("一级：AI产业链板块")

sector_rows: list[dict] = []
sector_series: dict[str, pd.Series] = {}

for sector, members in SECTORS.items():
    tickers = [ticker for ticker, _ in members]
    idx = build_sector_index(
        prices,
        tickers,
        caps,
        base_date=INDEX_BASE_DATE,
        base=INDEX_BASE_VALUE,
    )
    sector_series[sector] = idx

    latest = float(idx.iloc[-1]) if not idx.empty else None
    close_daily = float(idx.pct_change(fill_method=None).iloc[-1]) if len(idx) >= 2 else None
    daily = close_daily

    # During the session, a manually refreshed selected sector uses Twelve Data
    # exactly like Level 2 Daily Return. Other sectors remain at their latest close
    # until the user selects and refreshes them.
    if state.is_open and sector == selected_sector and td_q:
        live_daily = live_sector_daily_return(tickers, td_q)
        if live_daily is not None:
            daily = live_daily

            # Also update the displayed index point intraday from the most recent
            # completed index close. Historical period returns remain close-based.
            if not idx.empty:
                if pd.Timestamp(idx.index[-1]).date() == state.now_et.date() and len(idx) >= 2:
                    prior_index_close = float(idx.iloc[-2])
                else:
                    prior_index_close = float(idx.iloc[-1])
                latest = prior_index_close * (1.0 + live_daily)

    total_cap = sum(caps.get(ticker, 0.0) for ticker in tickers)

    sector_rows.append(
        {
            "板块": sector,
            "指数": latest,
            "当日": daily,
            "YTD": pct_change_period(idx, ytd=True),
            "1M": pct_change_period(idx, months=1),
            "3M": pct_change_period(idx, months=3),
            "6M": pct_change_period(idx, months=6),
            "1Y": pct_change_period(idx, years=1),
            "3Y": pct_change_period(idx, years=3),
            "5Y": pct_change_period(idx, years=5),
            "总市值": total_cap,
        }
    )

sector_df = pd.DataFrame(sector_rows)

sector_formatters = {
    "指数": lambda x: fmt_num(x, 1),
    "当日": fmt_pct,
    "YTD": fmt_pct,
    "1M": fmt_pct,
    "3M": fmt_pct,
    "6M": fmt_pct,
    "1Y": fmt_pct,
    "3Y": fmt_pct,
    "5Y": fmt_pct,
    "总市值": lambda x: fmt_compact(x, dollar=True),
}
sector_return_cols = ["当日", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"]
sector_styler = style_numeric_table(
    sector_df,
    formatters=sector_formatters,
    return_cols=sector_return_cols,
)

st.dataframe(
    sector_styler,
    use_container_width=True,
    hide_index=True,
    height=535,
)

# -----------------------------------------------------------------------------
# Sector chart
# -----------------------------------------------------------------------------
idx = sector_series[selected_sector]
if not idx.empty:
    chart_df = idx.rename("Index").reset_index()
    chart_df.columns = ["Date", "Index"]

    fig = px.line(
        chart_df,
        x="Date",
        y="Index",
        title=(
            f"{selected_sector} — 市值加权研究指数 "
            f"（{INDEX_BASE_DATE.strftime('%Y-%m-%d')} = {INDEX_BASE_VALUE:.0f}）"
        ),
    )
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
        yaxis_title="Index",
        xaxis_title="Date",
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Level 2 stock table
# -----------------------------------------------------------------------------
st.subheader("二级：个股明细")

volumes = download_volume_history(tuple(selected_tickers), period="6mo")
fundamentals = fundamentals_for(tuple(selected_tickers))
yahoo_q = yahoo_latest(tuple(selected_tickers))

sector_cap = sum(
    caps.get(ticker, 0.0)
    for ticker in selected_tickers
    if caps.get(ticker, 0.0) > 0
)

rows: list[dict] = []
weighted_live_return_num = 0.0
weighted_live_return_den = 0.0

for ticker, configured_name in selected_members:
    hist_last, hist_prev = latest_regular_close(prices, ticker)
    yq = yahoo_q.get(ticker, {})
    td = td_q.get(ticker, {})

    if state.is_open and td:
        try:
            latest_price = float(td.get("close")) if td.get("close") is not None else None
        except Exception:
            latest_price = None
        try:
            prev_close = (
                float(td.get("previous_close"))
                if td.get("previous_close") is not None
                else hist_prev
            )
        except Exception:
            prev_close = hist_prev
        try:
            volume = float(td.get("volume")) if td.get("volume") is not None else None
        except Exception:
            volume = None
        source = "Twelve Data"

    elif state.is_open:
        latest_price = yq.get("price") or hist_last
        prev_close = yq.get("previous_close") or hist_prev
        volume = yq.get("volume")
        source = "Yahoo"

    else:
        latest_price = yq.get("regular_close") or hist_last
        prev_close = hist_prev
        volume = yq.get("volume")
        source = "Close"

    daily_ret = None
    if latest_price is not None and prev_close not in (None, 0):
        daily_ret = float(latest_price) / float(prev_close) - 1.0

    avg_vol = avg_3m_volume(volumes, ticker)
    vol_ratio = (
        float(volume) / float(avg_vol)
        if volume is not None and avg_vol not in (None, 0)
        else None
    )

    price_series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
    fp = fundamentals.get(ticker, {}).get("forwardPE")
    name = fundamentals.get(ticker, {}).get("longName") or configured_name
    cap = caps.get(ticker)
    weight = float(cap) / float(sector_cap) if cap and sector_cap else None

    if source == "Twelve Data" and daily_ret is not None and cap:
        weighted_live_return_num += float(cap) * float(daily_ret)
        weighted_live_return_den += float(cap)

    rows.append(
        {
            "Ticker": ticker,
            "Name": name,
            "Price": latest_price,
            "Daily Return": daily_ret,
            "Volume": volume,
            "Vol / 3M Avg": vol_ratio,
            "Forward P/E": fp,
            "YTD Return": return_since(price_series, latest_price, ytd=True) if latest_price else None,
            "1M Return": return_since(price_series, latest_price, months=1) if latest_price else None,
            "3M Return": return_since(price_series, latest_price, months=3) if latest_price else None,
            "6M Return": return_since(price_series, latest_price, months=6) if latest_price else None,
            "1Y Return": return_since(price_series, latest_price, years=1) if latest_price else None,
            "3Y Return": return_since(price_series, latest_price, years=3) if latest_price else None,
            "5Y Return": return_since(price_series, latest_price, years=5) if latest_price else None,
            "Market Cap": cap,
            "Index Weight": weight,
            "Source": source,
        }
    )

stock_df = pd.DataFrame(rows)
preferred_order = [
    "Ticker",
    "Name",
    "Price",
    "Daily Return",
    "Volume",
    "Vol / 3M Avg",
    "Forward P/E",
    "YTD Return",
    "1M Return",
    "3M Return",
    "6M Return",
    "1Y Return",
    "3Y Return",
    "5Y Return",
    "Market Cap",
    "Index Weight",
    "Source",
]
stock_df = stock_df[preferred_order]

stock_formatters = {
    "Price": fmt_price,
    "Daily Return": fmt_pct,
    "Volume": lambda x: fmt_compact(x, dollar=False),
    "Vol / 3M Avg": fmt_ratio,
    "Forward P/E": fmt_pe,
    "YTD Return": fmt_pct,
    "1M Return": fmt_pct,
    "3M Return": fmt_pct,
    "6M Return": fmt_pct,
    "1Y Return": fmt_pct,
    "3Y Return": fmt_pct,
    "5Y Return": fmt_pct,
    "Market Cap": lambda x: fmt_compact(x, dollar=True),
    "Index Weight": fmt_pct,
}
stock_return_cols = [
    "Daily Return",
    "YTD Return",
    "1M Return",
    "3M Return",
    "6M Return",
    "1Y Return",
    "3Y Return",
    "5Y Return",
]
stock_styler = style_numeric_table(
    stock_df,
    formatters=stock_formatters,
    return_cols=stock_return_cols,
    volume_col="Vol / 3M Avg",
)

st.dataframe(
    stock_styler,
    use_container_width=True,
    hide_index=True,
    height=330,
)



# -----------------------------------------------------------------------------
# Level 3 single-stock analytics
# -----------------------------------------------------------------------------
st.subheader("三级：个股指标图")

# Every stock in the 14-sector universe appears once in this selector.
stock_to_sector: dict[str, str] = {}
stock_to_name: dict[str, str] = {}
for sector_name, members in SECTORS.items():
    for ticker, name in members:
        stock_to_sector[ticker] = sector_name
        stock_to_name[ticker] = name

stock_options = list(stock_to_sector.keys())
option_labels = {ticker: f"{ticker} — {stock_to_name[ticker]}" for ticker in stock_options}

def _default_level3_index() -> int:
    preferred = selected_tickers[0] if selected_tickers else stock_options[0]
    try:
        return stock_options.index(preferred)
    except ValueError:
        return 0

level3_ticker = st.selectbox(
    "选择个股",
    stock_options,
    index=_default_level3_index(),
    format_func=lambda x: option_labels[x],
    key="level3_ticker",
)

sector_name = stock_to_sector[level3_ticker]
st.caption(f"所属板块：**{sector_name}**")

TIME_RANGES = ["YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX"]
selected_range = st.radio(
    "时间范围",
    TIME_RANGES,
    index=4,
    horizontal=True,
    key="level3_range",
)


def range_start(last_date: pd.Timestamp, label: str) -> pd.Timestamp | None:
    last_date = pd.Timestamp(last_date).tz_localize(None)
    if label == "YTD":
        return pd.Timestamp(year=last_date.year, month=1, day=1)
    if label == "1M":
        return last_date - pd.DateOffset(months=1)
    if label == "3M":
        return last_date - pd.DateOffset(months=3)
    if label == "6M":
        return last_date - pd.DateOffset(months=6)
    if label == "1Y":
        return last_date - pd.DateOffset(years=1)
    if label == "3Y":
        return last_date - pd.DateOffset(years=3)
    if label == "5Y":
        return last_date - pd.DateOffset(years=5)
    return None


def clip_to_range(df: pd.DataFrame | pd.Series, label: str):
    if df is None or len(df) == 0:
        return df
    last_date = pd.Timestamp(df.index.max()).tz_localize(None)
    start = range_start(last_date, label)
    if start is None:
        return df
    return df.loc[df.index >= start]


with st.spinner(f"加载 {level3_ticker} 图表数据…"):
    ohlcv = stock_ohlcv(level3_ticker)
    valuations = valuation_history(level3_ticker)
    eps_df = eps_history(level3_ticker)

if ohlcv.empty:
    st.warning(f"{level3_ticker} 暂时没有可用的历史 OHLCV 数据。")
else:
    # Technical indicators are calculated before clipping so the first visible
    # point can still have 10/30/120-day averages and MACD context.
    tech = ohlcv.copy()
    tech["MA10"] = tech["Close"].rolling(10, min_periods=10).mean()
    tech["MA30"] = tech["Close"].rolling(30, min_periods=30).mean()
    tech["MA120"] = tech["Close"].rolling(120, min_periods=120).mean()

    ema12 = tech["Close"].ewm(span=12, adjust=False).mean()
    ema26 = tech["Close"].ewm(span=26, adjust=False).mean()
    tech["MACD"] = ema12 - ema26
    tech["Signal"] = tech["MACD"].ewm(span=9, adjust=False).mean()
    tech["Histogram"] = tech["MACD"] - tech["Signal"]

    visible = clip_to_range(tech, selected_range)

    # 1. Candlestick + moving averages
    price_fig = go.Figure()
    price_fig.add_trace(
        go.Candlestick(
            x=visible.index,
            open=visible["Open"],
            high=visible["High"],
            low=visible["Low"],
            close=visible["Close"],
            name="Price",
        )
    )
    price_fig.add_trace(go.Scatter(x=visible.index, y=visible["MA10"], mode="lines", name="MA10", line=dict(color="#2563eb", width=1.5)))
    price_fig.add_trace(go.Scatter(x=visible.index, y=visible["MA30"], mode="lines", name="MA30", line=dict(color="#f59e0b", width=1.5)))
    price_fig.add_trace(go.Scatter(x=visible.index, y=visible["MA120"], mode="lines", name="MA120", line=dict(color="#7c3aed", width=1.5)))
    price_fig.update_layout(
        title=f"1. {level3_ticker} 股价 — Candlestick + MA10 / MA30 / MA120",
        height=500,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # 2. Volume
    volume_fig = go.Figure(
        data=[go.Bar(x=visible.index, y=visible["Volume"], name="Volume")]
    )
    volume_fig.update_layout(
        title=f"2. {level3_ticker} 成交量",
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Volume",
        showlegend=False,
    )
    st.plotly_chart(volume_fig, use_container_width=True)

    # 3. MACD (12, 26, 9)
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Bar(x=visible.index, y=visible["Histogram"], name="Histogram"))
    macd_fig.add_trace(go.Scatter(x=visible.index, y=visible["MACD"], mode="lines", name="MACD"))
    macd_fig.add_trace(go.Scatter(x=visible.index, y=visible["Signal"], mode="lines", name="Signal"))
    macd_fig.add_hline(y=0, line_width=1, line_dash="dot")
    macd_fig.update_layout(
        title=f"3. {level3_ticker} MACD (12, 26, 9)",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="MACD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(macd_fig, use_container_width=True)

# 4 & 5. Valuation history. Yahoo valuation history is generally monthly;
# if a newly listed stock has no history in the requested window we simply do
# not draw that chart, per the requested rule.
visible_val = clip_to_range(valuations, selected_range) if not valuations.empty else valuations

if visible_val is not None and not visible_val.empty and "Forward P/E" in visible_val.columns and visible_val["Forward P/E"].notna().any():
    pe_data = visible_val["Forward P/E"].dropna()
    pe_fig = go.Figure(go.Scatter(x=pe_data.index, y=pe_data.values, mode="lines+markers", name="Forward P/E"))
    pe_fig.update_layout(
        title=f"4. {level3_ticker} Forward P/E",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Forward P/E (x)",
        showlegend=False,
    )
    st.plotly_chart(pe_fig, use_container_width=True)
else:
    st.caption(f"4. {level3_ticker}：所选时间范围内暂无可用的 Forward P/E 历史数据。")

if visible_val is not None and not visible_val.empty and "Price/Sales" in visible_val.columns and visible_val["Price/Sales"].notna().any():
    ps_data = visible_val["Price/Sales"].dropna()
    ps_fig = go.Figure(go.Scatter(x=ps_data.index, y=ps_data.values, mode="lines+markers", name="Price/Sales"))
    ps_fig.update_layout(
        title=f"5. {level3_ticker} Price / Sales",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="Price / Sales (x)",
        showlegend=False,
    )
    st.plotly_chart(ps_fig, use_container_width=True)
else:
    st.caption(f"5. {level3_ticker}：所选时间范围内暂无可用的 Price/Sales 历史数据。")

# 6. EPS. Use reported quarterly EPS; TTM EPS is the rolling sum of four
# reported quarters, which is a useful common earnings-per-share trend measure.
visible_eps = clip_to_range(eps_df, selected_range) if not eps_df.empty else eps_df
if visible_eps is not None and not visible_eps.empty and visible_eps["Quarterly EPS"].notna().any():
    eps_fig = go.Figure()
    eps_fig.add_trace(
        go.Bar(
            x=visible_eps.index,
            y=visible_eps["Quarterly EPS"],
            name="Quarterly EPS",
            opacity=0.45,
        )
    )
    if visible_eps["TTM EPS"].notna().any():
        eps_fig.add_trace(
            go.Scatter(
                x=visible_eps.index,
                y=visible_eps["TTM EPS"],
                mode="lines+markers",
                name="TTM EPS",
            )
        )
    eps_fig.add_hline(y=0, line_width=1, line_dash="dot")
    eps_fig.update_layout(
        title=f"6. {level3_ticker} EPS — Reported Quarterly / TTM",
        height=330,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="EPS",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(eps_fig, use_container_width=True)
    st.caption("EPS口径：Yahoo reported quarterly EPS；TTM EPS = 最近4个季度 reported EPS 之和。历史不足时不回填。")
else:
    st.caption(f"6. {level3_ticker}：所选时间范围内暂无可用的 EPS 历史数据。")

with st.expander("三级图表数据说明"):
    st.markdown(
        """
- **股价 / 成交量 / MACD**：Yahoo Finance 日频历史数据；MACD 参数为标准 12 / 26 / 9。
- **MA10 / MA30 / MA120**：基于日收盘价计算；先在完整历史上计算，再按所选时间范围裁剪，避免窗口起点均线失真。
- **Forward P/E / Price/Sales**：Yahoo Finance valuation measures 的月度历史序列；部分新股或个别证券可能没有完整历史。
- **EPS**：reported quarterly EPS，并额外计算滚动四季度 TTM EPS。
- **时间选择**：YTD、1M、3M、6M、1Y、3Y、5Y、MAX会同时作用于上述所有图。若股票尚未上市或对应基本面历史不足，只显示实际可用数据，不做跨证券回填。
        """
    )


st.markdown(
    f"""
<div class="small-note">
板块指数基准：<b>{INDEX_BASE_DATE.strftime('%Y-%m-%d')} = {INDEX_BASE_VALUE:.0f}</b>。
新上市成分股在拥有前一交易日有效价格后开始参与指数计算，不在上市前回填；SKHY只使用SKHY US自身历史。
所有收益率均使用该证券自身的历史调整价计算；历史不足则显示“—”。
</div>
""",
    unsafe_allow_html=True,
)

with st.expander("数据源、指数方法与刷新规则"):
    st.markdown(
        f"""
- **指数基准**：{INDEX_BASE_DATE.strftime('%Y-%m-%d')} 收盘 = {INDEX_BASE_VALUE:.0f}。
- **指数方法**：链式市值加权研究指数；后上市股票从具备前一交易日价格后纳入。
- **实时行情**：不自动刷新。仅在美股正常交易时段点击 **“刷新实时行情”** 时调用 Twelve Data，而且只刷新当前选中的板块。一级“当日”与二级 Daily Return 使用同一批实时数据。
- **实时数据保存范围**：本次浏览器会话内保留最近一次手动刷新结果；重新启动 App 后重新获取。
- **收盘后**：一级“当日”使用板块成分股 regular-session close 的收盘变动，二级主价格与 Daily Return 同样使用 regular-session close；不把盘后价作为主价格。
- **历史收益率**：YTD、1M、3M、6M、1Y、3Y、5Y均由该 ticker 自身历史计算；历史不足显示“—”。
- **SKHY**：只使用 SKHY US，自上市前不使用 000660.KS 回填。
- **成交量比**：当日累计/收盘成交量 ÷ 最近约63个交易日平均成交量。
- **Forward P/E / Market Cap**：Yahoo Finance 低频缓存。
- **颜色**：正收益绿色、负收益红色；Vol / 3M Avg ≥ 1.5x 为橙色。
        """
    )
