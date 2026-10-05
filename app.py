from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
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
    daily = float(idx.pct_change(fill_method=None).iloc[-1]) if len(idx) >= 2 else None
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

selected_sector = st.selectbox(
    "选择板块查看二级个股",
    list(SECTORS.keys()),
    index=0,
)
selected_members = SECTORS[selected_sector]
selected_tickers = [ticker for ticker, _ in selected_members]


# -----------------------------------------------------------------------------
# Manual Twelve Data refresh
# -----------------------------------------------------------------------------
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
        st.caption("Twelve Data API key 未配置；当前使用 Yahoo / 收盘数据。")
    elif not state.is_open:
        st.caption("美股当前已收盘：主价格固定显示 regular-session close，不调用实时行情。")
    elif last_refresh:
        st.caption(f"当前板块最近一次 Twelve Data 手动刷新：{last_refresh} ET")
    else:
        st.caption("美股交易时段：只有点击“刷新实时行情”才调用 Twelve Data。")

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
        st.success(f"已刷新 {len(quotes)} / {len(selected_tickers)} 只股票。")
    else:
        st.warning("Twelve Data 未返回有效行情；本页继续使用 Yahoo / 收盘数据。")

# Only use stored live data when the market is open.
td_q = (
    st.session_state.live_quotes.get(selected_sector, {})
    if state.is_open
    else {}
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

if weighted_live_return_den > 0:
    live_sector_return = weighted_live_return_num / weighted_live_return_den
    st.metric(
        "当前板块实时市值加权涨跌",
        fmt_pct(live_sector_return),
        help="仅根据本次 Twelve Data 手动刷新成功返回的当前板块股票计算。",
    )

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
- **实时行情**：不自动刷新。仅在美股正常交易时段点击 **“刷新实时行情”** 时调用 Twelve Data，而且只刷新当前选中的板块。
- **实时数据保存范围**：本次浏览器会话内保留最近一次手动刷新结果；重新启动 App 后重新获取。
- **收盘后**：主价格显示 regular-session close，不调用 Twelve Data，不把盘后价作为主价格。
- **历史收益率**：YTD、1M、3M、6M、1Y、3Y、5Y均由该 ticker 自身历史计算；历史不足显示“—”。
- **SKHY**：只使用 SKHY US，自上市前不使用 000660.KS 回填。
- **成交量比**：当日累计/收盘成交量 ÷ 最近约63个交易日平均成交量。
- **Forward P/E / Market Cap**：Yahoo Finance 低频缓存。
- **颜色**：正收益绿色、负收益红色；Vol / 3M Avg ≥ 1.5x 为橙色。
        """
    )
