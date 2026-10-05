from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from config import ALL_TICKERS, SECTORS
from data_layer import (
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
    twelve_data_quotes,
    yahoo_latest,
)

load_dotenv()

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Equity Dashboard",
    page_icon="◈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(127,127,127,.06);
        border: 1px solid rgba(127,127,127,.18);
        padding: 12px;
        border-radius: 12px;
    }

    .small-note {
        font-size: .82rem;
        opacity: .72;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DISPLAY HELPERS
# =========================================================

POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
ZERO_COLOR = "#6b7280"
VOLUME_ALERT_COLOR = "#d97706"


def fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.2%}"


def fmt_num(x, decimals=2):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def fmt_price(x):
    if x is None or pd.isna(x):
        return "—"
    return f"${x:,.2f}"


def fmt_compact(x, dollar=False):
    if x is None or pd.isna(x):
        return "—"

    x = float(x)
    prefix = "$" if dollar else ""

    for unit, div in [
        ("T", 1e12),
        ("B", 1e9),
        ("M", 1e6),
        ("K", 1e3),
    ]:
        if abs(x) >= div:
            return f"{prefix}{x / div:.2f}{unit}"

    return f"{prefix}{x:,.0f}"


def fmt_ratio(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}x"


def fmt_pe(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.1f}x"


def return_color(value):
    if value is None or pd.isna(value):
        return ZERO_COLOR

    try:
        value = float(value)
    except Exception:
        return ZERO_COLOR

    if value > 0:
        return POSITIVE_COLOR
    if value < 0:
        return NEGATIVE_COLOR
    return ZERO_COLOR


# =========================================================
# TABLE STYLING
# =========================================================

def style_sector_table(display_df, raw_df):
    """
    一级板块表：
    - 板块名称左对齐
    - 所有数字右对齐
    - 收益率正绿负红
    """

    styles = pd.DataFrame(
        "",
        index=display_df.index,
        columns=display_df.columns,
    )

    numeric_cols = [
        "指数",
        "当日",
        "YTD",
        "1M",
        "3M",
        "6M",
        "1Y",
        "总市值",
    ]

    return_cols = [
        "当日",
        "YTD",
        "1M",
        "3M",
        "6M",
        "1Y",
    ]

    for col in numeric_cols:
        if col in styles.columns:
            styles[col] = "text-align: right;"

    if "板块" in styles.columns:
        styles["板块"] = "text-align: left;"

    for col in return_cols:
        if col not in raw_df.columns:
            continue

        for i in raw_df.index:
            color = return_color(raw_df.loc[i, col])

            styles.loc[i, col] += (
                f" color: {color};"
                " font-weight: 600;"
            )

    styler = display_df.style.apply(
        lambda _: styles,
        axis=None,
    )

    # 表头对齐
    table_styles = []

    for i, col in enumerate(display_df.columns):
        alignment = "left" if col == "板块" else "right"

        table_styles.append(
            {
                "selector": f"th.col_heading.level0.col{i}",
                "props": [
                    ("text-align", alignment),
                ],
            }
        )

    styler = styler.set_table_styles(table_styles)

    return styler


def style_stock_table(display_df, raw_df):
    """
    二级个股表：
    - Ticker / Name / Source 左对齐
    - 其余数字右对齐
    - 收益率正绿负红
    - 成交量比 >1.5x 橙色
    """

    styles = pd.DataFrame(
        "",
        index=display_df.index,
        columns=display_df.columns,
    )

    text_cols = [
        "Ticker",
        "Name",
        "Source",
    ]

    numeric_cols = [
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
    ]

    return_cols = [
        "Daily Return",
        "YTD Return",
        "1M Return",
        "3M Return",
        "6M Return",
        "1Y Return",
        "3Y Return",
        "5Y Return",
    ]

    for col in text_cols:
        if col in styles.columns:
            styles[col] = "text-align: left;"

    for col in numeric_cols:
        if col in styles.columns:
            styles[col] = "text-align: right;"

    # 收益率颜色
    for col in return_cols:
        if col not in raw_df.columns:
            continue

        for i in raw_df.index:
            color = return_color(raw_df.loc[i, col])

            styles.loc[i, col] += (
                f" color: {color};"
                " font-weight: 600;"
            )

    # 成交量比颜色
    if "Vol / 3M Avg" in raw_df.columns:
        for i in raw_df.index:
            value = raw_df.loc[i, "Vol / 3M Avg"]

            if value is None or pd.isna(value):
                styles.loc[i, "Vol / 3M Avg"] += (
                    f" color: {ZERO_COLOR};"
                )

            elif value >= 1.5:
                styles.loc[i, "Vol / 3M Avg"] += (
                    f" color: {VOLUME_ALERT_COLOR};"
                    " font-weight: 700;"
                )

    styler = display_df.style.apply(
        lambda _: styles,
        axis=None,
    )

    # 表头对齐
    table_styles = []

    for i, col in enumerate(display_df.columns):
        alignment = "left" if col in text_cols else "right"

        table_styles.append(
            {
                "selector": f"th.col_heading.level0.col{i}",
                "props": [
                    ("text-align", alignment),
                ],
            }
        )

    styler = styler.set_table_styles(table_styles)

    return styler


# =========================================================
# MARKET STATUS
# =========================================================

state = market_state()

status_color = "🟢" if state.is_open else "⚪"

td_enabled = bool(
    os.getenv(
        "TWELVE_DATA_API_KEY",
        "",
    ).strip()
)


# =========================================================
# HEADER
# =========================================================

header_l, header_r = st.columns([3, 2])

with header_l:
    st.title("AI Equity Dashboard")

    st.caption(
        "14个AI产业链板块 · 市值加权指数 · 个股行情与估值"
    )

with header_r:
    st.markdown(
        f"""
**{status_color} Market Status: {state.label}**

ET {state.now_et.strftime('%Y-%m-%d %H:%M:%S')}

Realtime source: {
    'Twelve Data enabled'
    if td_enabled
    else 'Yahoo fallback'
}
"""
    )


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("加载历史行情与市值数据…"):

    # 使用10年历史，确保5Y Return有足够缓冲
    prices = download_history(
        tuple(ALL_TICKERS),
        period="10y",
    )

    caps = market_caps(
        tuple(ALL_TICKERS),
    )


if prices.empty:
    st.error(
        "暂时无法取得历史行情。请检查网络或稍后重试。"
    )
    st.stop()


# =========================================================
# LEVEL 1
# =========================================================

st.subheader("一级：AI产业链板块")

sector_rows = []
sector_series = {}


for sector, members in SECTORS.items():

    tickers = [
        ticker
        for ticker, _ in members
    ]

    idx = build_sector_index(
        prices,
        tickers,
        caps,
    )

    sector_series[sector] = idx

    latest = (
        float(idx.iloc[-1])
        if not idx.empty
        else None
    )

    daily = (
        idx.pct_change().iloc[-1]
        if len(idx) >= 2
        else None
    )

    total_cap = sum(
        caps.get(ticker, 0.0)
        for ticker in tickers
    )

    sector_rows.append(
        {
            "板块": sector,
            "指数": latest,
            "当日": daily,
            "YTD": pct_change_period(
                idx,
                ytd=True,
            ),
            "1M": pct_change_period(
                idx,
                months=1,
            ),
            "3M": pct_change_period(
                idx,
                months=3,
            ),
            "6M": pct_change_period(
                idx,
                months=6,
            ),
            "1Y": pct_change_period(
                idx,
                years=1,
            ),
            "总市值": total_cap,
        }
    )


sector_df = pd.DataFrame(
    sector_rows
)


# =========================================================
# FORMAT LEVEL 1
# =========================================================

show_sector = sector_df.copy()

show_sector["指数"] = (
    show_sector["指数"]
    .map(
        lambda x: fmt_num(
            x,
            1,
        )
    )
)

for col in [
    "当日",
    "YTD",
    "1M",
    "3M",
    "6M",
    "1Y",
]:

    show_sector[col] = (
        show_sector[col]
        .map(fmt_pct)
    )


show_sector["总市值"] = (
    show_sector["总市值"]
    .map(
        lambda x: fmt_compact(
            x,
            dollar=True,
        )
    )
)


sector_styler = style_sector_table(
    show_sector,
    sector_df,
)


st.dataframe(
    sector_styler,
    use_container_width=True,
    hide_index=True,
    height=535,
)


# =========================================================
# SECTOR SELECTOR
# =========================================================

selected_sector = st.selectbox(
    "选择板块查看二级个股",
    list(SECTORS.keys()),
    index=0,
)


selected_members = SECTORS[
    selected_sector
]

selected_tickers = [
    ticker
    for ticker, _ in selected_members
]


# =========================================================
# SECTOR CHART
# =========================================================

idx = sector_series[
    selected_sector
]


if not idx.empty:

    chart_df = (
        idx
        .rename("Index")
        .reset_index()
    )

    chart_df.columns = [
        "Date",
        "Index",
    ]

    fig = px.line(
        chart_df,
        x="Date",
        y="Index",
        title=(
            f"{selected_sector} "
            "— 市值加权指数（Base=1000）"
        ),
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =========================================================
# LEVEL 2
# =========================================================

st.subheader(
    "二级：个股明细"
)


volumes = download_volume_history(
    tuple(selected_tickers),
    period="6mo",
)


fundamentals = fundamentals_for(
    tuple(selected_tickers)
)


yahoo_q = yahoo_latest(
    tuple(selected_tickers)
)


# Twelve Data只在正常交易时段使用
td_q = (
    twelve_data_quotes(
        tuple(selected_tickers)
    )
    if state.is_open
    else {}
)


sector_cap = sum(
    caps.get(ticker, 0.0)
    for ticker in selected_tickers
    if caps.get(
        ticker,
        0.0,
    ) > 0
)


rows = []


for ticker, configured_name in selected_members:

    hist_last, hist_prev = (
        latest_regular_close(
            prices,
            ticker,
        )
    )

    yq = yahoo_q.get(
        ticker,
        {},
    )

    td = td_q.get(
        ticker,
        {},
    )

    # -----------------------------------------------------
    # 实时交易状态
    # -----------------------------------------------------

    if state.is_open and td:

        try:
            latest_price = (
                float(td.get("close"))
                if td.get("close")
                is not None
                else None
            )
        except Exception:
            latest_price = None

        try:
            prev_close = (
                float(
                    td.get(
                        "previous_close"
                    )
                )
                if td.get(
                    "previous_close"
                )
                is not None
                else hist_prev
            )
        except Exception:
            prev_close = hist_prev

        try:
            volume = (
                float(
                    td.get("volume")
                )
                if td.get(
                    "volume"
                )
                is not None
                else None
            )
        except Exception:
            volume = None

        source = "Twelve Data"

    elif state.is_open:

        latest_price = (
            yq.get("price")
            or hist_last
        )

        prev_close = (
            yq.get(
                "previous_close"
            )
            or hist_prev
        )

        volume = yq.get(
            "volume"
        )

        source = "Yahoo"

    else:

        # 收盘后强制使用正常交易时段收盘价
        latest_price = (
            yq.get(
                "regular_close"
            )
            or hist_last
        )

        prev_close = hist_prev

        volume = yq.get(
            "volume"
        )

        source = "Close"


    # -----------------------------------------------------
    # DAILY RETURN
    # -----------------------------------------------------

    daily_ret = None

    if (
        latest_price is not None
        and prev_close
        not in (
            None,
            0,
        )
    ):
        daily_ret = (
            latest_price
            / prev_close
            - 1.0
        )


    # -----------------------------------------------------
    # VOLUME RATIO
    # -----------------------------------------------------

    avg_vol = avg_3m_volume(
        volumes,
        ticker,
    )

    vol_ratio = (
        volume / avg_vol
        if volume is not None
        and avg_vol
        not in (
            None,
            0,
        )
        else None
    )


    # -----------------------------------------------------
    # HISTORICAL RETURNS
    # -----------------------------------------------------

    if ticker in prices.columns:
        price_series = (
            prices[ticker]
            .dropna()
        )
    else:
        price_series = pd.Series(
            dtype=float
        )


    fp = (
        fundamentals
        .get(
            ticker,
            {},
        )
        .get(
            "forwardPE"
        )
    )


    name = (
        fundamentals
        .get(
            ticker,
            {},
        )
        .get(
            "longName"
        )
        or configured_name
    )


    cap = caps.get(
        ticker
    )


    weight = (
        cap / sector_cap
        if cap
        and sector_cap
        else None
    )


    rows.append(
        {
            "Ticker": ticker,
            "Name": name,
            "Price": latest_price,
            "Daily Return": daily_ret,
            "Volume": volume,
            "Vol / 3M Avg": vol_ratio,
            "Forward P/E": fp,

            "YTD Return": (
                return_since(
                    price_series,
                    latest_price,
                    ytd=True,
                )
                if latest_price
                else None
            ),

            "1M Return": (
                return_since(
                    price_series,
                    latest_price,
                    months=1,
                )
                if latest_price
                else None
            ),

            "3M Return": (
                return_since(
                    price_series,
                    latest_price,
                    months=3,
                )
                if latest_price
                else None
            ),

            "6M Return": (
                return_since(
                    price_series,
                    latest_price,
                    months=6,
                )
                if latest_price
                else None
            ),

            "1Y Return": (
                return_since(
                    price_series,
                    latest_price,
                    years=1,
                )
                if latest_price
                else None
            ),

            "3Y Return": (
                return_since(
                    price_series,
                    latest_price,
                    years=3,
                )
                if latest_price
                else None
            ),

            "5Y Return": (
                return_since(
                    price_series,
                    latest_price,
                    years=5,
                )
                if latest_price
                else None
            ),

            "Market Cap": cap,
            "Index Weight": weight,
            "Source": source,
        }
    )


raw_df = pd.DataFrame(
    rows
)


# =========================================================
# FORMAT LEVEL 2
# =========================================================

display_df = raw_df.copy()


display_df["Price"] = (
    display_df["Price"]
    .map(fmt_price)
)


return_columns = [
    "Daily Return",
    "YTD Return",
    "1M Return",
    "3M Return",
    "6M Return",
    "1Y Return",
    "3Y Return",
    "5Y Return",
    "Index Weight",
]


for col in return_columns:

    display_df[col] = (
        display_df[col]
        .map(fmt_pct)
    )


display_df["Volume"] = (
    display_df["Volume"]
    .map(fmt_compact)
)


display_df["Vol / 3M Avg"] = (
    display_df[
        "Vol / 3M Avg"
    ]
    .map(fmt_ratio)
)


display_df["Forward P/E"] = (
    display_df[
        "Forward P/E"
    ]
    .map(fmt_pe)
)


display_df["Market Cap"] = (
    display_df[
        "Market Cap"
    ]
    .map(
        lambda x: fmt_compact(
            x,
            dollar=True,
        )
    )
)


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


display_df = display_df[
    preferred_order
]


raw_df = raw_df[
    preferred_order
]


stock_styler = style_stock_table(
    display_df,
    raw_df,
)


st.dataframe(
    stock_styler,
    use_container_width=True,
    hide_index=True,
    height=330,
)


# =========================================================
# FOOTNOTE
# =========================================================

st.markdown(
    """
<div class="small-note">

收益率均使用该证券自身的历史价格计算；
历史不足则显示“—”。

SKHY不会使用000660.KS回填历史数据。

板块指数按当前市值推导等效股数并利用历史调整价重建，
适合作为研究指数；
正式指数产品还需要维护 corporate-action divisor。

</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# DATA SOURCE NOTES
# =========================================================

with st.expander(
    "数据源与刷新规则"
):

    st.markdown(
        """
- **交易时段**：若配置 Twelve Data API key，当前选中板块优先使用 Twelve Data `/quote`；失败自动回退 Yahoo。
- **收盘后**：显示最近一个 regular session 的收盘价，不把盘后成交价作为主价格。
- **历史收益率**：由各 ticker 自身历史 adjusted price 计算 YTD、1M、3M、6M、1Y、3Y、5Y。
- **SKHY**：仅使用 SKHY US 自身历史，上市之前的数据不回填。
- **成交量比**：当日累计/收盘成交量 ÷ 最近约63个交易日平均成交量。
- **Forward P/E**：Yahoo Finance fundamentals，低频缓存。
- **Market Cap**：Yahoo Finance，低频缓存。
- **绿色**：正收益。
- **红色**：负收益。
- **橙色**：成交量超过3个月平均成交量的1.5倍。
        """
    )