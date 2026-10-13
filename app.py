from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from config import ALL_TICKERS, SECTORS, NAME_MAP
from event_engine import get_market_events, auto_event_metadata
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
    eps_revision_trend,
    company_event_snapshot,
    company_news_feed,
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
    .level3-chart-title {font-size:1.02rem; font-weight:700; margin:1.35rem 0 .35rem 0;}
    .level3-divider {height:1px; background:rgba(127,127,127,.16); margin:.45rem 0 1.1rem 0;}
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

if "live_quotes_all" not in st.session_state:
    st.session_state.live_quotes_all = {}
if "live_benchmark_quotes" not in st.session_state:
    st.session_state.live_benchmark_quotes = {}
if "live_updated_at_global" not in st.session_state:
    st.session_state.live_updated_at_global = None

MARKET_OVERVIEW = {
    "道琼斯工业指数": {"yahoo": "^DJI", "twelve": "DJI", "live_mode": "us"},
    "标普500指数": {"yahoo": "^GSPC", "twelve": "SPX", "live_mode": "us"},
    "纳斯达克综合指数": {"yahoo": "^IXIC", "twelve": "IXIC", "live_mode": "us"},
    "纳斯达克100": {"yahoo": "^NDX", "twelve": "NDX", "live_mode": "us"},
    "iShares Russell 1000 Value ETF（IWD）": {"yahoo": "IWD", "twelve": "IWD", "live_mode": "us"},
    "iShares Russell 1000 Growth ETF（IWF）": {"yahoo": "IWF", "twelve": "IWF", "live_mode": "us"},
    "NASDAQ Biotechnology Index": {"yahoo": "^NBI", "twelve": "NBI", "live_mode": "us"},
    "比特币": {"yahoo": "BTC-USD", "twelve": "BTC/USD", "live_mode": "always"},
    "黄金": {"yahoo": "GC=F", "twelve": "XAU/USD", "live_mode": "weekday_24h"},
    "石油（Brent）": {"yahoo": "BZ=F", "twelve": "XBR/USD", "live_mode": "weekday_24h"},
    "铜": {"yahoo": "HG=F", "twelve": "HG1", "live_mode": "weekday_24h"},
    "美元指数": {"yahoo": "DX-Y.NYB", "twelve": "DXY", "live_mode": "weekday_24h"},
}


def market_live_allowed(mode: str) -> bool:
    """Whether an instrument should use a freshly requested intraday quote now."""
    if mode == "always":
        return True
    if mode == "us":
        return state.is_open
    if mode == "weekday_24h":
        # FX/spot commodity feeds trade effectively around the clock Monday-Friday.
        return state.now_et.weekday() < 5
    if mode == "china":
        now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
        if now_cn.weekday() >= 5:
            return False
        t = now_cn.time()
        return (time(9, 30) <= t <= time(11, 30)) or (time(13, 0) <= t <= time(15, 0))
    return False

# -----------------------------------------------------------------------------
# Page header + top-level tabs
# -----------------------------------------------------------------------------
page_l, page_r = st.columns([3, 2])
with page_l:
    st.title("AI Equity Dashboard")
    st.caption("市场概览 · 15个AI产业链板块 · Breadth/Heatmap · 个股相对表现与估值 · 最新公司新闻")
with page_r:
    st.markdown(
        f"**{status_color} Market Status: {state.label}**  \n"
        f"ET {state.now_et.strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"Twelve Data: {'Ready' if td_enabled else 'Not configured'}"
    )

dashboard_tab, news_tab = st.tabs(["Dashboard", "Latest Company News"])

with dashboard_tab:
    refresh_col, refresh_note_col = st.columns([1.25, 4.75])
    with refresh_col:
        refresh_live = st.button(
            "刷新实时行情",
            type="primary",
            disabled=(not td_enabled),
            use_container_width=True,
        )
    with refresh_note_col:
        last_refresh = st.session_state.live_updated_at_global
        if not td_enabled:
            st.caption("Twelve Data API key 未配置；当前显示最近收盘数据。")
        elif last_refresh:
            st.caption(
                f"最近一次 Twelve Data 手动刷新：{last_refresh} ET。"
                "美股仅在正常交易时段采用实时价；BTC/商品/美元指数按各自交易状态更新。"
            )
        else:
            st.caption(
                "点击后刷新市场概览；美股交易时段同时刷新AI股票。"
                "收盘后的市场使用正式收盘价，Twelve Data不支持的标的自动回退 Yahoo。"
            )

    if refresh_live:
        try:
            twelve_data_quotes.clear()
        except Exception:
            pass

        with st.spinner("正在刷新市场与AI股票实时行情…"):
            # US equities only consume credits during the regular US session.
            stock_quotes = twelve_data_quotes(tuple(ALL_TICKERS)) if state.is_open else {}
            benchmark_symbols = tuple(
                v["twelve"] for v in MARKET_OVERVIEW.values() if v.get("twelve")
            )
            benchmark_live = twelve_data_quotes(benchmark_symbols)

        if stock_quotes:
            st.session_state.live_quotes_all = stock_quotes
        if benchmark_live:
            st.session_state.live_benchmark_quotes = benchmark_live

        if stock_quotes or benchmark_live:
            st.session_state.live_updated_at_global = datetime.now(state.now_et.tzinfo).strftime("%Y-%m-%d %H:%M:%S")
            st.success(
                f"实时刷新完成：AI股票 {len(stock_quotes)} / {len(ALL_TICKERS)}；"
                f"市场标的 {len(benchmark_live)} / {len([v for v in MARKET_OVERVIEW.values() if v.get('twelve')])}。"
            )
        else:
            st.warning("Twelve Data 未返回有效实时行情；本页继续使用最近 regular-session close 数据。")

    live_quotes_all = st.session_state.live_quotes_all if state.is_open else {}
    live_benchmark_quotes = st.session_state.live_benchmark_quotes


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
    # Helpers for consistent intraday daily-return logic
    # -----------------------------------------------------------------------------
    def live_sector_daily_return(
        tickers: list[str],
        quotes: dict[str, dict],
    ) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for ticker in tickers:
            td = quotes.get(ticker, {})
            cap = caps.get(ticker)
            if not td or not cap:
                continue
            _, hist_prev = latest_regular_close(prices, ticker)
            try:
                latest_price = float(td.get("close")) if td.get("close") is not None else None
            except Exception:
                latest_price = None
            try:
                prev_close = float(td.get("previous_close")) if td.get("previous_close") is not None else hist_prev
            except Exception:
                prev_close = hist_prev
            if latest_price is None or prev_close in (None, 0):
                continue
            stock_return = latest_price / float(prev_close) - 1.0
            numerator += float(cap) * stock_return
            denominator += float(cap)
        return numerator / denominator if denominator > 0 else None


    def latest_from_td(item: dict | None) -> tuple[float | None, float | None, float | None]:
        item = item or {}
        try:
            price = float(item.get("close")) if item.get("close") is not None else None
        except Exception:
            price = None
        try:
            prev = float(item.get("previous_close")) if item.get("previous_close") is not None else None
        except Exception:
            prev = None
        try:
            volume = float(item.get("volume")) if item.get("volume") is not None else None
        except Exception:
            volume = None
        return price, prev, volume


    # -----------------------------------------------------------------------------
    # 一级：市场概览
    # -----------------------------------------------------------------------------
    st.subheader("一级：市场概览")

    benchmark_yahoo_tickers = tuple(
        dict.fromkeys(v["yahoo"] for v in MARKET_OVERVIEW.values() if v.get("yahoo"))
    )
    with st.spinner("加载全球市场概览…"):
        benchmark_prices = download_history(benchmark_yahoo_tickers, period="10y")
        benchmark_quotes = yahoo_latest(benchmark_yahoo_tickers)

    market_rows: list[dict] = []
    for label, ids in MARKET_OVERVIEW.items():
        yt = ids.get("yahoo")
        td_symbol = ids.get("twelve")
        live_mode = ids.get("live_mode", "us")
        use_live = market_live_allowed(live_mode)

        if yt and yt in benchmark_prices.columns:
            hist = benchmark_prices[yt].dropna()
        else:
            hist = pd.Series(dtype=float)

        hist_last = float(hist.iloc[-1]) if not hist.empty else None
        hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else None
        yq = benchmark_quotes.get(yt, {}) if yt else {}

        td_item = live_benchmark_quotes.get(td_symbol, {}) if (use_live and td_symbol) else {}

        if td_item:
            level, prev_close, _ = latest_from_td(td_item)
            level = level or yq.get("price") or hist_last
            prev_close = prev_close or yq.get("previous_close") or hist_prev
        elif use_live:
            level = yq.get("price") or hist_last
            prev_close = yq.get("previous_close") or hist_prev
        else:
            level = yq.get("regular_close") or hist_last
            prev_close = hist_prev

        daily = (float(level) / float(prev_close) - 1.0) if level is not None and prev_close not in (None, 0) else None
        market_rows.append(
            {
                "市场": label,
                "点位/价格": level,
                "当日": daily,
                "YTD": return_since(hist, float(level), ytd=True) if level is not None and not hist.empty else None,
                "1M": return_since(hist, float(level), months=1) if level is not None and not hist.empty else None,
                "3M": return_since(hist, float(level), months=3) if level is not None and not hist.empty else None,
                "6M": return_since(hist, float(level), months=6) if level is not None and not hist.empty else None,
                "1Y": return_since(hist, float(level), years=1) if level is not None and not hist.empty else None,
                "3Y": return_since(hist, float(level), years=3) if level is not None and not hist.empty else None,
                "5Y": return_since(hist, float(level), years=5) if level is not None and not hist.empty else None,
            }
        )

    market_df = pd.DataFrame(market_rows)
    market_formatters = {
        "点位/价格": lambda x: fmt_num(x, 2),
        "当日": fmt_pct,
        "YTD": fmt_pct,
        "1M": fmt_pct,
        "3M": fmt_pct,
        "6M": fmt_pct,
        "1Y": fmt_pct,
        "3Y": fmt_pct,
        "5Y": fmt_pct,
    }
    market_styler = style_numeric_table(
        market_df,
        formatters=market_formatters,
        return_cols=["当日", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"],
    )
    st.dataframe(market_styler, use_container_width=True, hide_index=True, height=520)


    # -----------------------------------------------------------------------------
    # 二级：AI产业链板块总览
    # -----------------------------------------------------------------------------
    st.subheader("二级：AI产业链板块")

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

        if state.is_open and live_quotes_all:
            live_daily = live_sector_daily_return(tickers, live_quotes_all)
            if live_daily is not None:
                daily = live_daily
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
    sector_styler = style_numeric_table(
        sector_df,
        formatters=sector_formatters,
        return_cols=["当日", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"],
    )
    st.dataframe(sector_styler, use_container_width=True, hide_index=True, height=570)

    # AI sector heatmap: size = sector market cap, colour = selected return horizon.
    heatmap_left, heatmap_right = st.columns([1.6, 4.4])
    with heatmap_left:
        heatmap_period = st.radio(
            "AI Sector Heatmap",
            ["当日", "1M", "3M", "YTD"],
            index=0,
            horizontal=True,
            key="sector_heatmap_period",
        )
    heatmap_df = sector_df[["板块", "总市值", heatmap_period]].copy()
    heatmap_df = heatmap_df.dropna(subset=["总市值", heatmap_period])
    if not heatmap_df.empty:
        heatmap_df["ReturnPct"] = heatmap_df[heatmap_period] * 100.0
        heatmap_df["Label"] = heatmap_df["板块"].str.replace(r"^\d+\.\s*", "", regex=True)
        heatmap_fig = px.treemap(
            heatmap_df,
            path=["Label"],
            values="总市值",
            color="ReturnPct",
            color_continuous_scale=[
                (0.0, "#991b1b"),   # Larger losses: darker red.
                (0.25, "#f87171"),
                (0.5, "#f3f4f6"),   # Zero return: light neutral.
                (0.75, "#4ade80"),
                (1.0, "#166534"),   # Larger gains: darker green.
            ],
            color_continuous_midpoint=0,
            custom_data=["ReturnPct", "总市值"],
        )
        heatmap_fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%",
            hovertemplate="%{label}<br>Return: %{customdata[0]:+.2f}%<br>Market Cap: $%{customdata[1]:.3s}<extra></extra>",
        )
        heatmap_fig.update_layout(
            height=420,
            margin=dict(l=5, r=5, t=15, b=5),
            coloraxis_colorbar=dict(title=f"{heatmap_period} %"),
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, config={"displaylogo": False})
        st.caption("方块大小 = 板块总市值；颜色 = 所选期间回报。盘中手动刷新后，“当日”使用 Twelve Data 可获取的实时成分股收益计算。")


    # -----------------------------------------------------------------------------
    # 三级：板块详情
    # -----------------------------------------------------------------------------
    st.subheader("三级：板块详情")
    selected_sector = st.selectbox(
        "选择板块",
        list(SECTORS.keys()),
        index=0,
        key="sector_selector",
    )
    selected_members = SECTORS[selected_sector]
    selected_tickers = [ticker for ticker, _ in selected_members]

    idx = sector_series[selected_sector]
    market_events = get_market_events(include_auto=True, high_only=True)
    event_meta = auto_event_metadata()
    event_toggle_col, event_note_col = st.columns([1.45, 4.55])
    with event_toggle_col:
        show_market_events = st.toggle(
            "显示重要事件 / 区间",
            value=True,
            key="sector_event_overlay_toggle",
            help="一次性显示或隐藏人工确认与自动识别的宏观、政策、AI及市场重大事件。",
        )
    with event_note_col:
        auto_updated = event_meta.get("updated_at") or "尚未自动更新"
        st.caption(
            "浅色背景 = 重要时间区间；虚线 = 单日重要事件；关闭开关可统一隐藏全部标注。"
            f" 自动事件库：{event_meta.get('event_count', 0)}条，最近更新 {auto_updated}。"
        )

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
        fig.update_traces(line=dict(width=2.2), hovertemplate="%{x|%Y-%m-%d}<br>Index: %{y:,.1f}<extra></extra>")

        if show_market_events:
            chart_start = pd.Timestamp(chart_df["Date"].min()).tz_localize(None)
            chart_end = pd.Timestamp(chart_df["Date"].max()).tz_localize(None)
            point_events = []

            for event in market_events:
                if event.get("kind") == "range":
                    start = pd.Timestamp(event["start"])
                    end = pd.Timestamp(event["end"]) if event.get("end") else chart_end
                    if end < chart_start or start > chart_end:
                        continue
                    clipped_start = max(start, chart_start)
                    clipped_end = min(end, chart_end)
                    fig.add_vrect(
                        x0=clipped_start,
                        x1=clipped_end,
                        fillcolor=event.get("color", "rgba(107,114,128,0.08)"),
                        line_width=1,
                        line_color=event.get("line_color", "rgba(107,114,128,0.25)"),
                        layer="below",
                    )
                    mid = clipped_start + (clipped_end - clipped_start) / 2
                    fig.add_annotation(
                        x=mid, y=0.98, xref="x", yref="paper",
                        text=event["label"], showarrow=False,
                        font=dict(size=10, color="rgba(75,85,99,0.88)"),
                        bgcolor="rgba(255,255,255,0.58)",
                        borderpad=2,
                    )
                else:
                    event_date = pd.Timestamp(event["date"])
                    if event_date < chart_start or event_date > chart_end:
                        continue
                    fig.add_vline(
                        x=event_date,
                        line_width=1.2,
                        line_dash="dot",
                        line_color=event.get("color", "rgba(75,85,99,0.42)"),
                        layer="below",
                    )
                    point_events.append(event)

            # Alternate label heights so nearby point events do not overlap.
            for n, event in enumerate(point_events):
                fig.add_annotation(
                    x=pd.Timestamp(event["date"]),
                    y=1.025 + (n % 2) * 0.065,
                    xref="x", yref="paper",
                    text=event["label"],
                    showarrow=False,
                    xanchor="center",
                    font=dict(size=10, color="rgba(55,65,81,0.92)"),
                    bgcolor="rgba(255,255,255,0.76)",
                    bordercolor="rgba(156,163,175,0.20)",
                    borderwidth=1,
                    borderpad=2,
                )

        fig.update_layout(
            height=410,
            margin=dict(l=10, r=10, t=82 if show_market_events else 52, b=10),
            showlegend=False,
            yaxis_title="Index",
            xaxis_title="Date",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        if show_market_events:
            with st.expander("重要事件标注说明"):
                event_rows = []
                for event in market_events:
                    if event.get("kind") == "range":
                        when = f"{event['start']} → {event['end'] or '至今'}"
                    else:
                        when = event["date"]
                    event_rows.append({
                        "时间": when,
                        "类别": event.get("category", ""),
                        "事件": event["label"],
                        "重要性": event.get("importance", "High"),
                        "模式": "自动" if event.get("auto_generated") else "人工确认",
                        "来源": event.get("source", ""),
                    })
                st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)
                if event_meta.get("scheduled_count"):
                    st.caption(
                        f"自动事件引擎另外跟踪 {event_meta.get('scheduled_count')} 个未来FOMC/CPI日程；"
                        "为避免历史图过密，日程本身不直接画入指数图，只有实际形成高重要性市场事件后才进入标注。"
                    )

    volumes = download_volume_history(tuple(selected_tickers), period="6mo")
    fundamentals = fundamentals_for(tuple(selected_tickers))
    yahoo_q = yahoo_latest(tuple(selected_tickers))
    sector_cap = sum(caps.get(ticker, 0.0) for ticker in selected_tickers if caps.get(ticker, 0.0) > 0)

    rows: list[dict] = []
    for ticker, configured_name in selected_members:
        hist_last, hist_prev = latest_regular_close(prices, ticker)
        yq = yahoo_q.get(ticker, {})
        td = live_quotes_all.get(ticker, {}) if state.is_open else {}

        if state.is_open and td:
            latest_price, prev_close, volume = latest_from_td(td)
            latest_price = latest_price or yq.get("price") or hist_last
            prev_close = prev_close or yq.get("previous_close") or hist_prev
            volume = volume if volume is not None else yq.get("volume")
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

        daily_ret = (float(latest_price) / float(prev_close) - 1.0) if latest_price is not None and prev_close not in (None, 0) else None
        avg_vol = avg_3m_volume(volumes, ticker)
        vol_ratio = float(volume) / float(avg_vol) if volume is not None and avg_vol not in (None, 0) else None
        price_series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)
        fp = fundamentals.get(ticker, {}).get("forwardPE")
        name = fundamentals.get(ticker, {}).get("longName") or configured_name
        cap = caps.get(ticker)
        weight = float(cap) / float(sector_cap) if cap and sector_cap else None

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
        "Ticker", "Name", "Price", "Daily Return", "Volume", "Vol / 3M Avg", "Forward P/E",
        "YTD Return", "1M Return", "3M Return", "6M Return", "1Y Return", "3Y Return", "5Y Return",
        "Market Cap", "Index Weight", "Source",
    ]
    stock_df = stock_df[preferred_order]

    # Sector breadth: price participation and equal-weight vs cap-weight daily move.
    valid_daily = pd.to_numeric(stock_df["Daily Return"], errors="coerce").dropna()
    advancers = int((valid_daily > 0).sum())
    decliners = int((valid_daily < 0).sum())
    unchanged = int((valid_daily == 0).sum())
    equal_weight_daily = float(valid_daily.mean()) if not valid_daily.empty else None
    sector_match = sector_df.loc[sector_df["板块"] == selected_sector, "当日"]
    cap_weight_daily = float(sector_match.iloc[0]) if not sector_match.empty and pd.notna(sector_match.iloc[0]) else None

    latest_price_map = {
        str(row["Ticker"]): row["Price"]
        for _, row in stock_df[["Ticker", "Price"]].iterrows()
    }

    def pct_above_ma(window: int) -> float | None:
        flags = []
        for ticker in selected_tickers:
            if ticker not in prices.columns:
                continue
            series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
            if len(series) < window:
                continue
            ma = float(series.tail(window).mean())
            px_now = latest_price_map.get(ticker)
            try:
                px_now = float(px_now)
            except Exception:
                px_now = float(series.iloc[-1]) if not series.empty else None
            if px_now is not None and np.isfinite(px_now) and ma > 0:
                flags.append(px_now > ma)
        return float(np.mean(flags)) if flags else None

    st.markdown("**板块 Breadth**")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("上涨 / 下跌", f"{advancers} / {decliners}" + (f" / {unchanged}" if unchanged else ""))
    b2.metric("等权当日", fmt_pct(equal_weight_daily))
    b3.metric("市值加权当日", fmt_pct(cap_weight_daily))
    b4.metric("Above MA20", fmt_pct(pct_above_ma(20)))
    b5.metric("Above MA50", fmt_pct(pct_above_ma(50)))
    b6.metric("Above MA200", fmt_pct(pct_above_ma(200)))
    st.caption("Above MA 指当前价格高于对应日均线的成分股占比；等权与市值加权差异可用于判断涨跌是否集中在少数大市值龙头。")

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
    stock_styler = style_numeric_table(
        stock_df,
        formatters=stock_formatters,
        return_cols=["Daily Return", "YTD Return", "1M Return", "3M Return", "6M Return", "1Y Return", "3Y Return", "5Y Return"],
        volume_col="Vol / 3M Avg",
    )
    st.dataframe(stock_styler, use_container_width=True, hide_index=True, height=330)


    # -----------------------------------------------------------------------------
    # Level 4 single-stock analytics
    # -----------------------------------------------------------------------------
    st.subheader("四级：个股指标图")

    # Every stock in the 15-sector universe appears once in this selector.
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




    def style_level3_chart(
        fig: go.Figure,
        *,
        height: int,
        yaxis_title: str,
        showlegend: bool = False,
        legend_y: float = 1.055,
    ) -> go.Figure:
        """Apply one consistent visual system to every Level-3 chart."""
        fig.update_layout(
            height=height,
            margin=dict(l=18, r=18, t=48 if showlegend else 22, b=38),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            yaxis_title=yaxis_title,
            xaxis_title=None,
            showlegend=showlegend,
            font=dict(size=12),
            legend=(
                dict(
                    orientation="h",
                    yanchor="bottom",
                    y=legend_y,
                    xanchor="left",
                    x=0,
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                )
                if showlegend
                else None
            ),
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="rgba(127,127,127,.22)",
            ticks="outside",
            ticklen=4,
            rangeslider_visible=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(127,127,127,.14)",
            zeroline=False,
            showline=False,
        )
        return fig


    def level3_heading(number: int, text: str) -> None:
        st.markdown(
            f'<div class="level3-chart-title">{number}. {text}</div>',
            unsafe_allow_html=True,
        )


    def level3_divider() -> None:
        st.markdown('<div class="level3-divider"></div>', unsafe_allow_html=True)


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
        eps_revision_df = eps_revision_trend(level3_ticker)
        event_snapshot = company_event_snapshot(level3_ticker)

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
        level3_heading(1, f"{level3_ticker} 股价")
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
        style_level3_chart(price_fig, height=470, yaxis_title="Price", showlegend=True)
        price_fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(price_fig, use_container_width=True, config={"displaylogo": False})
        level3_divider()

        # 2. Volume
        level3_heading(2, f"{level3_ticker} 成交量")
        volume_fig = go.Figure(
            data=[go.Bar(x=visible.index, y=visible["Volume"], name="Volume")]
        )
        style_level3_chart(volume_fig, height=260, yaxis_title="Volume", showlegend=False)
        st.plotly_chart(volume_fig, use_container_width=True, config={"displaylogo": False})
        level3_divider()

        # 3. MACD (12, 26, 9)
        level3_heading(3, f"{level3_ticker} MACD (12, 26, 9)")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Bar(x=visible.index, y=visible["Histogram"], name="Histogram"))
        macd_fig.add_trace(go.Scatter(x=visible.index, y=visible["MACD"], mode="lines", name="MACD"))
        macd_fig.add_trace(go.Scatter(x=visible.index, y=visible["Signal"], mode="lines", name="Signal"))
        macd_fig.add_hline(y=0, line_width=1, line_dash="dot")
        style_level3_chart(macd_fig, height=300, yaxis_title="MACD", showlegend=True)
        st.plotly_chart(macd_fig, use_container_width=True, config={"displaylogo": False})
        level3_divider()

    # 4. Relative performance: stock, S&P 500, Nasdaq 100 and its AI sector in one chart.
    level3_heading(4, f"{level3_ticker} 相对表现")
    stock_rel = prices[level3_ticker].dropna().rename(level3_ticker) if level3_ticker in prices.columns else pd.Series(dtype=float)
    spx_rel = benchmark_prices["^GSPC"].dropna().rename("S&P 500") if "^GSPC" in benchmark_prices.columns else pd.Series(dtype=float)
    ndx_rel = benchmark_prices["^NDX"].dropna().rename("Nasdaq 100") if "^NDX" in benchmark_prices.columns else pd.Series(dtype=float)
    sector_rel = sector_series.get(sector_name, pd.Series(dtype=float)).dropna().rename("所属AI板块")

    rel_parts = [x for x in [stock_rel, spx_rel, ndx_rel, sector_rel] if x is not None and not x.empty]
    if len(rel_parts) >= 2:
        rel_df = pd.concat(rel_parts, axis=1).sort_index()
        last_date = rel_df.index.max()
        rel_start = range_start(last_date, selected_range)
        if rel_start is not None:
            rel_df = rel_df.loc[rel_df.index >= rel_start]
        # Only compare from the first date where every requested series exists.
        rel_df = rel_df.ffill().dropna(how="any")
        if not rel_df.empty:
            base_vals = rel_df.iloc[0].replace(0, np.nan)
            rel_norm = rel_df.div(base_vals).mul(100.0).dropna(how="all")
            rel_fig = go.Figure()
            for col in rel_norm.columns:
                width = 2.6 if col == level3_ticker else 1.8
                rel_fig.add_trace(go.Scatter(x=rel_norm.index, y=rel_norm[col], mode="lines", name=col, line=dict(width=width)))
            rel_fig.add_hline(y=100, line_width=1, line_dash="dot")
            style_level3_chart(rel_fig, height=330, yaxis_title="Normalized = 100", showlegend=True)
            st.plotly_chart(rel_fig, use_container_width=True, config={"displaylogo": False})
            st.caption("所有序列在所选期间的首个共同交易日归一化为100；可以直接观察个股相对 S&P 500、Nasdaq 100 和所属AI板块的 Alpha/相对强弱。")
        else:
            st.caption("所选区间没有足够的共同历史数据用于相对表现比较。")
    else:
        st.caption("暂时没有足够的基准历史数据用于相对表现比较。")
    level3_divider()

    # 5 & 6. Daily price-driven valuation history. Yahoo provides periodic
    # valuation snapshots; data_layer.py reconstructs a daily series by holding
    # the latest implied forward-EPS / sales-per-share denominator between snapshots.
    visible_val = clip_to_range(valuations, selected_range) if not valuations.empty else valuations

    level3_heading(5, f"{level3_ticker} Forward P/E")
    if visible_val is not None and not visible_val.empty and "Forward P/E" in visible_val.columns and visible_val["Forward P/E"].notna().any():
        pe_data = visible_val["Forward P/E"].dropna()
        pe_fig = go.Figure(go.Scatter(x=pe_data.index, y=pe_data.values, mode="lines", name="Forward P/E"))
        style_level3_chart(pe_fig, height=285, yaxis_title="Forward P/E (x)", showlegend=False)
        st.plotly_chart(pe_fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.caption(f"{level3_ticker}：所选时间范围内暂无可用的 Forward P/E 历史数据。")
    level3_divider()

    level3_heading(6, f"{level3_ticker} Price / Sales")
    if visible_val is not None and not visible_val.empty and "Price/Sales" in visible_val.columns and visible_val["Price/Sales"].notna().any():
        ps_data = visible_val["Price/Sales"].dropna()
        ps_fig = go.Figure(go.Scatter(x=ps_data.index, y=ps_data.values, mode="lines", name="Price/Sales"))
        style_level3_chart(ps_fig, height=285, yaxis_title="Price / Sales (x)", showlegend=False)
        st.plotly_chart(ps_fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.caption(f"{level3_ticker}：所选时间范围内暂无可用的 Price/Sales 历史数据。")
    level3_divider()

    # 6. EPS. Use reported quarterly EPS; TTM EPS is the rolling sum of four
    # reported quarters, which is a useful common earnings-per-share trend measure.
    visible_eps = clip_to_range(eps_df, selected_range) if not eps_df.empty else eps_df
    level3_heading(7, f"{level3_ticker} EPS — Reported Quarterly / TTM")
    if visible_eps is not None and not visible_eps.empty and visible_eps["Quarterly EPS"].notna().any():
        eps_fig = go.Figure()
        quarterly_text = [f"{v:.2f}" if pd.notna(v) else "" for v in visible_eps["Quarterly EPS"]]
        eps_fig.add_trace(
            go.Bar(
                x=visible_eps.index,
                y=visible_eps["Quarterly EPS"],
                name="Quarterly EPS",
                opacity=0.45,
                text=quarterly_text,
                textposition="outside",
                cliponaxis=False,
            )
        )
        if visible_eps["TTM EPS"].notna().any():
            ttm = visible_eps["TTM EPS"]
            ttm_text = [f"{v:.2f}" if pd.notna(v) else "" for v in ttm]
            eps_fig.add_trace(
                go.Scatter(
                    x=visible_eps.index,
                    y=ttm,
                    mode="lines+markers+text",
                    text=ttm_text,
                    textposition="top center",
                    name="TTM EPS",
                )
            )
        eps_fig.add_hline(y=0, line_width=1, line_dash="dot")
        style_level3_chart(eps_fig, height=330, yaxis_title="EPS", showlegend=True)
        st.plotly_chart(eps_fig, use_container_width=True, config={"displaylogo": False})
        st.caption("EPS口径：Yahoo reported quarterly EPS；TTM EPS = 最近4个季度 reported EPS 之和。历史不足时不回填。")
    else:
        st.caption(f"{level3_ticker}：所选时间范围内暂无可用的 EPS 历史数据。")

    level3_divider()
    level3_heading(8, f"{level3_ticker} EPS Consensus Revision")
    if eps_revision_df is not None and not eps_revision_df.empty and eps_revision_df.notna().any().any():
        rev_fig = go.Figure()
        for col in ["FY1 EPS", "FY2 EPS"]:
            if col in eps_revision_df.columns and eps_revision_df[col].notna().any():
                vals = eps_revision_df[col]
                rev_fig.add_trace(
                    go.Scatter(
                        x=eps_revision_df.index,
                        y=vals,
                        mode="lines+markers+text",
                        text=[f"{v:.2f}" if pd.notna(v) else "" for v in vals],
                        textposition="top center",
                        name=col,
                    )
                )
        style_level3_chart(rev_fig, height=310, yaxis_title="Consensus EPS", showlegend=True)
        st.plotly_chart(rev_fig, use_container_width=True, config={"displaylogo": False})
        st.caption("FY1 = 当前财年、FY2 = 下一财年；Yahoo/yfinance 提供 Current、7/30/60/90天前的分析师一致预期。上修/下修方向通常比单一 Forward P/E 更能反映基本面预期变化。")
    else:
        st.caption(f"{level3_ticker}：暂未获取到可用的分析师 EPS revision 数据。")

    level3_divider()
    level3_heading(9, f"{level3_ticker} 近期活动 / 下一阶段重要事件")

    recent_col, upcoming_col = st.columns(2, gap="large")

    with recent_col:
        st.markdown("**近期活动 / 公司动态**")
        recent_items = event_snapshot.get("recent", []) if event_snapshot else []
        if recent_items:
            for item in recent_items[:5]:
                date_txt = item.get("date") or "—"
                title = item.get("title") or "Untitled"
                source = item.get("source") or "Yahoo Finance"
                url = item.get("url")
                if url:
                    st.markdown(f"- **{date_txt}** — [{title}]({url})  \n  <span class='small-note'>{source}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"- **{date_txt}** — {title}  \n  <span class='small-note'>{source}</span>", unsafe_allow_html=True)
        else:
            st.caption("暂未获取到可用的近期活动 / 公司动态。")

    with upcoming_col:
        st.markdown("**下一阶段重要事件**")
        upcoming_items = event_snapshot.get("upcoming", []) if event_snapshot else []
        if upcoming_items:
            for item in upcoming_items[:6]:
                date_txt = item.get("date") or "待定"
                title = item.get("title") or "重要事件"
                detail = item.get("detail")
                url = item.get("url")
                if url:
                    line = f"- **{date_txt}** — [{title}]({url})"
                else:
                    line = f"- **{date_txt}** — {title}"
                if detail:
                    line += f"  \n  <span class='small-note'>{detail}</span>"
                st.markdown(line, unsafe_allow_html=True)
        else:
            st.caption("暂未获取到已公布的下一阶段重要事件。")

    st.caption("事件模块优先展示 Yahoo Finance 可获取的公司活动/新闻与公司日历；未公开或数据源未覆盖的活动不会推断补齐。")

    with st.expander("三级图表数据说明"):
        st.markdown(
            """
    - **股价 / 成交量 / MACD**：Yahoo Finance 日频历史数据；MACD 参数为标准 12 / 26 / 9。
    - **MA10 / MA30 / MA120**：基于日收盘价计算；先在完整历史上计算，再按所选时间范围裁剪，避免窗口起点均线失真。
    - **相对表现**：个股、S&P 500、Nasdaq 100、所属AI板块在所选区间首个共同交易日统一为100。
    - **Forward P/E / Price/Sales**：Yahoo Finance 提供周期性估值快照；网站用每个快照反推当时的 forward EPS / sales-per-share，并在下一次快照前用每日股价重算估值倍数，因此曲线会随股价逐日变化。
    - **EPS**：reported quarterly EPS，并额外计算滚动四季度 TTM EPS。
    - **EPS Revision**：Yahoo/yfinance analyst consensus 的 Current、7/30/60/90天前 FY1/FY2 EPS estimate。
    - **活动 / 重要事件**：Yahoo Finance 公司新闻、活动相关标题、earnings calendar 与可获取的公司日历；只展示可核验的公开事件。
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
    - **实时行情**：不自动刷新。顶部只有一个 **“刷新实时行情”** 按钮。市场概览会按各资产交易状态优先采用 Twelve Data；美股正常交易时段同时刷新全部AI股票。未获 Twelve Data 返回的标的自动回退 Yahoo。
    - **市场概览**：道琼斯、标普500、纳斯达克综合、纳斯达克100、IWD、IWF、NASDAQ Biotechnology Index、比特币、黄金、Brent、铜、美元指数；统一显示当日、YTD、1M、3M、6M、1Y、3Y、5Y。
    - **实时数据保存范围**：本次浏览器会话内保留最近一次手动刷新结果；重新启动 App 后重新获取。
    - **收盘后**：美股指数、AI板块和个股使用 regular-session close；商品/外汇/加密按各自市场状态决定是否使用最新行情。盘后美股价格不作为主价格。
    - **历史收益率**：YTD、1M、3M、6M、1Y、3Y、5Y均由该 ticker 自身历史计算；历史不足显示“—”。
    - **SKHY**：只使用 SKHY US，自上市前不使用 000660.KS 回填。
    - **成交量比**：当日累计/收盘成交量 ÷ 最近约63个交易日平均成交量。
    - **Forward P/E / Market Cap**：Yahoo Finance 低频缓存。
    - **颜色**：正收益绿色、负收益红色；Vol / 3M Avg ≥ 1.5x 为橙色。
            """
        )


with news_tab:
    st.subheader("公司最新重要新闻")
    st.caption(
        "覆盖上面15个板块内全部公司。新闻通过 Google News RSS 聚合公开网页标题，"
        "来源列显示原始发布媒体；默认按发布时间从新到旧。新增 Importance 与 Event Type，"
        "用于把财报/指引、产品、AI Capex、M&A、监管、评级等市场敏感事件优先筛出来。情绪标签仅作快速筛选，不代表投资观点。"
    )

    filter_a, filter_b, filter_c, filter_d, filter_e = st.columns([1.0, 1.0, 1.45, 1.0, 1.45])
    with filter_a:
        news_window = st.selectbox(
            "时间范围",
            ["24小时", "3天", "7天"],
            index=1,
            key="news_window",
        )
    with filter_b:
        news_importance_filter = st.selectbox(
            "重要性",
            ["高 + 中", "全部", "高", "中", "低"],
            index=0,
            key="news_importance_filter",
        )
    with filter_c:
        news_event_filter = st.selectbox(
            "Event Type",
            ["全部事件", "财报 / 指引", "产品 / 技术", "AI Capex / 数据中心", "M&A", "合作 / 订单", "监管 / 法律", "分析师评级", "管理层", "资本回报 / 融资", "宏观 / 贸易政策", "会议 / Investor Event", "其他"],
            index=0,
            key="news_event_filter",
        )
    with filter_d:
        news_sentiment = st.selectbox(
            "情绪",
            ["全部", "正面", "负面", "中性"],
            index=0,
            key="news_sentiment",
        )
    with filter_e:
        news_company = st.selectbox(
            "相关公司",
            ["全部公司"] + list(ALL_TICKERS),
            index=0,
            key="news_company",
        )

    refresh_news = st.button("刷新新闻", key="refresh_news_button")
    if refresh_news:
        try:
            company_news_feed.clear()
        except Exception:
            pass

    sector_news_input = tuple((sector, tuple(members)) for sector, members in SECTORS.items())
    with st.spinner("正在聚合公司相关新闻…"):
        news_df = company_news_feed(sector_news_input, lookback_days=7)

    if news_df is None or news_df.empty:
        st.info("暂未获取到可用的公司相关新闻。可以稍后点击“刷新新闻”重试。")
    else:
        filtered_news = news_df.copy()
        now_utc = pd.Timestamp.now(tz="UTC")
        window_days = {"24小时": 1, "3天": 3, "7天": 7}[news_window]
        cutoff = now_utc - pd.Timedelta(days=window_days)
        filtered_news = filtered_news[
            filtered_news["published"].isna() | (filtered_news["published"] >= cutoff)
        ]

        if news_importance_filter == "高 + 中":
            filtered_news = filtered_news[filtered_news["importance_label"].isin(["高", "中"])]
        elif news_importance_filter != "全部":
            filtered_news = filtered_news[filtered_news["importance_label"] == news_importance_filter]

        if news_event_filter != "全部事件":
            filtered_news = filtered_news[filtered_news["event_type"] == news_event_filter]

        if news_sentiment != "全部":
            filtered_news = filtered_news[filtered_news["sentiment"] == news_sentiment]

        if news_company != "全部公司":
            filtered_news = filtered_news[
                filtered_news["companies"].fillna("").str.split(", ").apply(
                    lambda xs: news_company in xs
                )
            ]

        filtered_news = filtered_news.sort_values(
            ["published", "importance"],
            ascending=[False, False],
            na_position="last",
        ).head(150)

        if filtered_news.empty:
            st.info("当前筛选条件下没有新闻。")
        else:
            display_news = filtered_news.copy()
            published_et = pd.to_datetime(display_news["published"], utc=True, errors="coerce")
            display_news["时间（ET）"] = published_et.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M")
            display_news["重要性"] = display_news["importance_label"].map(
                {"高": "🔴 高", "中": "🟠 中", "低": "⚪ 低"}
            ).fillna("⚪ 低")
            display_news["Event Type"] = display_news["event_type"].fillna("其他")
            display_news["正面/负面"] = display_news["sentiment"].map(
                {"正面": "🟢 正面", "负面": "🔴 负面", "中性": "⚪ 中性"}
            ).fillna("⚪ 中性")
            display_news["相关公司"] = display_news["companies"].fillna("")
            display_news["来源"] = display_news["source"].fillna("—")
            display_news["标题"] = display_news["title"].fillna("—")
            display_news["原文"] = display_news["url"].fillna("")

            show_cols = ["时间（ET）", "重要性", "Event Type", "来源", "标题", "正面/负面", "相关公司", "原文"]
            st.dataframe(
                display_news[show_cols],
                use_container_width=True,
                hide_index=True,
                height=650,
                column_config={
                    "时间（ET）": st.column_config.TextColumn("时间（ET）", width="small"),
                    "重要性": st.column_config.TextColumn("重要性", width="small"),
                    "Event Type": st.column_config.TextColumn("Event Type", width="medium"),
                    "来源": st.column_config.TextColumn("来源", width="small"),
                    "标题": st.column_config.TextColumn("标题", width="large"),
                    "正面/负面": st.column_config.TextColumn("正面/负面", width="small"),
                    "相关公司": st.column_config.TextColumn("相关公司", width="small"),
                    "原文": st.column_config.LinkColumn("原文", display_text="打开", width="small"),
                },
            )
            st.caption(
                f"当前显示 {len(display_news)} 条。新闻按发布时间倒序；同一标题会去重并合并相关公司。"
            )
