from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="S&P 500 Sector Fundamentals", page_icon="◈", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SUMMARY_PATH = DATA_DIR / "sp500_sector_fundamentals.csv"
HISTORY_PATH = DATA_DIR / "sp500_sector_fundamentals_history.csv"
META_PATH = DATA_DIR / "sp500_sector_fundamentals_meta.json"

SECTOR_ORDER = [
    "Information Technology",
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Industrials",
    "Materials",
    "Energy",
    "Utilities",
    "Real Estate",
    "Financials",
]

METRICS = {
    "P/E (TTM)": ("pe_ttm", "pe_5y_avg", "x"),
    "EV / EBITDA": ("ev_ebitda", "ev_ebitda_5y_avg", "x"),
    "FCF Yield": ("fcf_yield", "fcf_yield_5y_avg", "%"),
    "Net Debt / EBITDA": ("net_debt_ebitda", "net_debt_ebitda_5y_avg", "x"),
    "CFO Margin": ("cfo_margin", "cfo_margin_5y_avg", "%"),
    "Capex / CFO": ("capex_cfo", "capex_cfo_5y_avg", "%"),
    "FCF Margin": ("fcf_margin", "fcf_margin_5y_avg", "%"),
    "ROIC": ("roic", "roic_5y_avg", "%"),
    "EBITDA Margin": ("ebitda_margin", "ebitda_margin_5y_avg", "%"),
    "ROE (Financials reference)": ("roe", "roe_5y_avg", "%"),
}


@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    if not SUMMARY_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    summary = pd.read_csv(SUMMARY_PATH)
    history = pd.read_csv(HISTORY_PATH) if HISTORY_PATH.exists() else pd.DataFrame()
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return summary, history, meta


def style_view(df, percent_cols=(), ratio_cols=()):
    fmt = {"Coverage": "{:.0%}"}
    fmt.update({c: "{:.1%}" for c in percent_cols if c in df.columns})
    fmt.update({c: "{:.1f}x" for c in ratio_cols if c in df.columns})
    return df.style.format(fmt, na_rep="—")


summary, history, meta = load_data()

st.title("S&P 500 Sector Fundamentals")
st.caption("11个GICS板块 · 当前值 vs 过去5个可用财年平均 · 估值 / 杠杆 / 现金流 / ROIC / EBITDA Margin")

if summary.empty:
    st.warning(
        "板块基本面数据快照尚未生成。代码已支持由 GitHub Action 自动构建；"
        "首次数据任务完成后，本页会自动显示结果。"
    )
    st.stop()

summary["sector"] = pd.Categorical(summary["sector"], categories=SECTOR_ORDER, ordered=True)
summary = summary.sort_values("sector").reset_index(drop=True)

as_of = meta.get("as_of") or (
    summary["as_of"].iloc[0] if "as_of" in summary.columns and len(summary) else "—"
)
source_note = meta.get(
    "source",
    "Yahoo Finance statements + S&P 500 constituent classification",
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("数据日期", str(as_of))
m2.metric("覆盖板块", f"{summary['sector'].notna().sum()} / 11")
m3.metric(
    "成分股",
    f"{int(pd.to_numeric(summary.get('constituents'), errors='coerce').fillna(0).sum())}",
)
coverage = pd.to_numeric(summary.get("coverage_pct"), errors="coerce")
m4.metric(
    "中位数据覆盖",
    "—" if coverage is None or coverage.dropna().empty else f"{coverage.median():.0%}",
)

st.caption(
    "历史口径：以当前S&P 500成分股向后回溯其财务报表，按板块自下而上聚合；"
    "5Y Avg 为最近最多5个可用财年的板块年度比率算术平均。"
    "P/E 使用TTM口径，因为免费公开数据无法稳定获得5年历史 forward-consensus P/E。"
)

metric_name = st.selectbox("跨板块比较指标", list(METRICS.keys())[:-1], index=0)
current_col, avg_col, kind = METRICS[metric_name]
plot_df = summary[["sector", current_col, avg_col]].copy()
plot_df["sector"] = plot_df["sector"].astype(str)
plot_df = plot_df.dropna(how="all", subset=[current_col, avg_col])

if not plot_df.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=plot_df["sector"], y=plot_df[current_col], name="Current"))
    fig.add_trace(go.Bar(x=plot_df["sector"], y=plot_df[avg_col], name="5Y Avg"))
    fig.update_layout(
        barmode="group",
        height=430,
        margin=dict(l=10, r=10, t=30, b=110),
        yaxis_title=metric_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if kind == "%":
        fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

tab_all, tab_val, tab_cf, tab_bs, tab_hist = st.tabs(
    ["总览", "估值", "现金流", "杠杆与回报", "5Y历史"]
)

with tab_all:
    all_view = summary[
        [
            "sector",
            "pe_ttm",
            "pe_5y_avg",
            "ev_ebitda",
            "ev_ebitda_5y_avg",
            "fcf_yield",
            "fcf_yield_5y_avg",
            "net_debt_ebitda",
            "net_debt_ebitda_5y_avg",
            "cfo_margin",
            "cfo_margin_5y_avg",
            "capex_cfo",
            "capex_cfo_5y_avg",
            "fcf_margin",
            "fcf_margin_5y_avg",
            "roic",
            "roic_5y_avg",
            "ebitda_margin",
            "ebitda_margin_5y_avg",
            "coverage_pct",
        ]
    ].copy()
    all_view.columns = [
        "Sector",
        "P/E",
        "P/E 5Y",
        "EV/EBITDA",
        "EV/EBITDA 5Y",
        "FCF Yield",
        "FCF Yield 5Y",
        "Net Debt/EBITDA",
        "Net Debt/EBITDA 5Y",
        "CFO Margin",
        "CFO Margin 5Y",
        "Capex/CFO",
        "Capex/CFO 5Y",
        "FCF Margin",
        "FCF Margin 5Y",
        "ROIC",
        "ROIC 5Y",
        "EBITDA Margin",
        "EBITDA Margin 5Y",
        "Coverage",
    ]
    st.dataframe(
        style_view(
            all_view,
            percent_cols=[
                "FCF Yield",
                "FCF Yield 5Y",
                "CFO Margin",
                "CFO Margin 5Y",
                "Capex/CFO",
                "Capex/CFO 5Y",
                "FCF Margin",
                "FCF Margin 5Y",
                "ROIC",
                "ROIC 5Y",
                "EBITDA Margin",
                "EBITDA Margin 5Y",
            ],
            ratio_cols=[
                "P/E",
                "P/E 5Y",
                "EV/EBITDA",
                "EV/EBITDA 5Y",
                "Net Debt/EBITDA",
                "Net Debt/EBITDA 5Y",
            ],
        ),
        use_container_width=True,
        hide_index=True,
        height=500,
    )

with tab_val:
    val = summary[
        [
            "sector",
            "pe_ttm",
            "pe_5y_avg",
            "ev_ebitda",
            "ev_ebitda_5y_avg",
            "fcf_yield",
            "fcf_yield_5y_avg",
        ]
    ].copy()
    val.columns = [
        "Sector",
        "P/E (TTM)",
        "P/E 5Y Avg",
        "EV/EBITDA",
        "EV/EBITDA 5Y Avg",
        "FCF Yield",
        "FCF Yield 5Y Avg",
    ]
    st.dataframe(
        style_view(
            val,
            percent_cols=["FCF Yield", "FCF Yield 5Y Avg"],
            ratio_cols=["P/E (TTM)", "P/E 5Y Avg", "EV/EBITDA", "EV/EBITDA 5Y Avg"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "估值采用板块聚合口径：总市值/总净利润、总EV/总EBITDA、总FCF/总市值，"
        "不对个股倍数做简单平均。"
    )

with tab_cf:
    cf = summary[
        [
            "sector",
            "cfo_margin",
            "cfo_margin_5y_avg",
            "capex_cfo",
            "capex_cfo_5y_avg",
            "fcf_margin",
            "fcf_margin_5y_avg",
            "fcf_yield",
            "fcf_yield_5y_avg",
        ]
    ].copy()
    cf.columns = [
        "Sector",
        "CFO Margin",
        "CFO Margin 5Y Avg",
        "Capex/CFO",
        "Capex/CFO 5Y Avg",
        "FCF Margin",
        "FCF Margin 5Y Avg",
        "FCF Yield",
        "FCF Yield 5Y Avg",
    ]
    st.dataframe(
        style_view(cf, percent_cols=[c for c in cf.columns if c != "Sector"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "现金流链：Revenue → CFO → Capex → FCF → Market Valuation。"
        "Capex/CFO 是再投资强度指标，不机械按越低越好解读。"
    )

with tab_bs:
    bs = summary[
        [
            "sector",
            "net_debt_ebitda",
            "net_debt_ebitda_5y_avg",
            "roic",
            "roic_5y_avg",
            "ebitda_margin",
            "ebitda_margin_5y_avg",
            "roe",
            "roe_5y_avg",
        ]
    ].copy()
    bs.columns = [
        "Sector",
        "Net Debt/EBITDA",
        "Net Debt/EBITDA 5Y Avg",
        "ROIC",
        "ROIC 5Y Avg",
        "EBITDA Margin",
        "EBITDA Margin 5Y Avg",
        "ROE",
        "ROE 5Y Avg",
    ]
    st.dataframe(
        style_view(
            bs,
            percent_cols=[
                "ROIC",
                "ROIC 5Y Avg",
                "EBITDA Margin",
                "EBITDA Margin 5Y Avg",
                "ROE",
                "ROE 5Y Avg",
            ],
            ratio_cols=["Net Debt/EBITDA", "Net Debt/EBITDA 5Y Avg"],
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Financials：Net Debt/EBITDA、ROIC、EBITDA Margin 等企业工业口径标为 N/M；"
        "同时提供 ROE 作为参考。"
    )

with tab_hist:
    if history.empty:
        st.info("暂无年度历史明细。")
    else:
        selected_sector = st.selectbox(
            "板块",
            SECTOR_ORDER,
            index=0,
            key="fund_hist_sector",
        )
        hist_metric = st.selectbox(
            "历史指标",
            list(METRICS.keys()),
            index=0,
            key="fund_hist_metric",
        )
        hist_col, _, hist_kind = METRICS[hist_metric]
        h = (
            history.loc[history["sector"] == selected_sector, ["year", hist_col]]
            .dropna()
            .sort_values("year")
        )
        if h.empty:
            st.info(f"{selected_sector} 暂无 {hist_metric} 可用历史数据。")
        else:
            line = go.Figure(
                go.Scatter(
                    x=h["year"].astype(str),
                    y=h[hist_col],
                    mode="lines+markers",
                    name=hist_metric,
                )
            )
            line.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=20, b=30),
                yaxis_title=hist_metric,
                xaxis_title="Fiscal year",
            )
            if hist_kind == "%":
                line.update_yaxes(tickformat=".0%")
            st.plotly_chart(line, use_container_width=True, config={"displaylogo": False})

with st.expander("方法与数据质量说明"):
    st.markdown(
        f"""
- **数据源**：{source_note}。
- **Universe**：当前 S&P 500 成分股及其 GICS Sector；历史部分为 current-constituent backcast，因此存在 survivorship bias。
- **P/E**：Current = 当前总市值 / TTM总净利润；5Y Avg = 最近最多5个可用财年的板块年度 P/E 平均。
- **EV/EBITDA**：EV = Market Cap + Total Debt - Cash。
- **FCF**：Operating Cash Flow - Capex；Capex 按现金流出绝对值处理。
- **ROIC**：NOPAT / Invested Capital；税率由 Tax Provision / Pretax Income 估算并做合理边界处理。
- **Coverage**：可计算核心财务指标的公司市值 / 该板块可取得市值的公司总市值。
- **Financials**：企业净债务、EBITDA、FCF等口径不具备跨行业可比性，因此相关字段保留为空，并额外展示 ROE。
- **Real Estate**：P/E 对 REITs 解释力有限；后续如接入稳定 FFO/AFFO 数据，可再升级为 P/FFO 与 AFFO Yield。
        """
    )
