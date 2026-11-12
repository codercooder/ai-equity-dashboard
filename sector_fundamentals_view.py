from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
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
    "P/E (TTM)": ("pe_ttm", "pe_ttm_5y_avg", "x"),
    "EV / EBITDA": ("ev_ebitda", "ev_ebitda_5y_avg", "x"),
    "FCF Yield": ("fcf_yield", "fcf_yield_5y_avg", "%"),
    "Net Debt / EBITDA": ("net_debt_ebitda", "net_debt_ebitda_5y_avg", "x"),
    "CFO Margin": ("cfo_margin", "cfo_margin_5y_avg", "%"),
    "Capex / CFO": ("capex_cfo", "capex_cfo_5y_avg", "%"),
    "FCF Margin": ("fcf_margin", "fcf_margin_5y_avg", "%"),
    "ROIC": ("roic", "roic_5y_avg", "%"),
    "EBITDA Margin": ("ebitda_margin", "ebitda_margin_5y_avg", "%"),
    "ROE (Financials reference)": ("roe", "roe_5y_avg", "%"),
    "P/B (Financials)": ("p_book", "p_book_5y_avg", "x"),
    "Equity / Assets (Financials)": ("equity_assets", "equity_assets_5y_avg", "%"),
    "Assets / Equity (Financials)": ("assets_equity", "assets_equity_5y_avg", "x"),
    "P/FFO Proxy (Real Estate)": ("p_ffo_proxy", "p_ffo_proxy_5y_avg", "x"),
    "FFO Yield Proxy (Real Estate)": ("ffo_proxy_yield", "ffo_proxy_yield_5y_avg", "%"),
    "Debt / Assets (Real Estate)": ("debt_assets", "debt_assets_5y_avg", "%"),
    "Interest Coverage (Real Estate)": ("interest_coverage", "interest_coverage_5y_avg", "x"),
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
    styler = df.style.format(fmt, na_rep="—")
    avg_cols = [c for c in df.columns if "5Y" in str(c)]
    if avg_cols:
        styler = styler.set_properties(
            subset=avg_cols,
            **{
                "color": "#7A8494",
                "background-color": "#F3F5F7",
            },
        )
    return styler


def special_sector_table(summary):
    re_row = summary.loc[summary["sector"] == "Real Estate"].iloc[0]
    fin_row = summary.loc[summary["sector"] == "Financials"].iloc[0]

    rows = [
        (
            "P/FFO Proxy",
            re_row.get("p_ffo_proxy"),
            re_row.get("p_ffo_proxy_5y_avg"),
            "P/E (TTM)",
            fin_row.get("pe_ttm"),
            fin_row.get("pe_ttm_5y_avg"),
            "x",
            "x",
        ),
        (
            "FFO Yield Proxy",
            re_row.get("ffo_proxy_yield"),
            re_row.get("ffo_proxy_yield_5y_avg"),
            "P/B",
            fin_row.get("p_book"),
            fin_row.get("p_book_5y_avg"),
            "%",
            "x",
        ),
        (
            "Net Debt / EBITDA",
            re_row.get("net_debt_ebitda"),
            re_row.get("net_debt_ebitda_5y_avg"),
            "ROE",
            fin_row.get("roe"),
            fin_row.get("roe_5y_avg"),
            "x",
            "%",
        ),
        (
            "Debt / Assets",
            re_row.get("debt_assets"),
            re_row.get("debt_assets_5y_avg"),
            "Equity / Assets",
            fin_row.get("equity_assets"),
            fin_row.get("equity_assets_5y_avg"),
            "%",
            "%",
        ),
        (
            "Interest Coverage",
            re_row.get("interest_coverage"),
            re_row.get("interest_coverage_5y_avg"),
            "Assets / Equity",
            fin_row.get("assets_equity"),
            fin_row.get("assets_equity_5y_avg"),
            "x",
            "x",
        ),
    ]

    def fmt_value(value, kind):
        if value is None or pd.isna(value):
            return "—"
        if kind == "%":
            return f"{value:.1%}"
        return f"{value:.1f}x"

    data = []
    for re_metric, re_cur, re_avg, fin_metric, fin_cur, fin_avg, re_kind, fin_kind in rows:
        data.append(
            {
                "Real Estate 指标": re_metric,
                "Current": fmt_value(re_cur, re_kind),
                "5Y Avg": fmt_value(re_avg, re_kind),
                "Financials 指标": fin_metric,
                "Current ": fmt_value(fin_cur, fin_kind),
                "5Y Avg ": fmt_value(fin_avg, fin_kind),
            }
        )
    return pd.DataFrame(data)


def render_sector_fundamentals():
    summary, history, meta = load_data()
    
    st.title("S&P 500 Sector Fundamentals")
    st.caption("11个GICS板块 · 当前值 vs 过去5个可用财年平均 · 估值 / 杠杆 / 现金流 / ROIC / EBITDA Margin")
    
    if summary.empty:
        st.warning(
            "板块基本面数据快照尚未生成。代码已支持由 GitHub Action 自动构建；"
            "首次数据任务完成后，本页会自动显示结果。"
        )
        return
    
    summary["sector"] = pd.Categorical(summary["sector"], categories=SECTOR_ORDER, ordered=True)
    summary = summary.sort_values("sector").reset_index(drop=True)
    standard_summary = summary[
        ~summary["sector"].astype(str).isin(["Real Estate", "Financials"])
    ].copy()
    
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
    plot_df = standard_summary[["sector", current_col, avg_col]].copy()
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
        all_view = standard_summary[
            [
                "sector",
                "pe_ttm",
                "pe_ttm_5y_avg",
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

        st.markdown("#### Real Estate / Financials 专属指标")
        special = special_sector_table(summary)
        special_styler = (
            special.style
            .set_properties(
                subset=["5Y Avg", "5Y Avg "],
                **{
                    "color": "#7A8494",
                    "background-color": "#F3F5F7",
                },
            )
        )
        st.dataframe(
            special_styler,
            use_container_width=True,
            hide_index=True,
            height=245,
        )
        st.caption(
            "Real Estate 的 FFO 为公开财报推导 proxy（Net Income + D&A），"
            "不是正式 NAREIT FFO/AFFO；Financials 使用 P/B、ROE 和资本结构指标。"
        )
    
    with tab_val:
        val = standard_summary[
            [
                "sector",
                "pe_ttm",
                "pe_ttm_5y_avg",
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
        cf = standard_summary[
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
        bs = standard_summary[
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
    - **Financials**：企业净债务、EBITDA、FCF等口径不具备跨行业可比性，因此单独展示 P/E、P/B、ROE、Equity/Assets、Assets/Equity。
    - **Real Estate**：单独展示 P/FFO Proxy、FFO Yield Proxy、Net Debt/EBITDA、Debt/Assets、Interest Coverage。FFO Proxy = Net Income + D&A，不等同于正式 NAREIT FFO/AFFO。
            """
        )
    
