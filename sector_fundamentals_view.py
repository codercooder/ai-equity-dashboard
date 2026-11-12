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
    "ROA (Financials)": ("roa", "roa_5y_avg", "%"),
    "Net Margin (Financials)": ("net_margin", "net_margin_5y_avg", "%"),
    "Market Cap / Assets (Financials)": ("market_cap_assets", "market_cap_assets_5y_avg", "%"),
    "Revenue / Assets (Financials)": ("revenue_assets", "revenue_assets_5y_avg", "%"),
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


def fmt_metric(value, kind):
    if value is None or pd.isna(value):
        return "—"
    if kind == "%":
        return f"{value:.1%}"
    if kind == "coverage":
        return f"{value:.0%}"
    return f"{value:.1f}x"


OVERVIEW_COLUMNS = [
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


STANDARD_SPECS = [
    ("pe_ttm", "x"),
    ("pe_ttm_5y_avg", "x"),
    ("ev_ebitda", "x"),
    ("ev_ebitda_5y_avg", "x"),
    ("fcf_yield", "%"),
    ("fcf_yield_5y_avg", "%"),
    ("net_debt_ebitda", "x"),
    ("net_debt_ebitda_5y_avg", "x"),
    ("cfo_margin", "%"),
    ("cfo_margin_5y_avg", "%"),
    ("capex_cfo", "%"),
    ("capex_cfo_5y_avg", "%"),
    ("fcf_margin", "%"),
    ("fcf_margin_5y_avg", "%"),
    ("roic", "%"),
    ("roic_5y_avg", "%"),
    ("ebitda_margin", "%"),
    ("ebitda_margin_5y_avg", "%"),
    ("coverage_pct", "coverage"),
]


REAL_ESTATE_SPECS = [
    ("P/FFO Proxy", "p_ffo_proxy", "x"),
    ("P/FFO Proxy 5Y Avg", "p_ffo_proxy_5y_avg", "x"),
    ("FFO Yield Proxy", "ffo_proxy_yield", "%"),
    ("FFO Yield Proxy 5Y Avg", "ffo_proxy_yield_5y_avg", "%"),
    ("Net Debt / EBITDA", "net_debt_ebitda", "x"),
    ("Net Debt / EBITDA 5Y Avg", "net_debt_ebitda_5y_avg", "x"),
    ("Debt / Assets", "debt_assets", "%"),
    ("Debt / Assets 5Y Avg", "debt_assets_5y_avg", "%"),
    ("Interest Coverage", "interest_coverage", "x"),
    ("Interest Coverage 5Y Avg", "interest_coverage_5y_avg", "x"),
    ("EBITDA Margin", "ebitda_margin", "%"),
    ("EBITDA Margin 5Y Avg", "ebitda_margin_5y_avg", "%"),
    ("CFO Margin", "cfo_margin", "%"),
    ("CFO Margin 5Y Avg", "cfo_margin_5y_avg", "%"),
    ("Capex / CFO", "capex_cfo", "%"),
    ("Capex / CFO 5Y Avg", "capex_cfo_5y_avg", "%"),
    ("P/B", "p_book", "x"),
    ("P/B 5Y Avg", "p_book_5y_avg", "x"),
    ("Coverage", "coverage_pct", "coverage"),
]


FINANCIALS_SPECS = [
    ("P/E (TTM)", "pe_ttm", "x"),
    ("P/E 5Y Avg", "pe_ttm_5y_avg", "x"),
    ("P/B", "p_book", "x"),
    ("P/B 5Y Avg", "p_book_5y_avg", "x"),
    ("ROE", "roe", "%"),
    ("ROE 5Y Avg", "roe_5y_avg", "%"),
    ("ROA", "roa", "%"),
    ("ROA 5Y Avg", "roa_5y_avg", "%"),
    ("Net Margin", "net_margin", "%"),
    ("Net Margin 5Y Avg", "net_margin_5y_avg", "%"),
    ("Equity / Assets", "equity_assets", "%"),
    ("Equity / Assets 5Y Avg", "equity_assets_5y_avg", "%"),
    ("Assets / Equity", "assets_equity", "x"),
    ("Assets / Equity 5Y Avg", "assets_equity_5y_avg", "x"),
    ("Market Cap / Assets", "market_cap_assets", "%"),
    ("Market Cap / Assets 5Y Avg", "market_cap_assets_5y_avg", "%"),
    ("Revenue / Assets", "revenue_assets", "%"),
    ("Revenue / Assets 5Y Avg", "revenue_assets_5y_avg", "%"),
    ("Coverage", "coverage_pct", "coverage"),
]


def build_overview_table(summary, standard_summary):
    rows = []

    for _, row in standard_summary.iterrows():
        out = {"Sector": str(row["sector"])}
        for col, (field, kind) in zip(OVERVIEW_COLUMNS[1:], STANDARD_SPECS):
            out[col] = fmt_metric(row.get(field), kind)
        rows.append(out)

    def append_special(sector_name, specs):
        sector_row = summary.loc[
            summary["sector"].astype(str) == sector_name
        ]
        if sector_row.empty:
            return
        sector_row = sector_row.iloc[0]

        metric_row = {"Sector": f"{sector_name} 指标"}
        data_row = {"Sector": sector_name}
        for col, (label, field, kind) in zip(OVERVIEW_COLUMNS[1:], specs):
            metric_row[col] = label
            data_row[col] = fmt_metric(sector_row.get(field), kind)

        rows.append(metric_row)
        rows.append(data_row)

    append_special("Real Estate", REAL_ESTATE_SPECS)
    append_special("Financials", FINANCIALS_SPECS)

    return pd.DataFrame(rows, columns=OVERVIEW_COLUMNS)


def style_overview(df):
    styler = df.style

    avg_cols = [c for c in df.columns if "5Y" in c]
    if avg_cols:
        styler = styler.set_properties(
            subset=avg_cols,
            **{
                "color": "#7A8494",
                "background-color": "#F3F5F7",
            },
        )

    metric_rows = [
        i
        for i, label in enumerate(df["Sector"].astype(str))
        if label.endswith("指标")
    ]
    if metric_rows:
        styler = styler.set_properties(
            subset=pd.IndexSlice[metric_rows, :],
            **{
                "font-weight": "600",
                "background-color": "#EEF2F6",
                "color": "#4B5563",
            },
        )
        if avg_cols:
            styler = styler.set_properties(
                subset=pd.IndexSlice[metric_rows, avg_cols],
                **{
                    "font-weight": "600",
                    "background-color": "#E8EDF2",
                    "color": "#7A8494",
                },
            )

    return styler


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
        overview = build_overview_table(summary, standard_summary)
        st.dataframe(
            style_overview(overview),
            use_container_width=True,
            hide_index=True,
            height=560,
        )
        st.caption(
            "Utilities 后直接接 Real Estate 指标行/数据行，再接 Financials 指标行/数据行；"
            "所有 5Y Avg 位置使用浅色区分。Real Estate 的 FFO 为公开财报推导 proxy，"
            "并非正式 NAREIT FFO/AFFO。"
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
            if hist_col not in history.columns:
                h = pd.DataFrame(columns=["year", hist_col])
            else:
                h = (
                    history.loc[
                        history["sector"] == selected_sector,
                        ["year", hist_col],
                    ]
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
    - **Financials**：在总览中使用专属19格：P/E、P/B、ROE、ROA、Net Margin、Equity/Assets、Assets/Equity、Market Cap/Assets、Revenue/Assets，各自配 Current/5Y Avg，再加 Coverage。
    - **Real Estate**：在总览中使用专属19格：P/FFO Proxy、FFO Yield Proxy、Net Debt/EBITDA、Debt/Assets、Interest Coverage、EBITDA Margin、CFO Margin、Capex/CFO、P/B，各自配 Current/5Y Avg，再加 Coverage。FFO Proxy = Net Income + D&A，不等同于正式 NAREIT FFO/AFFO。
            """
        )
    
