from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

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

ALIASES = {
    "revenue": ["TotalRevenue", "OperatingRevenue"],
    "net_income": ["NetIncome", "NetIncomeCommonStockholders"],
    "ebit": ["EBIT", "OperatingIncome"],
    "ebitda": ["EBITDA", "NormalizedEBITDA"],
    "tax": ["TaxProvision"],
    "pretax": ["PretaxIncome"],
    "cfo": ["OperatingCashFlow", "TotalCashFromOperatingActivities"],
    "capex": ["CapitalExpenditure"],
    "depr": [
        "DepreciationAndAmortization",
        "Depreciation",
        "DepreciationAmortizationDepletion",
    ],
    "debt": ["TotalDebt"],
    "cash": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalents",
    ],
    "equity": ["StockholdersEquity", "TotalStockholderEquity"],
    "assets": ["TotalAssets"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonOperating"],
    "invested": ["InvestedCapital"],
    "shares": ["OrdinarySharesNumber", "ShareIssued"],
}


def finite(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def safe_div(a, b):
    a, b = finite(a), finite(b)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def get_constituents() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(
        url,
        headers={"User-Agent": "AIEquityDashboard/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise RuntimeError("Could not find S&P 500 constituents table")

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [
            td.get_text(" ", strip=True)
            for td in tr.find_all(["td", "th"])
        ]
        if len(cells) < 4:
            continue
        rows.append(
            {
                "ticker": cells[0].replace(".", "-"),
                "security": cells[1],
                "sector": cells[2],
            }
        )

    df = pd.DataFrame(rows)
    df = (
        df[df["sector"].isin(SECTOR_ORDER)]
        .drop_duplicates("ticker")
        .reset_index(drop=True)
    )
    if len(df) < 450:
        raise RuntimeError(f"Unexpected S&P 500 universe size: {len(df)}")
    return df


def stmt(obj, kind: str, freq: str) -> pd.DataFrame:
    try:
        getter = {
            "income": obj.get_income_stmt,
            "cashflow": obj.get_cash_flow,
            "balance": obj.get_balance_sheet,
        }[kind]
        df = getter(freq=freq)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def series(df: pd.DataFrame, key: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in ALIASES[key]:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")
            return s.dropna().sort_index(ascending=False)
    return pd.Series(dtype=float)


def nearest_value(s: pd.Series, date: pd.Timestamp, days: int = 75):
    if s is None or s.empty:
        return np.nan
    idx = pd.to_datetime(s.index, errors="coerce")
    diffs = np.abs((idx - pd.Timestamp(date)).days)
    if len(diffs) == 0:
        return np.nan
    pos = int(np.argmin(diffs))
    if diffs[pos] > days:
        return np.nan
    return finite(s.iloc[pos])


def ttm(s: pd.Series):
    if s is None or s.empty:
        return np.nan
    vals = (
        pd.to_numeric(s, errors="coerce")
        .dropna()
        .sort_index(ascending=False)
        .head(4)
    )
    return float(vals.sum()) if len(vals) >= 4 else np.nan


def latest(s: pd.Series):
    if s is None or s.empty:
        return np.nan
    vals = pd.to_numeric(s, errors="coerce").dropna().sort_index(ascending=False)
    return finite(vals.iloc[0]) if len(vals) else np.nan


def tax_rate(tax, pretax):
    r = safe_div(tax, pretax)
    if not np.isfinite(r):
        return 0.21
    return min(max(r, 0.0), 0.40)


def price_on_or_before(price_series: pd.Series, date: pd.Timestamp):
    if price_series is None or price_series.empty:
        return np.nan
    s = price_series.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    eligible = s.loc[s.index <= pd.Timestamp(date).tz_localize(None)]
    if eligible.empty:
        return np.nan
    return finite(eligible.iloc[-1])


def extract_company(ticker: str, sector_name: str, prices: pd.DataFrame) -> dict:
    obj = yf.Ticker(ticker)

    ia = stmt(obj, "income", "yearly")
    ca = stmt(obj, "cashflow", "yearly")
    ba = stmt(obj, "balance", "yearly")
    iq = stmt(obj, "income", "quarterly")
    cq = stmt(obj, "cashflow", "quarterly")
    bq = stmt(obj, "balance", "quarterly")

    try:
        fi = obj.fast_info
        mcap = finite(
            fi.get("market_cap")
            if hasattr(fi, "get")
            else fi.market_cap
        )
        current_price = finite(
            fi.get("last_price")
            if hasattr(fi, "get")
            else fi.last_price
        )
    except Exception:
        mcap, current_price = np.nan, np.nan

    q = {
        k: series(
            iq
            if k in {"revenue", "net_income", "ebit", "ebitda", "tax", "pretax", "interest_expense"}
            else cq
            if k in {"cfo", "capex", "depr"}
            else bq,
            k,
        )
        for k in ALIASES
    }
    a = {
        k: series(
            ia
            if k in {"revenue", "net_income", "ebit", "ebitda", "tax", "pretax", "interest_expense"}
            else ca
            if k in {"cfo", "capex", "depr"}
            else ba,
            k,
        )
        for k in ALIASES
    }

    revenue = ttm(q["revenue"])
    net_income = ttm(q["net_income"])
    ebit = ttm(q["ebit"])
    ebitda = ttm(q["ebitda"])
    depr = ttm(q["depr"])
    if not np.isfinite(ebitda) and np.isfinite(ebit) and np.isfinite(depr):
        ebitda = ebit + abs(depr)

    cfo = ttm(q["cfo"])
    capex_raw = ttm(q["capex"])
    capex = abs(capex_raw) if np.isfinite(capex_raw) else np.nan

    debt = latest(q["debt"])
    cash = latest(q["cash"])
    equity = latest(q["equity"])
    assets = latest(q["assets"])
    interest_expense = ttm(q["interest_expense"])
    invested = latest(q["invested"])

    if not np.isfinite(debt):
        debt = latest(a["debt"])
    if not np.isfinite(cash):
        cash = latest(a["cash"])
    if not np.isfinite(equity):
        equity = latest(a["equity"])
    if not np.isfinite(assets):
        assets = latest(a["assets"])
    if not np.isfinite(interest_expense):
        interest_expense = nearest_value(a["interest_expense"], a["interest_expense"].index[0]) if not a["interest_expense"].empty else np.nan
    if not np.isfinite(invested) and np.isfinite(debt) and np.isfinite(equity):
        invested = debt + equity - (cash if np.isfinite(cash) else 0.0)

    tr = tax_rate(ttm(q["tax"]), ttm(q["pretax"]))
    nopat = ebit * (1.0 - tr) if np.isfinite(ebit) else np.nan

    current_shares = latest(q["shares"])
    if not np.isfinite(current_shares):
        current_shares = latest(a["shares"])
    if (
        not np.isfinite(current_shares)
        and np.isfinite(mcap)
        and np.isfinite(current_price)
        and current_price > 0
    ):
        current_shares = mcap / current_price

    # fast_info can be incomplete/rate-limited in GitHub Actions. Use the
    # already-downloaded bulk close history as a deterministic price fallback,
    # then infer market cap from the latest reported share count.
    if (not np.isfinite(current_price)) and ticker in prices.columns:
        px_vals = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if not px_vals.empty:
            current_price = finite(px_vals.iloc[-1])
    if (
        not np.isfinite(mcap)
        and np.isfinite(current_shares)
        and np.isfinite(current_price)
        and current_price > 0
    ):
        mcap = current_shares * current_price

    current = {
        "ticker": ticker,
        "sector": sector_name,
        "market_cap": mcap,
        "revenue": revenue,
        "net_income": net_income,
        "ebitda": ebitda,
        "ebit": ebit,
        "cfo": cfo,
        "capex": capex,
        "debt": debt,
        "cash": cash,
        "equity": equity,
        "assets": assets,
        "interest_expense": abs(interest_expense) if np.isfinite(interest_expense) else np.nan,
        "depreciation": abs(depr) if np.isfinite(depr) else np.nan,
        "ffo_proxy": (
            net_income + abs(depr)
            if np.isfinite(net_income) and np.isfinite(depr)
            else np.nan
        ),
        "invested_capital": invested,
        "nopat": nopat,
    }

    years = []
    annual_dates = set()
    for key in [
        "revenue",
        "net_income",
        "ebit",
        "ebitda",
        "cfo",
        "capex",
        "debt",
        "equity",
        "assets",
        "interest_expense",
        "depr",
    ]:
        annual_dates.update(
            pd.to_datetime(a[key].index, errors="coerce").dropna().tolist()
        )
    annual_dates = sorted(annual_dates, reverse=True)[:6]
    price_series = (
        prices[ticker]
        if ticker in prices.columns
        else pd.Series(dtype=float)
    )

    for d in annual_dates:
        rev = nearest_value(a["revenue"], d)
        ni = nearest_value(a["net_income"], d)
        eb = nearest_value(a["ebit"], d)
        ebitda_y = nearest_value(a["ebitda"], d)
        dep_y = nearest_value(a["depr"], d)
        if (
            not np.isfinite(ebitda_y)
            and np.isfinite(eb)
            and np.isfinite(dep_y)
        ):
            ebitda_y = eb + abs(dep_y)

        cfo_y = nearest_value(a["cfo"], d)
        cap_y_raw = nearest_value(a["capex"], d)
        cap_y = abs(cap_y_raw) if np.isfinite(cap_y_raw) else np.nan
        debt_y = nearest_value(a["debt"], d)
        cash_y = nearest_value(a["cash"], d)
        eq_y = nearest_value(a["equity"], d)
        assets_y = nearest_value(a["assets"], d)
        interest_y = nearest_value(a["interest_expense"], d)
        inv_y = nearest_value(a["invested"], d)
        if (
            not np.isfinite(inv_y)
            and np.isfinite(debt_y)
            and np.isfinite(eq_y)
        ):
            inv_y = debt_y + eq_y - (
                cash_y if np.isfinite(cash_y) else 0.0
            )

        tr_y = tax_rate(
            nearest_value(a["tax"], d),
            nearest_value(a["pretax"], d),
        )
        nopat_y = eb * (1.0 - tr_y) if np.isfinite(eb) else np.nan
        shares_y = nearest_value(a["shares"], d)
        if not np.isfinite(shares_y):
            shares_y = current_shares

        px_y = price_on_or_before(price_series, pd.Timestamp(d))
        mcap_y = (
            shares_y * px_y
            if np.isfinite(shares_y) and np.isfinite(px_y)
            else np.nan
        )

        years.append(
            {
                "ticker": ticker,
                "sector": sector_name,
                "year": int(pd.Timestamp(d).year),
                "market_cap": mcap_y,
                "revenue": rev,
                "net_income": ni,
                "ebitda": ebitda_y,
                "ebit": eb,
                "cfo": cfo_y,
                "capex": cap_y,
                "debt": debt_y,
                "cash": cash_y,
                "equity": eq_y,
                "assets": assets_y,
                "interest_expense": abs(interest_y) if np.isfinite(interest_y) else np.nan,
                "depreciation": abs(dep_y) if np.isfinite(dep_y) else np.nan,
                "ffo_proxy": (
                    ni + abs(dep_y)
                    if np.isfinite(ni) and np.isfinite(dep_y)
                    else np.nan
                ),
                "invested_capital": inv_y,
                "nopat": nopat_y,
            }
        )

    time.sleep(0.05)
    return {"current": current, "history": years}


def aggregate(rows: list[dict], sector_name: str) -> dict:
    def eligible(*fields):
        return [
            r
            for r in rows
            if all(np.isfinite(finite(r.get(f))) for f in fields)
        ]

    def s(rs, field):
        if not rs:
            return np.nan
        return float(np.nansum([finite(r.get(field)) for r in rs]))

    total_cap_rows = eligible("market_cap")
    total_cap = s(total_cap_rows, "market_cap")
    core_rows = eligible("market_cap", "revenue", "cfo")
    coverage = safe_div(s(core_rows, "market_cap"), total_cap)

    pe_rows = eligible("market_cap", "net_income")
    pe_den = s(pe_rows, "net_income")
    pe = (
        safe_div(s(pe_rows, "market_cap"), pe_den)
        if np.isfinite(pe_den) and pe_den > 0
        else np.nan
    )

    ev_rows = eligible("market_cap", "debt", "cash", "ebitda")
    ev = (
        s(ev_rows, "market_cap")
        + s(ev_rows, "debt")
        - s(ev_rows, "cash")
        if ev_rows
        else np.nan
    )
    ev_ebitda = safe_div(ev, s(ev_rows, "ebitda"))

    lev_rows = eligible("debt", "cash", "ebitda")
    net_debt_ebitda = safe_div(
        s(lev_rows, "debt") - s(lev_rows, "cash"),
        s(lev_rows, "ebitda"),
    )

    cfo_rows = eligible("cfo", "revenue")
    cfo_margin = safe_div(
        s(cfo_rows, "cfo"),
        s(cfo_rows, "revenue"),
    )

    cap_rows = eligible("capex", "cfo")
    capex_cfo = safe_div(
        s(cap_rows, "capex"),
        s(cap_rows, "cfo"),
    )

    fcf_rows = eligible("cfo", "capex", "revenue")
    fcf = (
        s(fcf_rows, "cfo") - s(fcf_rows, "capex")
        if fcf_rows
        else np.nan
    )
    fcf_margin = safe_div(fcf, s(fcf_rows, "revenue"))

    fcfy_rows = eligible("cfo", "capex", "market_cap")
    fcf_yield = safe_div(
        s(fcfy_rows, "cfo") - s(fcfy_rows, "capex"),
        s(fcfy_rows, "market_cap"),
    )

    roic_rows = eligible("nopat", "invested_capital")
    roic = safe_div(
        s(roic_rows, "nopat"),
        s(roic_rows, "invested_capital"),
    )

    margin_rows = eligible("ebitda", "revenue")
    ebitda_margin = safe_div(
        s(margin_rows, "ebitda"),
        s(margin_rows, "revenue"),
    )

    roe_rows = eligible("net_income", "equity")
    roe = safe_div(
        s(roe_rows, "net_income"),
        s(roe_rows, "equity"),
    )

    roa_rows = eligible("net_income", "assets")
    roa = safe_div(
        s(roa_rows, "net_income"),
        s(roa_rows, "assets"),
    )

    margin_rows_fin = eligible("net_income", "revenue")
    net_margin = safe_div(
        s(margin_rows_fin, "net_income"),
        s(margin_rows_fin, "revenue"),
    )

    mcap_assets_rows = eligible("market_cap", "assets")
    market_cap_assets = safe_div(
        s(mcap_assets_rows, "market_cap"),
        s(mcap_assets_rows, "assets"),
    )

    revenue_assets_rows = eligible("revenue", "assets")
    revenue_assets = safe_div(
        s(revenue_assets_rows, "revenue"),
        s(revenue_assets_rows, "assets"),
    )

    pb_rows = eligible("market_cap", "equity")
    pb_den = s(pb_rows, "equity")
    p_book = (
        safe_div(s(pb_rows, "market_cap"), pb_den)
        if np.isfinite(pb_den) and pb_den > 0
        else np.nan
    )

    asset_rows = eligible("debt", "assets")
    debt_assets = safe_div(
        s(asset_rows, "debt"),
        s(asset_rows, "assets"),
    )

    equity_asset_rows = eligible("equity", "assets")
    equity_assets = safe_div(
        s(equity_asset_rows, "equity"),
        s(equity_asset_rows, "assets"),
    )
    assets_equity = safe_div(
        s(equity_asset_rows, "assets"),
        s(equity_asset_rows, "equity"),
    )

    intcov_rows = eligible("ebit", "interest_expense")
    interest_coverage = safe_div(
        s(intcov_rows, "ebit"),
        s(intcov_rows, "interest_expense"),
    )

    ffo_rows = eligible("ffo_proxy", "market_cap")
    ffo_proxy_total = s(ffo_rows, "ffo_proxy")
    ffo_proxy_yield = safe_div(
        ffo_proxy_total,
        s(ffo_rows, "market_cap"),
    )
    p_ffo_proxy = (
        safe_div(s(ffo_rows, "market_cap"), ffo_proxy_total)
        if np.isfinite(ffo_proxy_total) and ffo_proxy_total > 0
        else np.nan
    )

    if sector_name == "Financials":
        ev_ebitda = np.nan
        net_debt_ebitda = np.nan
        cfo_margin = np.nan
        capex_cfo = np.nan
        fcf_margin = np.nan
        fcf_yield = np.nan
        roic = np.nan
        ebitda_margin = np.nan

    return {
        "pe_ttm": pe,
        "ev_ebitda": ev_ebitda,
        "fcf_yield": fcf_yield,
        "net_debt_ebitda": net_debt_ebitda,
        "cfo_margin": cfo_margin,
        "capex_cfo": capex_cfo,
        "fcf_margin": fcf_margin,
        "roic": roic,
        "ebitda_margin": ebitda_margin,
        "roe": roe,
        "roa": roa,
        "net_margin": net_margin,
        "market_cap_assets": market_cap_assets,
        "revenue_assets": revenue_assets,
        "p_book": p_book,
        "debt_assets": debt_assets,
        "equity_assets": equity_assets,
        "assets_equity": assets_equity,
        "interest_coverage": interest_coverage,
        "ffo_proxy_yield": ffo_proxy_yield,
        "p_ffo_proxy": p_ffo_proxy,
        "coverage_pct": coverage,
    }


def download_prices(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        data = yf.download(
            chunk,
            period="6y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].copy()
        else:
            close = data[["Close"]].rename(columns={"Close": chunk[0]})
        if isinstance(close, pd.Series):
            close = close.to_frame(name=chunk[0])
        frames.append(close)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    constituents = get_constituents()
    tickers = constituents["ticker"].tolist()
    prices = download_prices(tickers)
    sector_map = dict(
        zip(constituents["ticker"], constituents["sector"])
    )

    company_results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                extract_company,
                ticker,
                sector_map[ticker],
                prices,
            ): ticker
            for ticker in tickers
        }
        for n, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                company_results.append(future.result())
            except Exception as exc:
                print(f"WARN {ticker}: {exc}")
            if n % 50 == 0:
                print(f"Processed {n}/{len(tickers)}")

    if len(company_results) < 300:
        raise RuntimeError(
            f"Only {len(company_results)} companies returned usable payloads; "
            "refusing to overwrite snapshot"
        )

    current_rows = [r["current"] for r in company_results]
    history_rows = [
        h
        for r in company_results
        for h in r["history"]
    ]

    annual_out = []
    for sector_name in SECTOR_ORDER:
        sector_hist = [
            r
            for r in history_rows
            if r["sector"] == sector_name
        ]
        years = sorted(
            {r["year"] for r in sector_hist},
            reverse=True,
        )[:5]
        for year in sorted(years):
            row = aggregate(
                [
                    r
                    for r in sector_hist
                    if r["year"] == year
                ],
                sector_name,
            )
            row.update(
                {
                    "sector": sector_name,
                    "year": year,
                }
            )
            annual_out.append(row)

    hist_df = pd.DataFrame(annual_out)
    metric_cols = [
        "pe_ttm",
        "ev_ebitda",
        "fcf_yield",
        "net_debt_ebitda",
        "cfo_margin",
        "capex_cfo",
        "fcf_margin",
        "roic",
        "ebitda_margin",
        "roe",
        "roa",
        "net_margin",
        "market_cap_assets",
        "revenue_assets",
        "p_book",
        "debt_assets",
        "equity_assets",
        "assets_equity",
        "interest_coverage",
        "ffo_proxy_yield",
        "p_ffo_proxy",
    ]

    summary_rows = []
    as_of = datetime.now(timezone.utc).date().isoformat()
    for sector_name in SECTOR_ORDER:
        cur_members = [
            r
            for r in current_rows
            if r["sector"] == sector_name
        ]
        cur = aggregate(cur_members, sector_name)
        h = (
            hist_df.loc[hist_df["sector"] == sector_name]
            .sort_values("year")
            .tail(5)
        )
        row = {
            "sector": sector_name,
            "as_of": as_of,
            "constituents": len(cur_members),
            **cur,
        }
        for col in metric_cols:
            vals = (
                pd.to_numeric(h[col], errors="coerce").dropna()
                if col in h.columns
                else pd.Series(dtype=float)
            )
            row[f"{col}_5y_avg"] = (
                float(vals.mean())
                if len(vals)
                else np.nan
            )
            row[f"{col}_history_years"] = int(len(vals))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    hist_df.to_csv(HISTORY_PATH, index=False)

    meta = {
        "as_of": as_of,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": (
            "S&P 500 constituents from Wikipedia; company financial statements, "
            "market cap and prices from Yahoo Finance via yfinance"
        ),
        "universe_method": (
            "Current S&P 500 constituents backcast to their available annual "
            "financial history"
        ),
        "companies_requested": len(tickers),
        "companies_processed": len(company_results),
        "history_window": "latest up to 5 available fiscal years",
        "notes": [
            (
                "Sector ratios are aggregated from numerator/denominator totals, "
                "not simple averages of company ratios."
            ),
            (
                "Historical market cap uses fiscal-date close multiplied by "
                "reported shares outstanding when available; current shares are a fallback."
            ),
            (
                "Financials excludes corporate-style net debt/EBITDA, FCF, ROIC "
                "and EBITDA margin as non-comparable; P/B, ROE and balance-sheet "
                "capital ratios are retained instead."
            ),
            (
                "Real Estate FFO is a proxy defined as net income plus depreciation "
                "and amortization; it does not adjust for property-sale gains/losses "
                "and is therefore not official NAREIT FFO/AFFO."
            ),
        ],
    }
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        summary_df[
            ["sector", "coverage_pct", "pe_ttm", "roic", "capex_cfo"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
