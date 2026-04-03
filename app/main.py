"""Streamlit entrypoint: stock total return comparison."""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/main.py` puts `app/` on sys.path; `app.*` imports need repo root.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.services.alignment import (
    build_aligned_frames,
    default_date_range_today,
    parse_tickers,
)
from app.services.data_loader import load_ticker_history
from app.services.return_engine import compute_ticker_analysis, inner_join_series_for_chart


INITIAL = 10_000.0


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_load_ticker_history(
    ticker: str,
    start_iso: str,
    end_iso: str,
) -> tuple[object, object]:
    """Hashable cache key via ISO strings; returns DataFrame and Series."""
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    h, d = load_ticker_history(ticker, start, end)
    return h, d


def _loader(sym: str, start: pd.Timestamp, end: pd.Timestamp):
    h, d = _cached_load_ticker_history(sym, start.isoformat(), end.isoformat())
    return h, d


st.set_page_config(
    page_title="Total Return Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Total Return Comparator")
st.caption(
    "Compare hypothetical **$10,000** positions per ticker using Yahoo Finance. "
    "Reinvested total return uses **adjusted close** as a total-return proxy."
)

default_start, default_end = default_date_range_today()

with st.sidebar:
    st.subheader("Date range")
    start_date = st.date_input(
        "Start date",
        value=default_start.date(),
        help="Defaults to 36 months before end date.",
    )
    end_date = st.date_input("End date", value=default_end.date(), help="Defaults to today.")

req_start = pd.Timestamp(start_date)
req_end = pd.Timestamp(end_date)

tickers_text = st.text_area(
    "Tickers (comma or newline separated)",
    value="AAPL\nMSFT",
    height=120,
    help="Enter as many symbols as you want; duplicates are removed.",
)

run = st.button("Run analysis", type="primary")

if run:
    tickers = parse_tickers(tickers_text)
    if not tickers:
        st.warning("Enter at least one ticker.")
    else:
        with st.spinner("Loading market data…"):
            try:
                hist_by, div_by, actual_start, actual_end, align_warnings = build_aligned_frames(
                    tickers,
                    req_start,
                    req_end,
                    loader=_loader,
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        for w in align_warnings:
            if w.severity == "warning":
                st.warning(w.message)
            else:
                st.info(w.message)

        results: list = []
        for sym in tickers:
            h = hist_by.get(sym)
            d = div_by.get(sym, pd.Series(dtype=float))
            if h is None or h.empty:
                st.warning(f"{sym}: skipped (no data).")
                continue
            r = compute_ticker_analysis(
                sym,
                h,
                d,
                req_start,
                req_end,
                actual_start,
                actual_end,
                initial_investment=INITIAL,
            )
            for w in r.warnings:
                if w.severity == "warning":
                    st.warning(w.message)
                else:
                    st.info(w.message)
            results.append(r)

        if not results:
            st.error("No results to display.")
            st.stop()

        st.subheader("Summary")
        rows = []
        for r in results:
            s = r.summary
            rows.append(
                {
                    "Ticker": r.ticker,
                    "Window": f"{r.actual_start.date()} → {r.actual_end.date()}",
                    "End value (price only)": s.get("end_value_price_only"),
                    "End value (reinvested)": s.get("end_value_reinvested"),
                    "End value (no reinvest)": s.get("end_value_no_reinvest"),
                    "Return % (price)": s.get("return_pct_price_only"),
                    "Return % (reinvested)": s.get("return_pct_reinvested"),
                    "Return % (no reinvest)": s.get("return_pct_no_reinvest"),
                    "Total dividend cash": s.get("total_dividend_cash"),
                }
            )
        summary_df = pd.DataFrame(rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            column_config={
                "End value (price only)": st.column_config.NumberColumn(format="$%.2f"),
                "End value (reinvested)": st.column_config.NumberColumn(format="$%.2f"),
                "End value (no reinvest)": st.column_config.NumberColumn(format="$%.2f"),
                "Return % (price)": st.column_config.NumberColumn(format="%.2f%%"),
                "Return % (reinvested)": st.column_config.NumberColumn(format="%.2f%%"),
                "Return % (no reinvest)": st.column_config.NumberColumn(format="%.2f%%"),
                "Total dividend cash": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

        tab1, tab2, tab3 = st.tabs(
            ["Price only", "Total return (reinvested, adj. close)", "No reinvestment (price + cash)"]
        )

        def chart_for(mode: str, title: str):
            df = inner_join_series_for_chart(results, mode=mode)
            if df.empty:
                st.info("No overlapping dates to plot.")
                return
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode="lines",
                        name=col,
                    )
                )
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Value ($)",
                hovermode="x unified",
                legend_title="Ticker",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab1:
            chart_for("price_only", "Position value — price only (no dividends)")
        with tab2:
            chart_for("reinvested", "Total return — reinvested (adjusted close proxy)")
        with tab3:
            chart_for("no_reinvest", "No reinvestment — stock value + cumulative dividend cash")

        st.subheader("Dividend schedules (non-reinvested basis)")
        for r in results:
            with st.expander(f"{r.ticker} — dividends"):
                if r.dividend_schedule is not None and not r.dividend_schedule.empty:
                    disp = r.dividend_schedule.copy()
                    st.dataframe(
                        disp,
                        use_container_width=True,
                        column_config={
                            "DividendPerShare": st.column_config.NumberColumn(format="%.4f"),
                            "Shares": st.column_config.NumberColumn(format="%.6f"),
                            "CashPaid": st.column_config.NumberColumn(format="$%.2f"),
                            "CumulativeCash": st.column_config.NumberColumn(format="$%.2f"),
                        },
                    )
                else:
                    st.caption("No dividend rows in this window.")

else:
    st.info("Enter tickers and click **Run analysis**.")
