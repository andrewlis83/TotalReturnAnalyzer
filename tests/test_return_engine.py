"""Unit tests for return_engine (synthetic data, no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.return_engine import compute_ticker_analysis, inner_join_series_for_chart


def _sample_hist(
    start: str,
    n_days: int,
    close_start: float = 100.0,
    daily_growth: float = 0.001,
) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n_days)
    close = [close_start * (1 + daily_growth) ** i for i in range(n_days)]
    # Adj close tracks total return slightly higher (div reinvestment proxy)
    adj = [c * 1.0001 * (1 + daily_growth * 0.5) ** i for i, c in enumerate(close)]
    return pd.DataFrame({"Close": close, "Adj Close": adj}, index=idx)


def test_price_and_reinvested_scale():
    hist = _sample_hist("2022-01-03", 10, close_start=50.0)
    div = pd.Series(dtype=float)
    req_s = pd.Timestamp("2022-01-01")
    req_e = pd.Timestamp("2022-12-31")
    act_s = hist.index[0]
    act_e = hist.index[-1]
    r = compute_ticker_analysis(
        "TEST",
        hist,
        div,
        req_s,
        req_e,
        act_s,
        act_e,
        initial_investment=10_000.0,
    )
    assert r.summary["shares_held"] == pytest.approx(10_000.0 / 50.0)
    assert r.price_only_value.iloc[0] == pytest.approx(10_000.0, rel=1e-9)
    assert r.reinvested_value.iloc[0] == pytest.approx(10_000.0, rel=1e-9)


def test_dividend_cash_accumulates():
    idx = pd.bdate_range("2022-01-03", periods=5)
    hist = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "Adj Close": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )
    # Ex-div on second trading day
    div = pd.Series([1.0], index=[idx[1]])
    r = compute_ticker_analysis(
        "DIV",
        hist,
        div,
        idx[0],
        idx[-1],
        idx[0],
        idx[-1],
        initial_investment=10_000.0,
    )
    shares = 10_000.0 / 100.0
    assert r.summary["total_dividend_cash"] == pytest.approx(shares * 1.0)
    # Last day: stock value + full cash (paid on day 2, held as cash)
    assert r.no_reinvest_total_value.iloc[-1] == pytest.approx(shares * 100.0 + shares * 1.0)


def test_inner_join_chart():
    idx = pd.bdate_range("2022-01-03", periods=4)
    h1 = pd.DataFrame({"Close": [10, 11, 12, 13], "Adj Close": [10, 11, 12, 13]}, index=idx)
    h2 = pd.DataFrame({"Close": [20, 21, 22, 23], "Adj Close": [20, 21, 22, 23]}, index=idx)
    r1 = compute_ticker_analysis("A", h1, pd.Series(dtype=float), idx[0], idx[-1], idx[0], idx[-1])
    r2 = compute_ticker_analysis("B", h2, pd.Series(dtype=float), idx[0], idx[-1], idx[0], idx[-1])
    df = inner_join_series_for_chart([r1, r2], mode="price_only")
    assert not df.empty
    assert list(df.columns) == ["A", "B"]
