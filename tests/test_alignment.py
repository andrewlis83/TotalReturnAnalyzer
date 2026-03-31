"""Unit tests for alignment helpers (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.models.schemas import Warning
from app.services.alignment import (
    compute_common_window,
    default_date_range_today,
    parse_tickers,
)


def test_parse_tickers():
    assert parse_tickers("aapl, msft\nGOOG") == ["AAPL", "MSFT", "GOOG"]
    assert parse_tickers("dup, DUP, x") == ["DUP", "X"]


def test_default_date_range():
    s, e = default_date_range_today()
    assert e >= s
    delta = (e - s).days
    assert 365 * 3 - 10 < delta < 365 * 3 + 120  # ~36 months calendar slack


def test_compute_common_window_truncates():
    idx1 = pd.bdate_range("2023-01-03", periods=10)
    idx2 = pd.bdate_range("2023-01-10", periods=5)
    h1 = pd.DataFrame({"Close": [1] * 10, "Adj Close": [1] * 10}, index=idx1)
    h2 = pd.DataFrame({"Close": [2] * 5, "Adj Close": [2] * 5}, index=idx2)
    ticker_hist = {"A": h1, "B": h2}
    req_s = pd.Timestamp("2023-01-01")
    req_e = pd.Timestamp("2023-12-31")
    actual_start, actual_end, warns = compute_common_window(ticker_hist, req_s, req_e)
    assert actual_start == idx2[0].normalize()
    assert any("moved" in w.message.lower() for w in warns if isinstance(w, Warning))


def test_start_after_end_raises():
    with pytest.raises(ValueError):
        compute_common_window({}, pd.Timestamp("2024-01-01"), pd.Timestamp("2020-01-01"))
