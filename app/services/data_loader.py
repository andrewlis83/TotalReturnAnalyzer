"""Yahoo Finance fetch wrappers and normalization."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_ticker_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load daily OHLCV and dividends for [start, end] (inclusive end for display).

    yfinance `history` end parameter is exclusive; we extend end by one calendar day.

    Returns:
        hist: DataFrame with Open, High, Low, Close, Adj Close, Volume
        dividends: Series indexed by date, dividend cash amount per share on ex-div dates
    """
    ticker = ticker.strip().upper()
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    end_fetch = end + pd.Timedelta(days=1)

    t = yf.Ticker(ticker)
    try:
        hist = t.history(start=start, end=end_fetch, auto_adjust=False, repair=True)
    except ModuleNotFoundError as exc:
        # yfinance repair path imports scipy; gracefully fall back if unavailable.
        if exc.name != "scipy":
            raise
        hist = t.history(start=start, end=end_fetch, auto_adjust=False, repair=False)
    if not hist.empty and getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_convert(None)

    div = t.dividends
    if not div.empty and getattr(div.index, "tz", None) is not None:
        div.index = div.index.tz_convert(None)
        div = div[(div.index >= start) & (div.index <= end)].sort_index()

    if hist.empty:
        return hist, div if isinstance(div, pd.Series) else pd.Series(dtype=float)

    # Normalize column names
    col_map = {c: c.title() for c in hist.columns}
    hist = hist.rename(columns=col_map)
    if "Adj Close" not in hist.columns:
        for c in list(hist.columns):
            if str(c).lower().replace(" ", "") == "adjclose":
                hist = hist.rename(columns={c: "Adj Close"})
                break

    return hist, div if isinstance(div, pd.Series) else pd.Series(dtype=float)


def first_valid_price_row(hist: pd.DataFrame) -> pd.Timestamp | None:
    """First date with non-null Close and Adj Close and positive close."""
    if hist.empty:
        return None
    if "Close" not in hist.columns or "Adj Close" not in hist.columns:
        return None
    mask = hist["Close"].notna() & hist["Adj Close"].notna() & (hist["Close"] > 0)
    if not mask.any():
        return None
    return pd.Timestamp(hist.index[mask][0]).normalize()


def last_valid_price_row(hist: pd.DataFrame) -> pd.Timestamp | None:
    if hist.empty:
        return None
    if "Close" not in hist.columns or "Adj Close" not in hist.columns:
        return None
    mask = hist["Close"].notna() & hist["Adj Close"].notna() & (hist["Close"] > 0)
    if not mask.any():
        return None
    return pd.Timestamp(hist.index[mask][-1]).normalize()
