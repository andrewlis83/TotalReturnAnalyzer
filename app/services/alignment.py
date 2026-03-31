"""Date-range handling, common-start truncation, and warnings."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd

from app.models.schemas import Warning
from app.services.data_loader import (
    first_valid_price_row,
    last_valid_price_row,
    load_ticker_history,
)


def parse_tickers(text: str) -> list[str]:
    """Parse comma/newline-separated tickers; uppercase, strip, dedupe preserving order."""
    if not text or not str(text).strip():
        return []
    raw = []
    for part in str(text).replace(",", "\n").split("\n"):
        t = part.strip().upper()
        if t:
            raw.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def default_date_range_today() -> tuple[pd.Timestamp, pd.Timestamp]:
    """End = today (calendar), start = 36 months prior (calendar)."""
    end = pd.Timestamp(date.today())
    start = end - pd.DateOffset(months=36)
    return start.normalize(), end.normalize()


def compute_common_window(
    ticker_hist: dict[str, pd.DataFrame],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp, list[Warning]]:
    """
    Determine actual_start = max(requested_start, max(first_valid per ticker)).
    actual_end = min(requested_end, min(last_valid per ticker)).
    """
    warnings: list[Warning] = []
    req_s = pd.Timestamp(requested_start).normalize()
    req_e = pd.Timestamp(requested_end).normalize()

    if req_s > req_e:
        raise ValueError("Start date must be on or before end date.")

    firsts: list[pd.Timestamp] = []
    lasts: list[pd.Timestamp] = []
    for sym, hist in ticker_hist.items():
        if hist.empty:
            warnings.append(
                Warning(
                    "warning",
                    f"{sym}: no price history in the selected range.",
                )
            )
            continue
        f = first_valid_price_row(hist)
        l = last_valid_price_row(hist)
        if f is None or l is None:
            warnings.append(
                Warning(
                    "warning",
                    f"{sym}: missing Close/Adj Close in range.",
                )
            )
            continue
        firsts.append(pd.Timestamp(f).normalize())
        lasts.append(pd.Timestamp(l).normalize())

    if not firsts:
        return req_s, req_e, warnings

    latest_first = max(firsts)
    earliest_last = min(lasts)

    actual_start = max(req_s, latest_first)
    actual_end = min(req_e, earliest_last)

    if actual_start > actual_end:
        warnings.append(
            Warning(
                "warning",
                "No overlapping valid trading window for all tickers.",
            )
        )
        return actual_start, actual_end, warnings

    if latest_first > req_s:
        warnings.append(
            Warning(
                "info",
                f"Analysis start moved to {actual_start.date()} (latest first available "
                f"price among tickers) so all series are comparable.",
            )
        )

    if earliest_last < req_e:
        warnings.append(
            Warning(
                "info",
                f"Analysis end moved to {actual_end.date()} (earliest last available "
                f"price among tickers).",
            )
        )

    return actual_start, actual_end, warnings


def build_aligned_frames(
    tickers: list[str],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    loader: Callable[[str, pd.Timestamp, pd.Timestamp], tuple[pd.DataFrame, pd.Series]]
    | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], pd.Timestamp, pd.Timestamp, list[Warning]]:
    """
    Load each ticker, compute common window, re-slice history and dividends.

    Returns:
        hist_by_ticker, div_by_ticker, actual_start, actual_end, warnings
    """
    warnings: list[Warning] = []
    req_s = pd.Timestamp(requested_start).normalize()
    req_e = pd.Timestamp(requested_end).normalize()

    # Fetch with same calendar window (loader extends end by 1 day internally)
    hist_by: dict[str, pd.DataFrame] = {}
    div_by: dict[str, pd.Series] = {}

    load_fn = loader or load_ticker_history
    for sym in tickers:
        h, d = load_fn(sym, req_s, req_e)
        if h.empty:
            warnings.append(
                Warning("warning", f"{sym}: no data returned for the selected date range.")
            )
        hist_by[sym] = h
        div_by[sym] = d

    actual_start, actual_end, w2 = compute_common_window(hist_by, req_s, req_e)
    warnings.extend(w2)

    # Re-slice to common window for strict alignment
    aligned_h: dict[str, pd.DataFrame] = {}
    aligned_d: dict[str, pd.Series] = {}
    for sym in tickers:
        h = hist_by[sym]
        d = div_by[sym]
        if h.empty:
            aligned_h[sym] = h
            aligned_d[sym] = pd.Series(dtype=float)
            continue
        mask = (h.index >= actual_start) & (h.index <= actual_end)
        hh = h.loc[mask].copy()
        aligned_h[sym] = hh
        if not d.empty:
            dm = (d.index >= actual_start) & (d.index <= actual_end)
            aligned_d[sym] = d.loc[dm].copy()
        else:
            aligned_d[sym] = pd.Series(dtype=float)

    return aligned_h, aligned_d, actual_start, actual_end, warnings
