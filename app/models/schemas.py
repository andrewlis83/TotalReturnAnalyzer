"""Typed containers for analysis outputs and warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Warning:
    """User-visible warning about data quality or truncation."""

    severity: str  # "info" | "warning"
    message: str


@dataclass
class AlignedData:
    """Per-ticker OHLCV + dividends aligned to a common date index."""

    ticker: str
    dates: pd.DatetimeIndex
    close: pd.Series
    adj_close: pd.Series
    dividends: pd.Series  # daily dividend amount (0 on non-div days), aligned to dates


@dataclass
class AnalysisResult:
    """Computed return series and summary for one ticker."""

    ticker: str
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    actual_start: pd.Timestamp
    actual_end: pd.Timestamp
    initial_investment: float
    # Indexed by date (aligned trading days)
    price_only_value: pd.Series
    reinvested_value: pd.Series
    no_reinvest_total_value: pd.Series
    cumulative_dividend_cash: pd.Series
    # Dividend schedule: rows with payment date, div per share, cash, running total
    dividend_schedule: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[Warning] = field(default_factory=list)
