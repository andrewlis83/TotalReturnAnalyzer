"""Price return, adjusted-close total return, and non-reinvested dividend cashflow."""

from __future__ import annotations

import pandas as pd

from app.models.schemas import AnalysisResult, Warning


def _assign_dividend_to_trading_day(
    trading_index: pd.DatetimeIndex,
    div_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Map ex-div date to a trading day in index (same day if present, else next session)."""
    div_date = pd.Timestamp(div_date).normalize()
    if div_date > trading_index[-1]:
        return None
    if div_date < trading_index[0]:
        return None
    if div_date in trading_index:
        return div_date
    pos = trading_index.searchsorted(div_date)
    if pos >= len(trading_index):
        return None
    return trading_index[pos]


def compute_ticker_analysis(
    ticker: str,
    hist: pd.DataFrame,
    div: pd.Series,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    actual_start: pd.Timestamp,
    actual_end: pd.Timestamp,
    initial_investment: float = 10_000.0,
) -> AnalysisResult:
    """
    Compute price-only, reinvested (adj close), and non-reinvest total value series.

    Requires non-empty hist with Close and Adj Close for all rows in window.
    """
    warnings: list[Warning] = []
    if hist.empty or "Close" not in hist.columns or "Adj Close" not in hist.columns:
        return AnalysisResult(
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
            actual_start=actual_start,
            actual_end=actual_end,
            initial_investment=initial_investment,
            price_only_value=pd.Series(dtype=float),
            reinvested_value=pd.Series(dtype=float),
            no_reinvest_total_value=pd.Series(dtype=float),
            cumulative_dividend_cash=pd.Series(dtype=float),
            dividend_schedule=pd.DataFrame(),
            summary={},
            warnings=[Warning("warning", f"{ticker}: insufficient price data.")],
        )

    hist = hist.sort_index()
    idx = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in hist.index])

    close = hist["Close"].astype(float)
    adj = hist["Adj Close"].astype(float)

    # Data quality: gaps
    if close.isna().any() or adj.isna().any():
        warnings.append(
            Warning(
                "warning",
                f"{ticker}: missing Close/Adj Close on some days; rows dropped.",
            )
        )
    valid = close.notna() & adj.notna() & (close > 0) & (adj > 0)
    if not valid.all():
        hist = hist.loc[valid]
        close = hist["Close"].astype(float)
        adj = hist["Adj Close"].astype(float)
        idx = pd.DatetimeIndex([pd.Timestamp(x).normalize() for x in hist.index])

    if hist.empty:
        return AnalysisResult(
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
            actual_start=actual_start,
            actual_end=actual_end,
            initial_investment=initial_investment,
            price_only_value=pd.Series(dtype=float),
            reinvested_value=pd.Series(dtype=float),
            no_reinvest_total_value=pd.Series(dtype=float),
            cumulative_dividend_cash=pd.Series(dtype=float),
            dividend_schedule=pd.DataFrame(),
            summary={},
            warnings=[Warning("warning", f"{ticker}: no valid rows after cleaning.")],
        )

    close0 = float(close.iloc[0])
    adj0 = float(adj.iloc[0])
    shares0 = initial_investment / close0

    price_only_value = (shares0 * close).copy()
    price_only_value.index = idx
    price_only_value.name = "price_only"

    reinvested_value = (initial_investment * (adj / adj0)).copy()
    reinvested_value.index = idx
    reinvested_value.name = "reinvested"

    # Per-day dividend cash assigned to trading days
    cash_on_day = pd.Series(0.0, index=idx)
    if div is not None and not div.empty:
        for div_date, per_share_raw in div.items():
            per_share = float(per_share_raw)
            if pd.isna(per_share) or per_share <= 0:
                continue
            cash_amt = shares0 * per_share
            mapped = _assign_dividend_to_trading_day(idx, pd.Timestamp(div_date))
            if mapped is None:
                warnings.append(
                    Warning(
                        "warning",
                        f"{ticker}: dividend on {div_date} could not be mapped to a session in range.",
                    )
                )
                continue
            cash_on_day.loc[mapped] = cash_on_day.loc[mapped] + cash_amt

    cumulative_dividend_cash = cash_on_day.cumsum()
    cumulative_dividend_cash.name = "cum_div_cash"

    no_reinvest_total = (shares0 * close).values + cumulative_dividend_cash.values
    no_reinvest_total_value = pd.Series(no_reinvest_total, index=idx, name="no_reinvest_total")

    # Dividend schedule: one row per dividend event (chronological)
    sched_rows: list[dict] = []
    running = 0.0
    if div is not None and not div.empty:
        for div_date, per_share_raw in div.sort_index().items():
            per_share = float(per_share_raw)
            if pd.isna(per_share) or per_share <= 0:
                continue
            mapped = _assign_dividend_to_trading_day(idx, pd.Timestamp(div_date))
            if mapped is None:
                continue
            cash_paid = shares0 * per_share
            running += cash_paid
            sched_rows.append(
                {
                    "PaymentDate": mapped,
                    "ExDivDate": pd.Timestamp(div_date).normalize(),
                    "DividendPerShare": per_share,
                    "Shares": shares0,
                    "CashPaid": cash_paid,
                    "CumulativeCash": running,
                }
            )

    dividend_schedule = pd.DataFrame(sched_rows)
    total_div_cash = float(cumulative_dividend_cash.iloc[-1]) if len(cumulative_dividend_cash) else 0.0

    def pct_change(end_v: float) -> float:
        return (end_v / initial_investment - 1.0) * 100.0 if initial_investment else 0.0

    end_price = float(price_only_value.iloc[-1])
    end_reinv = float(reinvested_value.iloc[-1])
    end_nr = float(no_reinvest_total_value.iloc[-1])

    summary = {
        "end_value_price_only": end_price,
        "end_value_reinvested": end_reinv,
        "end_value_no_reinvest": end_nr,
        "return_pct_price_only": pct_change(end_price),
        "return_pct_reinvested": pct_change(end_reinv),
        "return_pct_no_reinvest": pct_change(end_nr),
        "total_dividend_cash": total_div_cash,
        "shares_held": shares0,
    }

    return AnalysisResult(
        ticker=ticker,
        requested_start=requested_start,
        requested_end=requested_end,
        actual_start=actual_start,
        actual_end=actual_end,
        initial_investment=initial_investment,
        price_only_value=price_only_value,
        reinvested_value=reinvested_value,
        no_reinvest_total_value=no_reinvest_total_value,
        cumulative_dividend_cash=cumulative_dividend_cash,
        dividend_schedule=dividend_schedule,
        summary=summary,
        warnings=warnings,
    )


def inner_join_series_for_chart(
    results: list[AnalysisResult],
    mode: str = "reinvested",
) -> pd.DataFrame:
    """
    Build a single DataFrame of aligned dates (inner join across tickers).

    mode: 'price_only' | 'reinvested' | 'no_reinvest'
    """
    series_map: dict[str, pd.Series] = {}
    for r in results:
        if mode == "price_only":
            s = r.price_only_value
        elif mode == "reinvested":
            s = r.reinvested_value
        else:
            s = r.no_reinvest_total_value
        if s is None or s.empty:
            continue
        series_map[r.ticker] = s

    if not series_map:
        return pd.DataFrame()

    # Inner join on date index
    df = pd.concat(series_map.values(), axis=1, keys=series_map.keys(), join="inner")
    return df
