from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Config:
    start_date: str = "2022-01-01"
    end_date: str = "2025-12-31"
    top_n: int = 5
    initial_capital: float = 1_000_000
    commission_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage_rate: float = 0.001
    risk_free_rate: float = 0.0
    random_seed: int = 42


def generate_sample_data(config: Config):
    rng = np.random.default_rng(config.random_seed)

    tickers = [
        "2330", "2317", "2454", "2308", "2303",
        "2382", "3711", "2881", "2882", "2891"
    ]

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    price_dates = pd.bdate_range(start, end)
    months = pd.period_range(
        start=start.to_period("M"),
        end=end.to_period("M"),
        freq="M"
    )

    price_rows = []
    base_prices = np.linspace(80, 700, len(tickers))
    annual_drifts = rng.uniform(0.04, 0.18, len(tickers))
    annual_vols = rng.uniform(0.18, 0.38, len(tickers))

    for i, ticker in enumerate(tickers):
        daily_returns = (
            annual_drifts[i] / 252
            + annual_vols[i] / np.sqrt(252)
            * rng.normal(size=len(price_dates))
        )
        prices = base_prices[i] * np.cumprod(1 + daily_returns)

        for date, price in zip(price_dates, prices):
            price_rows.append((date, ticker, float(price)))

    prices = pd.DataFrame(
        price_rows,
        columns=["date", "ticker", "close"]
    )

    revenue_rows = []

    for i, ticker in enumerate(tickers):
        level = rng.uniform(1000, 10000)
        growth = rng.uniform(-0.015, 0.025)

        for period in months:
            seasonal = 1 + 0.08 * np.sin(
                2 * np.pi * period.month / 12
            )
            noise = rng.normal(0, 0.035)

            revenue = level * seasonal * (
                1 + growth
            ) ** (
                (period.year - start.year) * 12
                + period.month - start.month
            )

            revenue *= max(0.5, 1 + noise)

            period_end = period.to_timestamp(
                how="end"
            ).normalize()

            available_date = (
                period_end
                + pd.offsets.MonthBegin(1)
                + pd.Timedelta(days=9)
            )

            revenue_rows.append(
                (ticker, period_end, available_date, float(revenue))
            )

    revenues = pd.DataFrame(
        revenue_rows,
        columns=[
            "ticker",
            "period_end",
            "available_date",
            "revenue"
        ]
    )

    quarter_rows = []

    quarters = pd.period_range(
        start=start.to_period("Q"),
        end=end.to_period("Q"),
        freq="Q"
    )

    for i, ticker in enumerate(tickers):
        book_value = rng.uniform(30, 300)

        for q in quarters:
            q_end = q.end_time.normalize()

            if q.quarter == 1:
                available = pd.Timestamp(q.year, 5, 15)
            elif q.quarter == 2:
                available = pd.Timestamp(q.year, 8, 14)
            elif q.quarter == 3:
                available = pd.Timestamp(q.year, 11, 14)
            else:
                available = pd.Timestamp(q.year + 1, 3, 31)

            book_value *= 1 + rng.normal(0.025, 0.04)

            quarter_rows.append(
                (ticker, q_end, available, float(max(book_value, 1)))
            )

    quarterly = pd.DataFrame(
        quarter_rows,
        columns=[
            "ticker",
            "period_end",
            "available_date",
            "book_value"
        ]
    )

    return prices, revenues, quarterly, tickers


def build_pit_snapshot(
    as_of: pd.Timestamp,
    prices: pd.DataFrame,
    revenues: pd.DataFrame,
    quarterly: pd.DataFrame,
    tickers: List[str],
) -> pd.DataFrame:

    as_of = pd.Timestamp(as_of)

    price_now = (
        prices.loc[prices["date"] <= as_of]
        .sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "date", "close"]]
    )

    rev_available = revenues.loc[
        revenues["available_date"] <= as_of
    ].copy()

    rev_available = rev_available.sort_values(
        ["ticker", "period_end"]
    )

    rev_rows = []

    for ticker in tickers:
        hist = rev_available.loc[
            rev_available["ticker"] == ticker
        ]

        if hist.empty:
            rev_rows.append((ticker, np.nan))
            continue

        latest = hist.iloc[-1]

        prior_year = hist.loc[
            hist["period_end"].dt.to_period("M")
            == (
                latest["period_end"].to_period("M") - 12
            )
        ]

        if prior_year.empty:
            yoy = np.nan
        else:
            prior_revenue = float(
                prior_year.iloc[-1]["revenue"]
            )

            yoy = (
                float(latest["revenue"]) / prior_revenue - 1
                if prior_revenue
                else np.nan
            )

        rev_rows.append((ticker, yoy))

    rev_factor = pd.DataFrame(
        rev_rows,
        columns=["ticker", "revenue_yoy"]
    )

    q_available = quarterly.loc[
        quarterly["available_date"] <= as_of
    ].copy()

    q_latest = (
        q_available.sort_values(["ticker", "period_end"])
        .groupby("ticker")
        .tail(1)[["ticker", "book_value"]]
    )

    snapshot = (
        pd.DataFrame({"ticker": tickers})
        .merge(
            price_now[["ticker", "close"]],
            on="ticker",
            how="left"
        )
        .merge(
            rev_factor,
            on="ticker",
            how="left"
        )
        .merge(
            q_latest,
            on="ticker",
            how="left"
        )
    )

    snapshot["pb"] = (
        snapshot["close"] / snapshot["book_value"]
    )

    return snapshot


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)

    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)

    return (series - mean) / std


def calculate_factor_scores(
    snapshot: pd.DataFrame
) -> pd.DataFrame:

    result = snapshot.copy()

    result["revenue_yoy"] = result[
        "revenue_yoy"
    ].replace([np.inf, -np.inf], np.nan)

    result["pb"] = result[
        "pb"
    ].replace([np.inf, -np.inf], np.nan)

    result["revenue_yoy"] = result[
        "revenue_yoy"
    ].fillna(result["revenue_yoy"].median())

    result["pb"] = result[
        "pb"
    ].fillna(result["pb"].median())

    result["revenue_score"] = zscore(
        result["revenue_yoy"]
    )

    result["pb_score"] = -zscore(result["pb"])

    result["total_score"] = (
        0.5 * result["revenue_score"]
        + 0.5 * result["pb_score"]
    )

    return result


def select_portfolio(
    scores: pd.DataFrame,
    top_n: int
) -> Dict[str, float]:

    eligible = scores.dropna(
        subset=["close", "total_score"]
    ).copy()

    selected = eligible.sort_values(
        "total_score",
        ascending=False
    ).head(top_n)

    if selected.empty:
        return {}

    weight = 1.0 / len(selected)

    return {
        ticker: weight
        for ticker in selected["ticker"]
    }


def calculate_turnover(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float]
) -> Tuple[float, float]:

    all_tickers = set(old_weights) | set(new_weights)

    buy = 0.0
    sell = 0.0

    for ticker in all_tickers:
        change = (
            new_weights.get(ticker, 0.0)
            - old_weights.get(ticker, 0.0)
        )

        if change > 0:
            buy += change
        elif change < 0:
            sell += abs(change)

    return buy, sell


def calculate_performance(
    equity: pd.Series,
    risk_free_rate: float = 0.0
) -> dict:

    equity = equity.dropna()
    returns = equity.pct_change().dropna()

    if len(equity) < 2 or returns.empty:
        return {
            "累積報酬率": 0.0,
            "年化報酬率": 0.0,
            "年化波動率": 0.0,
            "夏普比率": 0.0,
            "最大回撤": 0.0,
        }

    total_return = (
        equity.iloc[-1] / equity.iloc[0] - 1
    )

    years = len(returns) / 12

    annual_return = (
        (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        if years > 0
        else 0.0
    )

    annual_vol = (
        returns.std(ddof=1) * np.sqrt(12)
        if len(returns) > 1
        else 0.0
    )

    excess = returns - risk_free_rate / 12

    sharpe = (
        excess.mean() / excess.std(ddof=1) * np.sqrt(12)
        if len(excess) > 1
        and excess.std(ddof=1) != 0
        else 0.0
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "累積報酬率": float(total_return),
        "年化報酬率": float(annual_return),
        "年化波動率": float(annual_vol),
        "夏普比率": float(sharpe),
        "最大回撤": float(drawdown.min()),
    }


def run_backtest(config: Config):

    prices, revenues, quarterly, tickers = (
        generate_sample_data(config)
    )

    monthly_dates = (
        prices.assign(
            month=prices["date"].dt.to_period("M")
        )
        .groupby("month")["date"]
        .min()
        .sort_values()
        .tolist()
    )

    equity = config.initial_capital
    old_weights: Dict[str, float] = {}

    equity_records = []
    holdings_records = []

    for i, rebalance_date in enumerate(monthly_dates):

        if i == len(monthly_dates) - 1:
            break

        next_date = monthly_dates[i + 1]

        signal_date = (
            pd.Timestamp(rebalance_date)
            - pd.Timedelta(days=1)
        )

        snapshot = build_pit_snapshot(
            signal_date,
            prices,
            revenues,
            quarterly,
            tickers
        )

        scores = calculate_factor_scores(snapshot)

        new_weights = select_portfolio(
            scores,
            config.top_n
        )

        if not new_weights:
            continue

        buy_turnover, sell_turnover = calculate_turnover(
            old_weights,
            new_weights
        )

        buy_cost_rate = (
            config.commission_rate
            + config.slippage_rate
        )

        sell_cost_rate = (
            config.commission_rate
            + config.tax_rate
            + config.slippage_rate
        )

        transaction_cost = equity * (
            buy_turnover * buy_cost_rate
            + sell_turnover * sell_cost_rate
        )

        equity -= transaction_cost

        period_prices = prices.loc[
            prices["date"].isin([rebalance_date, next_date])
            & prices["ticker"].isin(new_weights)
        ]

        price_pivot = period_prices.pivot(
            index="date",
            columns="ticker",
            values="close"
        )

        portfolio_return = 0.0
        valid_weight = 0.0

        for ticker, weight in new_weights.items():

            if (
                rebalance_date in price_pivot.index
                and next_date in price_pivot.index
                and ticker in price_pivot.columns
            ):

                p0 = price_pivot.loc[rebalance_date, ticker]
                p1 = price_pivot.loc[next_date, ticker]

                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    portfolio_return += (
                        weight * (p1 / p0 - 1)
                    )

                    valid_weight += weight

        if valid_weight > 0:
            portfolio_return /= valid_weight

        equity *= 1 + portfolio_return

        equity_records.append((next_date, equity))

        holdings_records.append({
            "調倉日": rebalance_date,
            "訊號日": signal_date,
            "持有股票": ", ".join(new_weights.keys()),
            "持股數": len(new_weights),
            "買進週轉率": buy_turnover,
            "賣出週轉率": sell_turnover,
            "交易成本": transaction_cost,
            "期間報酬率": portfolio_return,
            "期末資產": equity,
        })

        old_weights = new_weights.copy()

    equity_curve = pd.Series(
        [config.initial_capital]
        + [x[1] for x in equity_records],
        index=[
            pd.Timestamp(config.start_date)
        ] + [x[0] for x in equity_records],
        name="portfolio_value",
    )

    equity_curve = equity_curve[
        ~equity_curve.index.duplicated(keep="last")
    ].sort_index()

    performance = calculate_performance(
        equity_curve,
        config.risk_free_rate
    )

    holdings = pd.DataFrame(holdings_records)

    return performance, equity_curve, holdings


def run_self_checks():

    sample = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "close": [100.0, 100.0, 100.0],
        "revenue_yoy": [0.10, 0.05, 0.00],
        "pb": [1.0, 2.0, 3.0],
    })

    scored = calculate_factor_scores(sample)

    portfolio = select_portfolio(
        scored,
        top_n=2
    )

    assert len(portfolio) == 2
    assert math.isclose(sum(portfolio.values()), 1.0)

    buy, sell = calculate_turnover(
        {"A": 0.5, "B": 0.5},
        {"A": 0.5, "C": 0.5}
    )

    assert math.isclose(buy, 0.5)
    assert math.isclose(sell, 0.5)

    test_revenue = pd.DataFrame({
        "ticker": ["A"],
        "period_end": [pd.Timestamp("2024-01-31")],
        "available_date": [pd.Timestamp("2024-02-10")],
        "revenue": [100.0],
    })

    assert test_revenue.loc[
        test_revenue["available_date"]
        <= pd.Timestamp("2024-02-09")
    ].empty

    assert len(
        test_revenue.loc[
            test_revenue["available_date"]
            <= pd.Timestamp("2024-02-10")
        ]
    ) == 1

    print("Basic checks passed.")


def main():

    config = Config()

    run_self_checks()

    performance, equity_curve, holdings = run_backtest(
        config
    )

    print("\n========== Factor Strategy Backtest ==========")
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Top N: {config.top_n}")
    print("Factor weights: Revenue YoY 50% + PB 50%")
    print(f"Initial capital: {config.initial_capital:,.0f}")
    print("---------------------------------------------")

    for metric, value in performance.items():
        if metric == "夏普比率":
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value:.2%}")

    print("---------------------------------------------")
    print("\nLatest 5 rebalances:")

    if holdings.empty:
        print("No holdings records.")
    else:
        print(holdings.tail(5).to_string(index=False))

    equity_curve.rename_axis("date").reset_index().to_csv(
        "equity_curve.csv",
        index=False,
        encoding="utf-8-sig"
    )

    holdings.to_csv(
        "holdings.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nOutput files: equity_curve.csv, holdings.csv")


if __name__ == "__main__":
    main()
  