"""Unit tests for PerformanceMetrics module."""

from typing import Any, cast
import numpy as np
import pandas as pd
import pytest

from src.metrics import PerformanceMetrics


@pytest.fixture
def sample_portfolio_results() -> pd.DataFrame:
    """Fixture providing synthetic backtest output with equity curve."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D", tz="UTC")
    # Portfolio values: 10000 -> 11000 -> 9000 -> 12000 ...
    equity = [10000.0, 10500.0, 11000.0, 10000.0, 9000.0, 9500.0, 11000.0, 12000.0, 11500.0, 12500.0]
    prices = [100.0, 102.0, 105.0, 103.0, 98.0, 101.0, 110.0, 115.0, 112.0, 120.0]
    
    df = pd.DataFrame(
        {
            "close": prices,
            "total_portfolio_value": equity,
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


def test_metrics_init_validation(sample_portfolio_results):
    """Test validation logic in PerformanceMetrics initialization."""
    # Non-DataFrame input
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        PerformanceMetrics(cast(Any, "invalid_input"))

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        PerformanceMetrics(pd.DataFrame())

    # Missing column
    df_missing = sample_portfolio_results.drop(columns=["total_portfolio_value"])
    with pytest.raises(ValueError, match="missing required column"):
        PerformanceMetrics(df_missing)

    # Invalid initial capital
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        PerformanceMetrics(sample_portfolio_results, initial_capital=0.0)


def test_cumulative_return_calculation(sample_portfolio_results):
    """Test calculation of cumulative return percentage."""
    metrics = PerformanceMetrics(sample_portfolio_results, initial_capital=10000.0)
    cum_ret = metrics.calculate_cumulative_return()
    # Final value: 12500.0, Initial: 10000.0 -> +25.0%
    assert cum_ret == pytest.approx(25.0)


def test_buy_and_hold_return_calculation(sample_portfolio_results):
    """Test calculation of Buy & Hold benchmark return percentage."""
    metrics = PerformanceMetrics(sample_portfolio_results, initial_capital=10000.0)
    bnh = metrics.calculate_buy_and_hold_return()
    # Initial close = 100.0, Final close = 120.0 -> +20.0%
    assert bnh == pytest.approx(20.0)


def test_buy_and_hold_missing_close():
    """Test Buy & Hold benchmark return behavior when 'close' column is missing."""
    dates = pd.date_range(start="2026-01-01", periods=5, freq="D", tz="UTC")
    df_no_close = pd.DataFrame({"total_portfolio_value": [10000.0] * 5}, index=dates)
    metrics = PerformanceMetrics(df_no_close, initial_capital=10000.0)
    assert metrics.calculate_buy_and_hold_return() == 0.0


def test_max_drawdown_calculation(sample_portfolio_results):
    """Test calculation of maximum drawdown percentage."""
    metrics = PerformanceMetrics(sample_portfolio_results, initial_capital=10000.0)
    mdd = metrics.calculate_max_drawdown()
    # Peak was 11000.0 on bar 2, dropped to 9000.0 on bar 4.
    # Drawdown = (9000 - 11000) / 11000 = -2000 / 11000 = -18.1818...%
    expected_mdd = ((9000.0 - 11000.0) / 11000.0) * 100.0
    assert mdd == pytest.approx(expected_mdd)


def test_sharpe_ratio_calculation(sample_portfolio_results):
    """Test Sharpe ratio calculation for non-zero variance returns."""
    metrics = PerformanceMetrics(sample_portfolio_results, initial_capital=10000.0)
    sharpe = metrics.calculate_sharpe_ratio(risk_free_rate=0.0, periods_per_year=365)
    assert isinstance(sharpe, float)
    assert sharpe > 0.0  # Overall positive trending portfolio equity curve


def test_sharpe_ratio_zero_variance():
    """Test Sharpe ratio behavior when portfolio returns have zero standard deviation."""
    dates = pd.date_range(start="2026-01-01", periods=5, freq="D", tz="UTC")
    df_flat = pd.DataFrame({"total_portfolio_value": [10000.0] * 5}, index=dates)

    metrics = PerformanceMetrics(df_flat, initial_capital=10000.0)
    sharpe = metrics.calculate_sharpe_ratio()
    assert sharpe == 0.0


def test_get_summary(sample_portfolio_results):
    """Test get_summary output format and keys."""
    metrics = PerformanceMetrics(sample_portfolio_results, initial_capital=10000.0)
    summary = metrics.get_summary()

    assert "cumulative_return_pct" in summary
    assert "annualized_sharpe" in summary
    assert "max_drawdown_pct" in summary
    assert "buy_and_hold_return_pct" in summary

    assert summary["cumulative_return_pct"] == pytest.approx(25.0)
    assert summary["buy_and_hold_return_pct"] == pytest.approx(20.0)
