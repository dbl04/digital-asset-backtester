"""Unit tests for EventDrivenBacktester module."""

from typing import Any, cast
import numpy as np
import pandas as pd
import pytest

from src.backtester import EventDrivenBacktester


@pytest.fixture
def sample_signals_data() -> pd.DataFrame:
    """Fixture providing market data with explicit signal transitions."""
    dates = pd.date_range(start="2026-01-01", periods=10, freq="D", tz="UTC")
    # Prices: 100, 100, 100, 100, 100, 120, 120, 120, 120, 120
    prices = [100.0, 100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0, 120.0]
    # Signals: Buy on bar 2, Sell on bar 6
    signals = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

    df = pd.DataFrame(
        {
            "close": prices,
            "signal": signals,
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


def test_backtester_init_validation():
    """Test parameter validation in EventDrivenBacktester initialization."""
    with pytest.raises(ValueError, match="initial_capital must be positive"):
        EventDrivenBacktester(initial_capital=0.0)

    with pytest.raises(ValueError, match="initial_capital must be positive"):
        EventDrivenBacktester(initial_capital=-5000.0)

    with pytest.raises(ValueError, match="fee_rate must be between 0 and 1"):
        EventDrivenBacktester(fee_rate=-0.01)

    with pytest.raises(ValueError, match="fee_rate must be between 0 and 1"):
        EventDrivenBacktester(fee_rate=1.0)

    with pytest.raises(ValueError, match="position_size must be between 0 and 1.0"):
        EventDrivenBacktester(position_size=0.0)

    with pytest.raises(ValueError, match="position_size must be between 0 and 1.0"):
        EventDrivenBacktester(position_size=1.5)


def test_backtester_input_validation(sample_signals_data):
    """Test input DataFrame validation in run_backtest."""
    backtester = EventDrivenBacktester(initial_capital=10000.0)

    # Non-DataFrame input
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        backtester.run_backtest(cast(Any, "invalid_input"))

    # Empty DataFrame
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        backtester.run_backtest(pd.DataFrame())

    # Missing required column 'signal'
    df_missing = sample_signals_data.drop(columns=["signal"])
    with pytest.raises(ValueError, match="missing required columns"):
        backtester.run_backtest(df_missing)


def test_backtester_trade_execution(sample_signals_data):
    """Test step-by-step cash, position, and portfolio accounting during backtest."""
    initial_cap = 10000.0
    fee_rate = 0.001  # 0.1%
    pos_size = 0.5   # 50% cash allocation

    backtester = EventDrivenBacktester(
        initial_capital=initial_cap,
        fee_rate=fee_rate,
        position_size=pos_size,
    )
    df_res = backtester.run_backtest(sample_signals_data)

    # Required output columns added
    expected_cols = {"cash", "position", "total_portfolio_value"}
    assert expected_cols.issubset(set(df_res.columns))

    # Bar 0 & 1: Signal = 0 -> Cash = 10000, Position = 0, Portfolio = 10000
    assert df_res["cash"].iloc[0] == initial_cap
    assert df_res["position"].iloc[0] == 0.0
    assert df_res["total_portfolio_value"].iloc[0] == initial_cap

    # Bar 2: Signal turns to 1 at price 100.0
    # Cash to spend = 10000 * 0.5 = 5000
    # Fee = 5000 * 0.001 = 5.0 -> Net cash = 4995.0
    # Position = 4995.0 / 100.0 = 49.95 BTC
    # Remaining Cash = 10000 - 5000 = 5000.0
    expected_pos = 4995.0 / 100.0
    assert pytest.approx(df_res["cash"].iloc[2]) == 5000.0
    assert pytest.approx(df_res["position"].iloc[2]) == expected_pos

    # Bar 5: Signal turns to 0 at price 120.0 -> Sell position
    # Gross proceeds = 49.95 * 120.0 = 5994.0
    # Fee = 5994.0 * 0.001 = 5.994 -> Net proceeds = 5988.006
    # Final Cash = 5000.0 + 5988.006 = 10988.006
    # Position = 0.0
    expected_final_cash = 5000.0 + (expected_pos * 120.0 * (1 - fee_rate))
    assert pytest.approx(df_res["cash"].iloc[5]) == expected_final_cash
    assert df_res["position"].iloc[5] == 0.0
    assert pytest.approx(df_res["total_portfolio_value"].iloc[5]) == expected_final_cash


def test_backtester_immutability(sample_signals_data):
    """Verify that input DataFrame is not mutated in-place during backtesting."""
    original_cols = list(sample_signals_data.columns)
    backtester = EventDrivenBacktester(initial_capital=10000.0)
    backtester.run_backtest(sample_signals_data)
    assert list(sample_signals_data.columns) == original_cols
