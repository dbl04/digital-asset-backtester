"""Unit tests for BollingerBandsStrategy module."""

from typing import Any, cast
import numpy as np
import pandas as pd
import pytest

from src.strategy import BollingerBandsStrategy


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Fixture providing synthetic OHLCV market data for testing."""
    dates = pd.date_range(start="2026-01-01", periods=50, freq="D", tz="UTC")
    # Generate prices with a known drop to trigger lower band crossing
    prices = np.linspace(100.0, 110.0, 50)
    # Inject an oversold dip between bars 25 and 28
    prices[25:28] = [80.0, 78.0, 82.0]
    
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": 1000.0,
        },
        index=dates,
    )
    df.index.name = "datetime"
    return df


def test_bollinger_bands_strategy_init_validation():
    """Test parameter validation in BollingerBandsStrategy initialization."""
    with pytest.raises(ValueError, match="Window must be a positive integer"):
        BollingerBandsStrategy(window=0)

    with pytest.raises(ValueError, match="Window must be a positive integer"):
        BollingerBandsStrategy(window=-5)

    with pytest.raises(ValueError, match="Number of standard deviations must be positive"):
        BollingerBandsStrategy(window=20, num_std=0.0)

    with pytest.raises(ValueError, match="Number of standard deviations must be positive"):
        BollingerBandsStrategy(window=20, num_std=-1.5)


def test_bollinger_bands_strategy_input_validation(sample_market_data):
    """Test input DataFrame validation in generate_signals."""
    strategy = BollingerBandsStrategy(window=20, num_std=2.0)

    # Non-DataFrame input
    with pytest.raises(TypeError, match="Expected pandas DataFrame"):
        strategy.generate_signals(cast(Any, "invalid_input"))

    # Missing 'close' column
    df_missing = sample_market_data.drop(columns=["close"])
    with pytest.raises(ValueError, match="must contain a 'close' price column"):
        strategy.generate_signals(df_missing)

    # Insufficient rows
    df_short = sample_market_data.iloc[:10]
    with pytest.raises(ValueError, match="requires at least window=20 rows"):
        strategy.generate_signals(df_short)


def test_bollinger_bands_signal_generation(sample_market_data):
    """Test indicator calculation and signal shifting logic."""
    strategy = BollingerBandsStrategy(window=20, num_std=2.0)
    df_signals = strategy.generate_signals(sample_market_data)

    # Check required output columns added
    expected_cols = {"sma", "std", "upper_band", "lower_band", "raw_signal", "signal"}
    assert expected_cols.issubset(set(df_signals.columns))

    # Verify look-ahead bias prevention: signal at bar T equals raw_signal at bar T-1
    # Check that data['signal'] is raw_signal.shift(1).fillna(0)
    expected_signal = df_signals["raw_signal"].shift(1).fillna(0).astype(int)
    pd.testing.assert_series_equal(df_signals["signal"], expected_signal, check_names=False)

    # Verify entry trigger on raw_signal: when close < lower_band, raw_signal should be 1
    oversold_bars = df_signals["close"] < df_signals["lower_band"]
    if oversold_bars.any():
        assert (df_signals.loc[oversold_bars, "raw_signal"] == 1).all()


def test_bollinger_bands_immutability(sample_market_data):
    """Verify that input DataFrame is not mutated in-place."""
    original_cols = list(sample_market_data.columns)
    strategy = BollingerBandsStrategy(window=20, num_std=2.0)
    strategy.generate_signals(sample_market_data)
    assert list(sample_market_data.columns) == original_cols
