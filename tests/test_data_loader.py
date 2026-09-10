"""Unit tests for CryptoDataLoader module."""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import CryptoDataLoader


@pytest.fixture
def mock_csv_data(tmp_path) -> Path:
    """Fixture to create a temporary CSV file simulating cached market data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "BTC_USDT_1d.csv"

    now = pd.Timestamp.now(tz="UTC").floor("D")
    dates = pd.date_range(end=now, periods=10, freq="D")
    df = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [500.0] * 10,
        },
        index=dates,
    )
    df.index.name = "datetime"
    df.to_csv(file_path)
    return data_dir



def test_data_loader_initialization(tmp_path):
    """Test CryptoDataLoader directory creation and properties."""
    data_dir = tmp_path / "test_data"
    loader = CryptoDataLoader(data_dir=str(data_dir))

    assert loader.data_dir == data_dir
    assert data_dir.exists()
    assert data_dir.is_dir()


def test_load_data_from_local_csv_cache(mock_csv_data):
    """Test reading market data from existing CSV disk cache."""
    loader = CryptoDataLoader(data_dir=str(mock_csv_data))
    df = loader.load_data(symbol="BTC/USDT", timeframe="1d", days_back=5)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"open", "high", "low", "close", "volume"}
    assert expected_cols.issubset(set(df.columns))
    assert df.index.name == "datetime"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
