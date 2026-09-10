"""Quantitative Strategy Module for Digital Asset Backtesting.

Defines the abstract base class BaseStrategy and concrete implementations
such as BollingerBandsStrategy for signal generation without look-ahead bias.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract Base Class for quantitative trading strategies.

    Enforces a unified API interface across all algorithmic strategies in the engine.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates trading signals for a given market dataset.

        Args:
            df (pd.DataFrame): Input market data containing price history.

        Returns:
            pd.DataFrame: Copy of input DataFrame enriched with strategy indicators
                and execution signals.
        """
        pass


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Mean-Reversion Strategy.

    Statistical Rationale:
    ---------------------
    Bollinger Bands consist of an N-period Simple Moving Average (SMA) acting as
    the statistical mean, bounded by upper and lower bands situated K standard
    deviations (sigma) away. Under the assumption of local stationarity or mean-reverting
    price returns, prices spending extended periods near or below the lower band
    (mean - K * sigma) represent statistically oversold conditions, presenting a high-probability
    long entry opportunity.

    Signal & Execution Logic (Look-Ahead Bias Prevention):
    -----------------------------------------------------
    - Entry Trigger (Long = 1): Close price falls below the lower band (`close < lower_band`).
    - Exit Trigger (Flat = 0): Close price reverts to or exceeds the SMA (`close >= sma`).
    - Intermediate State: Position state is held via forward-filling until an exit condition is met.
    - Execution Signal (`signal`): `raw_signal.shift(1)` is applied. In institutional quantitative
      backtesting, signal generation based on candle T's close price can only be executed on
      candle T+1 (or next market open/close). Applying a 1-period lag strictly eliminates
      look-ahead / data-leakage bias.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        """Initializes the Bollinger Bands Strategy parameters.

        Args:
            window (int): Rolling window size for moving average and standard deviation. Defaults to 20.
            num_std (float): Standard deviation multiplier for band width. Defaults to 2.0.

        Raises:
            ValueError: If window or num_std are non-positive.
        """
        if window <= 0:
            raise ValueError(f"Window must be a positive integer, got {window}.")
        if num_std <= 0:
            raise ValueError(f"Number of standard deviations must be positive, got {num_std}.")

        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Bollinger Bands and generates execution signals.

        Args:
            df (pd.DataFrame): OHLCV market DataFrame containing a 'close' column.

        Returns:
            pd.DataFrame: New DataFrame containing original data plus 'sma', 'std',
                'upper_band', 'lower_band', 'raw_signal', and 'signal' columns.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If 'close' column is missing or DataFrame rows < window.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

        if 'close' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' price column.")

        if len(df) < self.window:
            raise ValueError(
                f"Input DataFrame has {len(df)} rows, but strategy requires at least "
                f"window={self.window} rows to compute rolling statistics."
            )

        # Create a clean copy to preserve input DataFrame immutability
        data = df.copy()

        # 1. Rolling statistics: 20-day SMA and standard deviation
        data['sma'] = data['close'].rolling(window=self.window).mean()
        data['std'] = data['close'].rolling(window=self.window).std()

        # 2. Upper and lower Bollinger Bands
        data['upper_band'] = data['sma'] + (self.num_std * data['std'])
        data['lower_band'] = data['sma'] - (self.num_std * data['std'])

        # 3. Raw position signal state machine:
        #    1 when close < lower_band (enter long)
        #    0 when close >= sma (exit position)
        raw = pd.Series(np.nan, index=data.index, dtype=float)
        raw.loc[data['close'] < data['lower_band']] = 1.0
        raw.loc[data['close'] >= data['sma']] = 0.0

        # Forward fill active position state, fill initial NaNs with 0 (flat)
        data['raw_signal'] = raw.ffill().fillna(0).astype(int)

        # 4. Execution signal shifted by 1 to strictly avoid look-ahead bias
        data['signal'] = data['raw_signal'].shift(1).fillna(0).astype(int)

        return data


if __name__ == "__main__":
    # Ensure project root is on sys.path for direct script execution
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import CryptoDataLoader

    print("\n--- Quantitative Strategy Module: Bollinger Bands ---")
    loader = CryptoDataLoader()
    df_market = loader.load_data(symbol="BTC/USDT", timeframe="1d", days_back=365)

    strategy = BollingerBandsStrategy(window=20, num_std=2.0)
    df_signals = strategy.generate_signals(df_market)

    # Identify signal entry transitions (0 -> 1)
    entry_mask = (df_signals['signal'] == 1) & (df_signals['signal'].shift(1) == 0)
    entries = df_signals.loc[entry_mask, ['close', 'sma', 'lower_band', 'upper_band', 'raw_signal', 'signal']]

    print("\n[Strategy Execution Summary]")
    print(f"Total Candles Processed: {len(df_signals)}")
    print(f"Total Long Entry Signals Triggered: {len(entries)}")

    print("\n[Long Position Entry Events (Signal Transition 0 -> 1)]")
    if not entries.empty:
        print(entries.to_string())
    else:
        print("No long entry signals triggered in the current date window.")
