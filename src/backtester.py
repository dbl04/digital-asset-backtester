"""
Event-Driven Backtester Engine for Quantitative Trading Strategies.

Simulates trade execution on historical market data and trading signals row-by-row,
accurately tracking cash balance, crypto asset position size, transaction fees,
and daily mark-to-market total portfolio value.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Configure module-level logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EventDrivenBacktester:
    """
    Event-driven simulation backtester that processes market data row-by-row.

    Attributes:
        initial_capital (float): Starting cash balance (default: 10,000.0 USD).
        fee_rate (float): Transaction fee percentage as a decimal (default: 0.001 = 0.1%).
        position_size (float): Fraction of current cash balance allocated per trade (default: 0.5 = 50%).
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        position_size: float = 0.5,
    ) -> None:
        """
        Initialize the EventDrivenBacktester with capital, fee, and sizing constraints.

        Args:
            initial_capital (float): Starting portfolio cash (must be > 0).
            fee_rate (float): Trade execution fee rate (must be 0 <= fee_rate < 1).
            position_size (float): Proportion of cash to invest per long entry (0 < position_size <= 1.0).

        Raises:
            ValueError: If input parameters fall outside valid numerical ranges.
        """
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if not (0 <= fee_rate < 1.0):
            raise ValueError(f"fee_rate must be between 0 and 1, got {fee_rate}")
        if not (0 < position_size <= 1.0):
            raise ValueError(f"position_size must be between 0 and 1.0, got {position_size}")

        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.position_size = position_size

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the event-driven backtest simulation row-by-row.

        Args:
            df (pd.DataFrame): Input market data containing at least 'close' and 'signal' columns.

        Returns:
            pd.DataFrame: A new DataFrame with added simulation columns:
                - 'cash': Ending cash balance at each bar.
                - 'position': Asset position size (BTC units) at each bar.
                - 'total_portfolio_value': Mark-to-market portfolio value (cash + position * close).

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If required columns ('close', 'signal') are missing or DataFrame is empty.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        required_cols = {'close', 'signal'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Input DataFrame is missing required columns: {missing}")

        # Preserve input DataFrame immutability
        data = df.copy()

        current_cash: float = self.initial_capital
        current_position: float = 0.0

        cash_history = []
        position_history = []
        portfolio_value_history = []

        logger.info(
            "Starting event-driven backtest | Initial Capital: $%.2f | Fee: %.2f%% | Position Size: %.0f%%",
            self.initial_capital,
            self.fee_rate * 100,
            self.position_size * 100,
        )

        # Row-by-row simulation loop (event-driven context)
        for close_val, signal_val in zip(data['close'], data['signal']):
            close_price: float = float(close_val)
            signal: int = int(signal_val)

            # 1. Buy Execution Logic
            # Trigger: Signal turns to 1 and we currently hold no position
            if signal == 1 and current_position == 0.0:
                cash_to_spend = current_cash * self.position_size
                fee = cash_to_spend * self.fee_rate
                net_cash = cash_to_spend - fee
                current_position = net_cash / close_price
                current_cash -= cash_to_spend

            # 2. Sell Execution Logic
            # Trigger: Signal turns to 0 and we currently hold an active position
            elif signal == 0 and current_position > 0.0:
                gross_proceeds = current_position * close_price
                fee = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - fee
                current_cash += net_proceeds
                current_position = 0.0

            # 3. Mark-to-Market Portfolio Valuation
            total_portfolio_value = current_cash + (current_position * close_price)

            cash_history.append(current_cash)
            position_history.append(current_position)
            portfolio_value_history.append(total_portfolio_value)

        # Record simulation results into DataFrame
        data['cash'] = cash_history
        data['position'] = position_history
        data['total_portfolio_value'] = portfolio_value_history

        logger.info(
            "Backtest complete | Final Portfolio Value: $%.2f | Total Return: %.2f%%",
            portfolio_value_history[-1],
            ((portfolio_value_history[-1] - self.initial_capital) / self.initial_capital) * 100,
        )

        return data


if __name__ == "__main__":
    # Ensure project root is on sys.path for direct script execution
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import CryptoDataLoader
    from src.strategy import BollingerBandsStrategy

    print("\n--- Event-Driven Backtester Module Execution ---")

    # 1. Load historical market data
    loader = CryptoDataLoader()
    df_market = loader.load_data(symbol="BTC/USDT", timeframe="1d", days_back=365)

    # 2. Generate strategy signals
    strategy = BollingerBandsStrategy(window=20, num_std=2.0)
    df_signals = strategy.generate_signals(df_market)

    # 3. Execute event-driven simulation
    backtester = EventDrivenBacktester(initial_capital=10000.0, fee_rate=0.001, position_size=0.5)
    df_results = backtester.run_backtest(df_signals)

    # 4. Calculate execution metrics
    initial_val = backtester.initial_capital
    final_val = df_results['total_portfolio_value'].iloc[-1]
    total_return_pct = ((final_val - initial_val) / initial_val) * 100

    # Count trades executed (0 -> 1 buy entries)
    buy_trades = ((df_results['signal'] == 1) & (df_results['signal'].shift(1) == 0)).sum()
    sell_trades = ((df_results['signal'] == 0) & (df_results['signal'].shift(1) == 1)).sum()

    print("\n" + "=" * 50)
    print("           BACKTEST PERFORMANCE SUMMARY           ")
    print("=" * 50)
    print(f"Initial Capital:         ${initial_val:,.2f}")
    print(f"Final Portfolio Value:   ${final_val:,.2f}")
    print(f"Total Net Return:        {total_return_pct:+.2f}%")
    print(f"Buy Orders Executed:     {buy_trades}")
    print(f"Sell Orders Executed:    {sell_trades}")
    print("=" * 50)

    print("\n[Sample Portfolio Value Records (Head)]")
    print(df_results[['close', 'signal', 'cash', 'position', 'total_portfolio_value']].head(10).to_string())

    print("\n[Sample Portfolio Value Records (Tail)]")
    print(df_results[['close', 'signal', 'cash', 'position', 'total_portfolio_value']].tail(10).to_string())
