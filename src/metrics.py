"""
Performance Metrics Module for Quantitative Trading Strategies.

Calculates key risk-adjusted performance statistics including Cumulative Return,
Annualized Sharpe Ratio (24/7 crypto convention), and Maximum Drawdown from
mark-to-market portfolio equity history.
"""

import logging
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Calculates statistical performance and risk metrics for a backtested portfolio.

    Attributes:
        df (pd.DataFrame): Backtest results DataFrame containing 'total_portfolio_value'.
        initial_capital (float): Starting portfolio value for return calculations.
    """

    def __init__(self, df: pd.DataFrame, initial_capital: Optional[float] = None) -> None:
        """
        Initialize PerformanceMetrics with backtest output data.

        Args:
            df (pd.DataFrame): DataFrame produced by EventDrivenBacktester containing
                'total_portfolio_value'.
            initial_capital (Optional[float]): Initial capital balance. If None, defaults to the
                first row of 'total_portfolio_value'.

        Raises:
            TypeError: If df is not a pandas DataFrame.
            ValueError: If df is empty or missing 'total_portfolio_value' column.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}.")

        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        if 'total_portfolio_value' not in df.columns:
            raise ValueError("Input DataFrame is missing required column: 'total_portfolio_value'")

        self.df = df.copy()

        if initial_capital is not None:
            if initial_capital <= 0:
                raise ValueError(f"initial_capital must be positive, got {initial_capital}")
            self.initial_capital: float = initial_capital
        else:
            self.initial_capital: float = float(self.df['total_portfolio_value'].iloc[0])

    def calculate_cumulative_return(self) -> float:
        """
        Calculate total percentage change from initial capital to final portfolio value.

        Formula:
            Cumulative Return (%) = ((Final Value - Initial Capital) / Initial Capital) * 100

        Returns:
            float: Cumulative return as a percentage (e.g., 25.5 for +25.5%).
        """
        final_value: float = float(self.df['total_portfolio_value'].iloc[-1])
        cum_return: float = ((final_value - self.initial_capital) / self.initial_capital) * 100.0
        return cum_return

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
        """
        Calculate the Annualized Sharpe Ratio based on daily portfolio returns.

        Formula:
            Daily Return = (Portfolio_t - Portfolio_{t-1}) / Portfolio_{t-1}
            Sharpe Ratio = ((Mean Daily Return - Rf_daily) / Std Daily Return) * sqrt(365)

        Note:
            Crypto markets trade 24/7/365, so periods_per_year defaults to 365.

        Args:
            risk_free_rate (float): Annualized risk-free rate as a decimal (default: 0.0).
            periods_per_year (int): Trading periods per year (default: 365 for crypto).

        Returns:
            float: Annualized Sharpe Ratio.
        """
        daily_returns: pd.Series = self.df['total_portfolio_value'].pct_change().dropna()

        if daily_returns.empty or daily_returns.std(ddof=1) == 0.0:
            logger.warning("Daily returns have zero variance or insufficient data points. Returning 0.0 Sharpe.")
            return 0.0

        daily_rf: float = risk_free_rate / periods_per_year
        excess_returns: pd.Series = daily_returns - daily_rf
        mean_excess_return: float = excess_returns.mean()
        std_daily_return: float = daily_returns.std(ddof=1)

        annualized_sharpe: float = (mean_excess_return / std_daily_return) * np.sqrt(periods_per_year)
        return annualized_sharpe

    def calculate_max_drawdown(self) -> float:
        """
        Calculate the Maximum Drawdown (MDD) as a percentage peak-to-trough drop.

        Formula:
            Rolling Peak_t = max(Portfolio_0, ..., Portfolio_t)
            Drawdown_t = (Portfolio_t - Rolling Peak_t) / Rolling Peak_t
            Max Drawdown (%) = min(Drawdown_t) * 100

        Returns:
            float: Maximum drawdown as a negative percentage (e.g., -14.25 for a 14.25% drop).
        """
        portfolio_series: pd.Series = self.df['total_portfolio_value']
        rolling_peak: pd.Series = portfolio_series.cummax()
        drawdown_series: pd.Series = (portfolio_series - rolling_peak) / rolling_peak
        max_drawdown_pct: float = float(drawdown_series.min()) * 100.0
        return max_drawdown_pct

    def calculate_buy_and_hold_return(self) -> float:
        """
        Calculate Buy-and-Hold benchmark return percentage based on asset close prices.

        Returns:
            float: Percentage change in underlying asset price over the backtest period.
                   Returns 0.0 if 'close' column is missing or initial close is zero.
        """
        if 'close' not in self.df.columns:
            logger.warning("'close' column missing in DataFrame. Returning 0.0 for Buy & Hold benchmark.")
            return 0.0

        first_close = float(self.df['close'].iloc[0])
        last_close = float(self.df['close'].iloc[-1])

        if first_close == 0.0:
            return 0.0

        return ((last_close - first_close) / first_close) * 100.0

    def get_summary(self, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> Dict[str, float]:
        """
        Get all calculated performance metrics as a structured dictionary.

        Args:
            risk_free_rate (float): Annualized risk-free rate (default: 0.0).
            periods_per_year (int): Annualization factor (default: 365).

        Returns:
            Dict[str, float]: Dictionary containing cumulative_return_pct, annualized_sharpe, max_drawdown_pct,
                              and optionally buy_and_hold_return_pct.
        """
        summary = {
            "cumulative_return_pct": self.calculate_cumulative_return(),
            "annualized_sharpe": self.calculate_sharpe_ratio(risk_free_rate, periods_per_year),
            "max_drawdown_pct": self.calculate_max_drawdown(),
        }
        if 'close' in self.df.columns:
            summary["buy_and_hold_return_pct"] = self.calculate_buy_and_hold_return()
        return summary

    def print_summary(self, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> None:
        """
        Print a formatted performance metrics report to the console.
        """
        summary = self.get_summary(risk_free_rate, periods_per_year)
        print("\n" + "=" * 55)
        print("          STRATEGY PERFORMANCE METRICS REPORT          ")
        print("=" * 55)
        print(f"Initial Capital:          ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value:    ${float(self.df['total_portfolio_value'].iloc[-1]):,.2f}")
        print(f"Cumulative Return:        {summary['cumulative_return_pct']:+.2f}%")
        if "buy_and_hold_return_pct" in summary:
            print(f"Buy & Hold Return:        {summary['buy_and_hold_return_pct']:+.2f}%")
        print(f"Annualized Sharpe Ratio:  {summary['annualized_sharpe']:.2f}")
        print(f"Maximum Drawdown:         {summary['max_drawdown_pct']:.2f}%")
        print("=" * 55)


if __name__ == "__main__":
    # Ensure project root is on sys.path for direct script execution
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import CryptoDataLoader
    from src.strategy import BollingerBandsStrategy
    from src.backtester import EventDrivenBacktester

    print("\n--- Performance Metrics Module Self-Test ---")

    # 1. Load data
    loader = CryptoDataLoader()
    df_market = loader.load_data(symbol="BTC/USDT", timeframe="1d", days_back=365)

    # 2. Strategy signals
    strategy = BollingerBandsStrategy(window=20, num_std=2.0)
    df_signals = strategy.generate_signals(df_market)

    # 3. Backtest
    backtester = EventDrivenBacktester(initial_capital=10000.0, fee_rate=0.001, position_size=0.5)
    df_results = backtester.run_backtest(df_signals)

    # 4. Metrics evaluation
    metrics = PerformanceMetrics(df_results, initial_capital=10000.0)
    metrics.print_summary()
