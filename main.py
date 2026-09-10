"""
Main Orchestration Script for Digital Asset Backtester Framework.

Executes the quantitative backtesting workflow end-to-end:
1. Data Loading: Fetches/loads historical BTC/USDT daily OHLCV market data.
2. Strategy Signal Generation: Computes Bollinger Bands and lag-adjusted execution signals.
3. Event-Driven Execution: Simulates trade entries, exits, position sizing, fees, and mark-to-market portfolio value.
4. Risk & Performance Evaluation: Computes Cumulative Return, Annualized Sharpe Ratio, and Max Drawdown.
5. Data Visualization: Renders a 2-subplot matplotlib dashboard showing market indicators, execution triggers, and equity curve.
"""

import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CryptoDataLoader
from src.strategy import BollingerBandsStrategy
from src.backtester import EventDrivenBacktester
from src.metrics import PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def plot_backtest_results(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    symbol: str = "BTC/USDT",
    timeframe: str = "1d",
    window: int = 20,
    num_std: float = 2.0,
    filename: str | None = None
) -> None:
    """
    Renders a professional 2-subplot matplotlib chart visualizing strategy execution.

    Top Subplot:
        - BTC/USDT Close price
        - 20-period SMA (Middle Band)
        - Upper Band & Lower Band with shaded envelope region
        - Green upward triangles for Buy Entries
        - Red downward triangles for Sell Exits

    Bottom Subplot:
        - Mark-to-market Total Portfolio Value (Equity Curve) over time
        - Initial capital reference baseline

    Args:
        df (pd.DataFrame): Backtested DataFrame containing price, band indicators,
            signals, and portfolio values.
        initial_capital (float): Starting portfolio cash baseline (default: 10000.0).
        symbol (str): Asset symbol string (default: "BTC/USDT").
        timeframe (str): Candle timeframe (default: "1d").
        window (int): Bollinger Bands lookback window (default: 20).
        num_std (float): Bollinger Bands standard deviation multiplier (default: 2.0).
        filename (str | None): Custom output image filename. Auto-generated if None.
    """
    # Configure clean, modern plotting style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f"Digital Asset Backtesting Engine: {symbol} Bollinger Bands ({timeframe})", fontsize=15, fontweight='bold', y=0.98)

    # Identify trade signal transitions
    # Buy Entry: Signal changes 0 -> 1
    buy_mask = (df['signal'] == 1) & (df['signal'].shift(1) == 0)
    # Sell Exit: Signal changes 1 -> 0
    sell_mask = (df['signal'] == 0) & (df['signal'].shift(1) == 1)

    buy_dates = df.index[buy_mask]
    buy_prices = df.loc[buy_mask, 'close']

    sell_dates = df.index[sell_mask]
    sell_prices = df.loc[sell_mask, 'close']

    # --- TOP SUBPLOT: Price & Bollinger Bands ---
    ax1.plot(df.index, df['close'], label=f'{symbol} Close Price', color='#1f77b4', linewidth=1.5, alpha=0.9)
    ax1.plot(df.index, df['sma'], label=f'{window} SMA (Middle Band)', color='#ff7f0e', linewidth=1.2, linestyle='--')
    ax1.plot(df.index, df['upper_band'], label=f'Upper Band (+{num_std:g}σ)', color='#2ca02c', linewidth=1.0, linestyle=':')
    ax1.plot(df.index, df['lower_band'], label=f'Lower Band (-{num_std:g}σ)', color='#d62728', linewidth=1.0, linestyle=':')

    # Shaded band region
    ax1.fill_between(df.index, df['lower_band'], df['upper_band'], color='gray', alpha=0.10, label='Bollinger Envelope')

    # Scatter markers for Buy / Sell signals
    ax1.scatter(
        buy_dates,
        buy_prices,
        marker='^',
        color='#00c853',
        s=120,
        zorder=5,
        label=f'Buy Signal ({len(buy_dates)})',
        edgecolors='black',
        linewidth=0.5
    )
    ax1.scatter(
        sell_dates,
        sell_prices,
        marker='v',
        color='#d50000',
        s=120,
        zorder=5,
        label=f'Sell Signal ({len(sell_dates)})',
        edgecolors='black',
        linewidth=0.5
    )

    ax1.set_ylabel("Price (USDT)", fontsize=11, fontweight='bold')
    ax1.set_title("Market Price & Technical Execution Triggers", fontsize=12, fontweight='semibold')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- BOTTOM SUBPLOT: Total Portfolio Value (Equity Curve) ---
    ax2.plot(df.index, df['total_portfolio_value'], label='Total Portfolio Value', color='#9467bd', linewidth=2.0)
    ax2.axhline(
        initial_capital,
        color='black',
        linestyle='--',
        linewidth=1.0,
        alpha=0.7,
        label=f'Initial Capital (${initial_capital:,.0f})'
    )

    ax2.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Portfolio Value (USD)", fontsize=11, fontweight='bold')
    ax2.set_title("Mark-to-Market Portfolio Equity Curve", fontsize=12, fontweight='semibold')
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Save figure plot to reports directory if available or root
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        clean_symbol = symbol.replace('/', '_')
        if hasattr(df.index, 'strftime') and len(df.index) > 0:
            start_str = df.index[0].strftime('%Y%m%d')
            end_str = df.index[-1].strftime('%Y%m%d')
            date_range_str = f"_{start_str}_to_{end_str}"
        else:
            date_range_str = ""
        filename = f"backtest_{clean_symbol}_{timeframe}_w{window}_std{num_std:g}{date_range_str}.png"

    chart_path = reports_dir / filename
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    plt.savefig(chart_path, dpi=300)
    logger.info("Saved backtest visualization chart to: %s", chart_path)

    # Display plot window
    plt.show()


def main() -> None:
    """
    Main orchestration routine. Runs all framework components in sequence:
    DataLoader -> Strategy -> Backtester -> Metrics -> Plotter.
    """
    logger.info("=== STARTING QUANTITATIVE BACKTESTING FRAMEWORK ===")

    # 1. Load Market Data
    symbol = "BTC/USDT"
    timeframe = "1d"
    days_back = 365 * 5

    logger.info("Step 1: Loading historical market data for %s (%s, %d days)...", symbol, timeframe, days_back)
    loader = CryptoDataLoader()
    df_market = loader.load_data(symbol=symbol, timeframe=timeframe, days_back=days_back)

    # 2. Generate Trading Signals
    window = 20
    num_std = 2.0
    logger.info("Step 2: Generating strategy signals (Bollinger Bands window=%d, num_std=%.1f)...", window, num_std)
    strategy = BollingerBandsStrategy(window=window, num_std=num_std)
    df_signals = strategy.generate_signals(df_market)

    # 3. Execute Event-Driven Backtest
    initial_capital = 10000.0
    fee_rate = 0.001       # 0.1% transaction fee
    position_size = 0.5    # 50% allocation per entry

    logger.info("Step 3: Executing event-driven simulation (Capital: $%.2f, Fee: %.2f%%)...", initial_capital, fee_rate * 100)
    backtester = EventDrivenBacktester(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        position_size=position_size
    )
    df_results = backtester.run_backtest(df_signals)

    # 4. Evaluate Performance & Risk Metrics
    logger.info("Step 4: Calculating quantitative performance metrics...")
    metrics = PerformanceMetrics(df_results, initial_capital=initial_capital)
    metrics.print_summary()

    # 5. Render Visualization Dashboard
    logger.info("Step 5: Rendering graphical visualization dashboard...")
    plot_backtest_results(
        df_results,
        initial_capital=initial_capital,
        symbol=symbol,
        timeframe=timeframe,
        window=window,
        num_std=num_std
    )

    logger.info("=== BACKTEST WORKFLOW COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
