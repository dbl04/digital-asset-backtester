# Modular Quantitative Backtesting Engine

## Overview
An event-driven backtesting framework built in Python for designing, simulating, and analyzing quantitative trading strategies in digital asset markets. This engine is designed with a strict focus on preventing data leakage and modeling realistic market execution environments.

## Core Features
*   **Event-Driven Architecture:** Replaces vectorized simulation with a sequential, day-by-day event loop to strictly prevent look-ahead bias and simulate real-time decision-making.
*   **Realistic Market Friction:** Incorporates customizable exchange taker fees (default `0.10%`) applied to every state transition (buy/sell) to prevent inflated, zero-fee theoretical returns.
*   **Dynamic Position Sizing:** Includes risk-management scaling. The default configuration uses a fixed-fractional sizing model (allocating 50% of available cash per signal) to mitigate drawdown severity and prevent total capital ruin.
*   **Automated Data Pipeline:** Integrates with the `ccxt` library to fetch historical OHLCV data directly from exchange REST APIs (Binance), featuring local CSV caching to optimize bandwidth and respect rate limits.

## Default Strategy Implementation
The framework comes pre-configured with a statistical mean-reversion strategy utilizing **Bollinger Bands**:
*   **Mathematical Logic:** Calculates a 20-period Simple Moving Average (SMA) and a 2-period standard deviation. 
*   **Signal Triggers:** Executes long positions when the asset price drops below the lower band (statistically oversold) and liquidates the position when the price reverts to the mean (SMA).

## Performance Visualizations
The framework includes a built-in `matplotlib` reporting suite that generates dual-pane charts mapping execution triggers (buy/sell) directly against the portfolio's mark-to-market equity curve.

![5-Year Backtest Regime Stress Test](reports/backtest_BTC_USDT_1d_w20_std2_20210912_to_20260910.png)
![1-Year Volatility Backtest](reports/backtest_BTC_USDT_1d_w20_std2_20250910_to_20260909.png)

## Quick Start Configuration

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/dbl04/digital-asset-backtester.git
cd digital-asset-backtester
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
