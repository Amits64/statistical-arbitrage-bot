# Statistical Arbitrage Bot

## Overview
This is a **Statistical Arbitrage Trading Bot** that identifies and trades mean-reverting cryptocurrency pairs using **cointegration**, **hedge ratios**, **ARIMA forecasting**, and **Z-score trading signals**. It supports both **backtesting** and **real-time trading** on Binance using the `ccxt` library.

## Features
- **Cointegration Testing**: Selects pairs with a stable mean-reverting relationship.
- **Kalman Filter Hedge Ratio**: Determines the optimal hedge ratio dynamically.
- **Z-Score Signal Generation**: Identifies trading opportunities based on statistical deviation.
- **ARIMA Forecasting**: Predicts spread movements to enhance trading decisions.
- **Volatility-Based Position Sizing**: Adjusts trade size based on market conditions.
- **Risk Management**: Implements stop loss and take profit to control drawdowns.
- **Real-Time Trading**: Executes live trades using Binance API.
- **Performance Metrics**: Computes Sharpe Ratio, Max Drawdown, Win Rate, and Total Return.
- **Backtesting Module**: Simulates trading strategy on historical data.

## Installation
### Prerequisites
Ensure you have **Python 3.10+** and the following libraries installed:
```sh
pip install numpy ccxt statsmodels hurst matplotlib
```

## Configuration
Update `config` in the script with your API credentials and trading parameters:
```python
config = {
    'apiKey': 'YOUR_BINANCE_API_KEY',
    'secret': 'YOUR_BINANCE_SECRET_KEY',
    'enableRateLimit': True,
    'symbols': ['XRP/USDT', 'DOGE/USDT'],
    'timeframe': '1h',
    'zscore_entry': 1.6,
    'zscore_exit': 0.3,
    'risk_per_trade': 0.01,
    'portfolio_vol_target': 0.15,
    'stop_loss': 0.02,
    'take_profit': 0.04,
    'capital': 100,
    'real_time': False,
}
```

## Usage
### Backtesting
Run the script with `real_time` set to `False` to analyze past performance:
```sh
python pairs_trader.py
```

### Real-Time Trading
Enable live trading by setting `real_time = True` in `config`:
```python
'config['real_time'] = True
```
Then execute the bot:
```sh
python pairs_trader.py
```

## Trading Logic
1. **Fetch Historical Prices**: Retrieves OHLCV data for selected pairs.
2. **Cointegration Test**: Ensures the pair has a stable statistical relationship.
3. **Calculate Spread & Hedge Ratio**: Uses Kalman Filter for dynamic weighting.
4. **Generate Trading Signals**:
   - **Long Entry**: If the spread's Z-score < `-zscore_entry`.
   - **Short Entry**: If the spread's Z-score > `zscore_entry`.
   - **Exit Trade**: If Z-score < `zscore_exit`.
5. **Execute Trades**: Places market orders on Binance.
6. **Risk Management**: Implements stop loss and take profit rules.
7. **Performance Analysis**: Evaluates returns and risk metrics.

## Performance Metrics
After backtesting, the bot will display:
- **Sharpe Ratio**: Measures risk-adjusted returns.
- **Max Drawdown**: Largest peak-to-trough loss.
- **Win Rate**: Percentage of profitable trades.
- **Total Return**: Cumulative PnL of the strategy.

## Visualization
The bot plots the spread along with entry/exit thresholds:
![Spread Chart Example](example_spread_chart.png)

## Disclaimer
This bot is for educational purposes only. Cryptocurrency trading involves significant risk. Use at your own discretion.
