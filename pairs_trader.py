import numpy as np
import ccxt
from statsmodels.tsa.stattools import coint
from hurst import compute_Hc
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pairs_trader.log", mode="w")  # Logs will be saved to a file
    ]
)

# Configuration
config = {
    'apiKey': 'FpmgOpAE2bez7ct136mQVPdRt6lbanMnuDK54iqP0l928bQ13pAN5VPKuqH71XK4',
    'secret': 'KWmXhMKxAvkofRbrsOrLusKLB351t6kBBAKzHDlOFd53y2uNnX88vtj73czZls3j',
    'enableRateLimit': True,
    'symbols': ['BTC/USDT', 'ETH/USDT'],  # Add more pairs here
    'timeframe': '1h',
    'zscore_entry': 1.6,
    'zscore_exit': 0.3,
    'risk_per_trade': 0.01,  # 1% of capital per trade
    'portfolio_vol_target': 0.15,  # Target portfolio volatility
    'stop_loss': 0.02,  # Stop loss at 2% of spread
    'take_profit': 0.04,  # Take profit at 4% of spread
    'capital': 100,  # Total capital for position sizing
    'real_time': True,  # Set True for live trading
}

exchange = ccxt.binance({
    'apiKey': config['apiKey'],
    'secret': config['secret'],
    'enableRateLimit': config['enableRateLimit'],
})


# -------------------------
# Optimized Pairs Trader
# -------------------------
class OptimizedPairsTrader:
    def __init__(self, symbols, timeframe):
        self.symbols = symbols
        self.timeframe = timeframe
        self.spread_series = None
        self.historical_returns = None
        self.trade_log = []

    def fetch_data(self, live=False):
        """Fetch historical or live OHLCV data for given symbols."""
        price_data = {}
        for symbol in self.symbols:
            if live:
                ticker = exchange.fetch_ticker(symbol)
                price_data[symbol] = [ticker['last']]
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=500)
                price_data[symbol] = np.array([x[4] for x in ohlcv])  # Closing prices
        logging.info(f"Fetched data for symbols: {', '.join(self.symbols)}")
        return price_data

    def dynamic_cointegration_test(self, series1, series2):
        """Cointegration test."""
        score, p_value, _ = coint(series1, series2)
        return p_value < 0.05

    def kalman_hedge_ratio(self, series1, series2):
        """Approximate Kalman filter for hedge ratio."""
        x = np.vstack([series1, np.ones(len(series1))]).T
        hedge_ratio, _ = np.linalg.lstsq(x, series2, rcond=None)[0]
        return hedge_ratio

    def regime_filter(self, spread_series):
        """Avoid trading in trending markets."""
        H, _, _ = compute_Hc(spread_series, kind='price')
        return H < 0.45

    def position_sizing(self, volatility, capital, risk_per_trade):
        """Volatility-adjusted position sizing."""
        position_size = (capital * risk_per_trade) / volatility
        logging.info(f"Calculated position size: {position_size}")
        return position_size

    def arima_forecast(self, spread):
        """ARIMA-based forecast for the spread."""
        if len(spread) < 10:  # Ensure there are enough observations
            return np.zeros(10)
        try:
            model = ARIMA(spread, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)
            return forecast
        except Exception as e:
            logging.error(f"ARIMA forecast failed: {e}")
            return np.zeros(10)  # Return a default forecast (e.g., no movement)

    def generate_signals(self, price_series1, price_series2):
        """Generate entry and exit signals."""
        hedge_ratio = self.kalman_hedge_ratio(price_series1, price_series2)
        spread = price_series2 - hedge_ratio * price_series1
        forecast = self.arima_forecast(spread)
        mean, std = spread.mean(), spread.std()

        if std == 0:
            zscores = np.zeros_like(spread)
        else:
            zscores = (spread - mean) / std

        long_signal = zscores < -config['zscore_entry']
        short_signal = zscores > config['zscore_entry']
        exit_signal = abs(zscores) < config['zscore_exit']

        logging.info("Generated signals (long, short, exit).")
        return spread, zscores, long_signal, short_signal, exit_signal

    def live_trade(self, symbol1, symbol2, signal, volatility):
        """Execute live trades based on signals."""
        position_size = self.position_sizing(volatility, config['capital'], config['risk_per_trade'])
        try:
            if signal == 'long':
                order1 = exchange.create_market_buy_order(symbol1, position_size)
                order2 = exchange.create_market_sell_order(symbol2, position_size)
                logging.info(f"Executed Long on {symbol1} and Short on {symbol2}")
            elif signal == 'short':
                order1 = exchange.create_market_sell_order(symbol1, position_size)
                order2 = exchange.create_market_buy_order(symbol2, position_size)
                logging.info(f"Executed Short on {symbol1} and Long on {symbol2}")

            # Check if orders were successful
            if order1 and order2:
                logging.info("Orders executed successfully")
            else:
                logging.error("Order execution failed")
        except Exception as e:
            logging.error(f"Trade execution failed: {e}")

    def backtest_pair(self, price_series1, price_series2):
        """Backtest trading strategy on a single pair."""
        spread, zscores, long_signal, short_signal, exit_signal = self.generate_signals(price_series1, price_series2)
        self.spread_series = spread

        # Simulate trades
        positions = np.zeros(len(spread))
        positions[long_signal] = 1
        positions[short_signal] = -1
        positions[exit_signal] = 0
        positions = np.cumsum(positions)

        # Returns and Risk Management
        returns = positions[:-1] * np.diff(spread)
        stop_loss_level = config['stop_loss'] * spread.std()
        take_profit_level = config['take_profit'] * spread.std()
        returns = np.clip(returns, -stop_loss_level, take_profit_level)

        self.historical_returns = returns
        logging.info(f"Backtest completed. Final PnL: {returns.cumsum()[-1]}")
        return returns.cumsum()

    def real_time_monitor(self):
        """Real-time trading loop."""
        while True:
            try:
                price_data = self.fetch_data(live=True)
                if len(price_data) < 2:
                    logging.warning("Waiting for live data...")
                    time.sleep(60)
                    continue

                series1 = np.array(price_data[self.symbols[0]])
                series2 = np.array(price_data[self.symbols[1]])

                _, zscores, long_signal, short_signal, exit_signal = self.generate_signals(series1, series2)

                # Get volatility for position sizing
                volatility = np.std(series1 - series2)

                # Check signals
                if long_signal[-1]:
                    self.live_trade(self.symbols[0], self.symbols[1], 'long', volatility)
                elif short_signal[-1]:
                    self.live_trade(self.symbols[0], self.symbols[1], 'short', volatility)
                elif exit_signal[-1]:
                    logging.info("Exiting positions...")

                time.sleep(60)  # Monitor every minute
            except Exception as e:
                logging.error(f"Error in real-time monitoring: {e}")
                time.sleep(60)  # Wait before retrying

    def performance_metrics(self):
        """Compute performance metrics."""
        returns = self.historical_returns
        if len(returns) == 0:
            return {
                'Sharpe Ratio': 0,
                'Max Drawdown': 0,
                'Win Rate': 0,
                'Total Return': 0
            }
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        max_drawdown = np.min(returns.cumsum() - np.maximum.accumulate(returns.cumsum()))
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
        total_return = returns.cumsum()[-1] if len(returns) > 0 else 0

        logging.info(f"Performance Metrics: Sharpe Ratio: {sharpe_ratio}, Max Drawdown: {max_drawdown}, Win Rate: {win_rate}, Total Return: {total_return}")
        return {
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Total Return': total_return
        }


# -------------------------
# Execution and Testing
# -------------------------
if __name__ == "__main__":
    trader = OptimizedPairsTrader(config['symbols'], config['timeframe'])

    if config['real_time']:
        # Run real-time monitoring and trading
        trader.real_time_monitor()
    else:
        # Fetch historical price data
        price_data = trader.fetch_data()
        price_series1 = price_data[config['symbols'][0]]
        price_series2 = price_data[config['symbols'][1]]

        # Backtest
        pnl = trader.backtest_pair(price_series1, price_series2)
        metrics = trader.performance_metrics()

        logging.info(f"Backtest PnL: {pnl[-1]}")
        logging.info(f"Performance Metrics: {metrics}")

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(trader.spread_series, label="Spread")
        plt.axhline(trader.spread_series.mean(), color="green", linestyle="--", label="Mean")
        plt.axhline(trader.spread_series.mean() + config['zscore_entry'] * trader.spread_series.std(),
                    color="red", linestyle="--", label=f"+{config['zscore_entry']} Std Dev")
        plt.axhline(trader.spread_series.mean() - config['zscore_entry'] * trader.spread_series.std(),
                    color="red", linestyle="--", label=f"-{config['zscore_entry']} Std Dev")
        plt.title("Spread with Entry/Exit Boundaries")
        plt.legend()
        plt.show()

