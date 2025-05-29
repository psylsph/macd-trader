import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting import Backtest, Strategy
from ta import trend as ta_trend
from ta import volatility as ta_vol
import warnings

DEBUG = False

def EMA_Backtesting(values, n):
    """
    Return exponential moving average of `values`, at
    each step taking into account `n` previous values.
    """
    close = pd.Series(values)
    return ta_trend.EMAIndicator(close, window=n).ema_indicator()

def ATR_Backtesting(data, window):
    """
    Calculate the Average True Range (ATR) for the given data.
    """
    high = pd.Series(data.High)
    low = pd.Series(data.Low)
    close = pd.Series(data.Close)
    return ta_vol.AverageTrueRange(high, low, close, window=window).average_true_range()

class MACDStrategy(Strategy):

    ema_window_size = 30
    atr_window_size = 5
    risk_reward = 1.1
    signal_width = 12

    def init(self):
        close = self.data.Close
        # Calculate MACD components using exponential moving averages
        self.ema12 = self.I(EMA_Backtesting, close, 12)
        self.ema26 = self.I(EMA_Backtesting, close, 26)
        self.emaStop = self.I(EMA_Backtesting, close, self.ema_window_size)
        self.ema200 = self.I(EMA_Backtesting, close, 200)
        # Track buy/sell signals
        self.buy_signals = []
        self.sell_signals = []
        self.macd = self.ema12 - self.ema26
        self.signal = self.I(EMA_Backtesting, self.macd, self.signal_width)
        self.atr = self.I(ATR_Backtesting, self.data, self.atr_window_size)

    def next(self):
           
        # Buy when MACD crosses above signal line, both are below zero, price is above EMA200, and all EMAs are not trending down
        if (not self.position and
            self.macd[-1] > self.signal[-1] and
            self.macd[-1] < 0 and
            self.signal[-1] < 0 and
            self.ema12[-1] >= self.ema12[-2] and  # EMA12 not trending down
            self.ema26[-1] >= self.ema26[-2] and  # EMA26 not trending down
            self.emaStop[-1] >= self.emaStop[-2] and # emaStop not trending down
            self.signal[-1] >= self.signal[-2]):  # Signal not trending down
            if DEBUG:
                print("BUY SIGNAL")
            self.buy_signals.append((self.data.index[-1], self.data.Close[-1]))
            self.buy(size=0.5, sl=self.data.Close[-1] - 2*self.atr[-1])
            
        # Consolidated position closing logic
        if self.position:
            exit_reason = None
            sl_price = None
            
            # Check exit conditions for long positions
            if self.position.is_long:
                if self.macd[-1] < self.signal[-1]:
                    exit_reason = "MACD crossover below signal"
                elif self.data.Close[-1] < self.emaStop[-1]:
                    exit_reason = "Price below emaStop"
                # Simplified emaStop-based stop loss and take profit
                entry_price = self.position.pl + self.data.Close[-1]
                sl_price = self.emaStop[-1]
                tp_price = entry_price + self.risk_reward*(entry_price - sl_price)
                
                if self.data.Close[-1] < sl_price:
                    exit_reason = "emaStop stop loss triggered"
                elif self.data.Close[-1] > tp_price:
                    exit_reason = "Take profit target reached"
            
            # Check exit conditions for short positions
            elif self.position.is_short:
                if self.macd[-1] > self.signal[-1]:
                    exit_reason = "MACD crossover above signal"
                elif self.macd[-1] < 0:
                    exit_reason = "MACD below zero (oversold)"
                elif self.data.Close[-1] > self.ema200[-1]:
                    exit_reason = "Price above EMA200"
                # Enhanced stop loss logic
                entry_price = self.position.pl + self.data.Close[-1]
                if self.data.Close[-1] > self.ema200[-1]:
                    # Above EMA200 - use tighter stop loss
                    sl_price = max(entry_price + 1*self.atr[-1], self.ema200[-1])
                    if self.data.Close[-1] > sl_price:
                        exit_reason = "EMA200 zone stop loss triggered"
                else:
                    # Below EMA200 - normal stop loss
                    sl_price = entry_price + 2*self.atr[-1]
                    if self.data.Close[-1] > sl_price:
                        exit_reason = "ATR stop loss triggered"
            
            # Close position if any exit condition met
            if exit_reason:
                if DEBUG:
                    print(f"CLOSING {'LONG' if self.position.is_long else 'SHORT'} POSITION: {exit_reason}")
                self.sell_signals.append((self.data.index[-1], self.data.Close[-1]))
                self.position.close()


def plot_macd(data, equity=None, buy_signals=None, sell_signals=None):
    """Plot close price, MACD indicator and equity curve"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Calculate EMA12 and EMA26 for plotting
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']

    
    # Plot price and EMAs
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data['EMA12'], label='EMA 12', color='orange', alpha=0.5)
    ax1.plot(data.index, data['EMA26'], label='EMA 26', color='purple', alpha=0.5)
    ax1.plot(data.index, data['EMA200'], label='EMA 200', color='red', alpha=0.7)
    
    # Plot buy/sell signals if provided
    if buy_signals:
        dates, prices = zip(*buy_signals)
        ax1.scatter(dates, prices, color='green', marker='^',
                   label='Buy Signals', alpha=0.7, s=100)
    if sell_signals:
        dates, prices = zip(*sell_signals)
        ax1.scatter(dates, prices, color='red', marker='v',
                   label='Sell Signals', alpha=0.7, s=100)
    
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MACD components
    ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
    ax2.plot(data.index, data['Signal'], label='Signal Line', color='orange')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Zero line
    
    # Plot histogram with color based on value
    pos_mask = data['Histogram'] > 0
    neg_mask = data['Histogram'] < 0
    ax2.bar(data.index[pos_mask], data['Histogram'][pos_mask],
            color='green', width=0.8, label='Bullish')
    ax2.bar(data.index[neg_mask], data['Histogram'][neg_mask],
            color='red', width=0.8, label='Bearish')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    # Plot equity curve if provided
    if equity is not None:
        ax3.plot(equity.index, equity, label='Equity', color='blue')
        ax3.set_ylabel('Equity ($)')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Load data and prepare for backtesting
    data = pd.read_csv('data.csv', parse_dates=['timestamp'])
    data = data.set_index('timestamp')
    data = data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    bt = Backtest(data, MACDStrategy, cash=10000, commission=.002, margin=0.9,
                exclusive_orders=False, finalize_trades=True)
    stats = bt.run()
    #stats = bt.optimize(ema_window_size=range(30, 75), atr_window_size=range(5, 15),
    #                    risk_reward=[1.1, 1.2, 1.3, 1.4, 1.5], signal_width=range(5,16), maximize='Equity Final [$]')
    print("\n--- Backtest Statistics (Best Run) ---")
    print(stats) # Prints the main statistics Series for the best run

    # Access and print the best parameters from the strategy object
    best_strategy = stats._strategy
    print("\n--- Optimal Parameters Found ---")
    print(f"ema_window_size: {best_strategy.ema_window_size}")
    print(f"atr_window_size: {best_strategy.atr_window_size}")
    print(f"risk_reward: {best_strategy.risk_reward}")
    print(f"signal_width: {best_strategy.signal_width}")
    # Add any other parameters you might optimize in the future here
    warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
    warnings.filterwarnings("ignore", message="Superimposed OHLC plot matches the original plot. Skipping.")
    bt.plot()
