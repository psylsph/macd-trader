import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting import Backtest, Strategy
from ta import trend as ta_trend
from ta import volatility as ta_vol

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
    def init(self):
        close = self.data.Close
        # Calculate MACD components using exponential moving averages
        self.ema12 = self.I(EMA_Backtesting, close, 12)
        self.ema26 = self.I(EMA_Backtesting, close, 26)
        self.ema50 = self.I(EMA_Backtesting, close, 50)
        self.ema200 = self.I(EMA_Backtesting, close, 200)
        # Track buy/sell signals
        self.buy_signals = []
        self.sell_signals = []
        self.macd = self.ema12 - self.ema26
        self.signal = self.I(EMA_Backtesting, self.macd, 9)
        self.atr = self.I(ATR_Backtesting, self.data, 14)

    def next(self):
        # Only trade if we have enough data
        if len(self.data) < 50:
            return
            
      
        # Buy when MACD crosses above signal line, both are below zero, price is above EMA200, and all EMAs are not trending down
        if (not self.position and
            self.macd[-1] > self.signal[-1] and
            self.macd[-1] < 0 and
            self.signal[-1] < 0 and
            self.data.Close[-1] > self.ema200[-1] and
            self.ema12[-1] >= self.ema12[-2] and  # EMA12 not trending down
            self.ema26[-1] >= self.ema26[-2] and  # EMA26 not trending down
            self.ema50[-1] >= self.ema50[-2] and  # EMA50 not trending down
            self.ema200[-1] >= self.ema200[-2]):  # EMA200 not trending down
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
                # elif self.macd[-1] > 0:
                #     exit_reason = "MACD above zero (overbought)"
                elif self.data.Close[-1] < self.ema50[-1]:
                    exit_reason = "Price below EMA50"
                # Simplified EMA50-based stop loss and take profit
                entry_price = self.position.pl + self.data.Close[-1]
                sl_price = self.ema50[-1]
                tp_price = entry_price + 1.5*(entry_price - sl_price)  # 1.5:1 reward:risk
                
                if self.data.Close[-1] < sl_price:
                    exit_reason = "EMA50 stop loss triggered"
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
                print(f"CLOSING {'LONG' if self.position.is_long else 'SHORT'} POSITION: {exit_reason}")
                self.sell_signals.append((self.data.index[-1], self.data.Close[-1]))
                self.position.close()



def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculate MACD indicator"""
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    return data

def plot_macd(data, equity=None, buy_signals=None, sell_signals=None):
    """Plot close price, MACD indicator and equity curve"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot price and EMAs
    ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax1.plot(data.index, data['EMA12'], label='EMA 12', color='orange', alpha=0.5)
    ax1.plot(data.index, data['EMA26'], label='EMA 26', color='purple', alpha=0.5)
    ax1.plot(data.index, data['EMA50'], label='EMA 50', color='green', alpha=0.7)
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
        ax3.plot(equity.index, equity['Equity'], label='Equity', color='blue')
        ax3.set_ylabel('Equity ($)')
        ax3.legend()
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data and prepare for backtesting
    data = pd.read_csv('data.csv', parse_dates=['timestamp'])
    data = data.set_index('timestamp')
    data.index = pd.to_datetime(data.index)
    data = data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    data = calculate_macd(data)
    bt = Backtest(data, MACDStrategy, cash=10000, commission=.002,
                exclusive_orders=True)
    stats = bt.run()
    print(stats)
    #bt.plot()
    
    # Plot results with equity curve and signals
    equity = pd.DataFrame({
        'Equity': stats['_equity_curve']['Equity']
    }, index=data.index[:len(stats['_equity_curve'])])
    
    # Get strategy instance from backtest results
    strategy_instance = stats['_strategy']
    plot_macd(data,
             equity=equity,
             buy_signals=strategy_instance.buy_signals,
             sell_signals=strategy_instance.sell_signals)