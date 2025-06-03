from ast import And
import pandas as pd
import backtrader as bt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

DEBUG = True

class MACDStrategy(bt.Strategy):

    params = (
        ('symbol', 'ETHUSD'),
        ('interval', 1),
        ('atr_window_size', 6),
        ('signal_width', 14),
        ('cooldown_bars', 24),  # Number of bars to wait after a sell before buying again
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9))

    def __init__(self):

        self.signals = None
        self.order = None
        self.buy_signal = None
        self.sell_signal = None
        self.buy_signals = []
        self.sell_signals = []
        self.last_sell_bar = 0  # Track the bar number of last sell
        self.bar_count = 0  # Track total bars seen

        # Initialize buy/sell signal prices to track executed prices
        self.buy_signal_price = None
        self.sell_signal_price = None

        # Store data reference
        self.dataclose = self.data0.close
        
        print("\nInitializing strategy indicators...")
        
        # Create MACD with histogram
        self.macd = bt.ind.MACDHisto()
        
        # Create moving averages
        self.ema_fast = bt.ind.EMA(self.data0, period=self.p.macd_fast)
        self.ema_slow = bt.ind.EMA(self.data0, period=self.p.macd_slow)
        self.macd_line = self.ema_fast - self.ema_slow
        self.signal_line = bt.ind.EMA(self.macd_line, period=self.p.macd_signal)
        self.ema200 = bt.ind.EMA(self.data0, period=200)
        self.rsi = bt.ind.RSI(self.data0, period=30)

        # Create ATR
        self.atr = bt.ind.ATR(self.data0, period=self.p.signal_width)

        
        # Set minimum period required
        minperiod = max(200, 26 + 9)
        self.addminperiod(minperiod)
        print(f"Minimum period set to {minperiod}")
        
        print("\nAll indicators initialized successfully")
        
        # Track buy/sell signals
        self.buy_timestamps = []
        self.position_openings = []

    def next(self):
        # Get current datetime
        current_dt = self.data.datetime.datetime(-1)
        current_close = self.dataclose[-1]
        self.bar_count += 1  # Increment bar counter

        # Check if we're still in cooldown period
        bars_since_last_sell = self.bar_count - self.last_sell_bar
        
        if (not self.position and
            self.ema_fast[-1] > self.ema_slow[-1] and
            #self.macd_line[-1] < 0 and
            self.dataclose[-1] > self.ema200[-1] and
            self.ema_slow[-2] < self.ema_slow[-1] and
            self.ema_fast[-2] < self.ema_fast[-1] and
            self.ema200[-2] < self.ema200[-1]
        ):

            if DEBUG:
                print("\n>>> BUY SIGNAL TRIGGERED <<<")
                print(f"Processing bar at {current_dt}: Close={current_close:.2f}")
                print(f"Entry Price: {current_close:.2f}")
                if bars_since_last_sell < self.p.cooldown_bars:
                    print(f"Buy Stopped in Cooldown")
                    return

            self.buy_signals.append((current_dt, current_close))
            self.buy_timestamps.append(current_dt)
            self.position_openings.append((current_dt, current_close, 'long'))
            #self.buy(size=0.5, sl=self.dataclose[-1] - 2*self.atr[-1])
            self.buy(size=0.5, sl=current_close)
            
        # Only proceed with position closing logic if we actually have an open position
        if hasattr(self, 'position') and self.position:
            exit_reason = None
            
            # Check exit conditions for long positions
            if self.position:
                entry_price = self.position.price
                tp_price = entry_price*1.05
                if self.dataclose[-1] > tp_price:
                    exit_reason = "Take profit target reached"
                elif self.ema_fast[-1] < self.ema_slow[-1]:  # Fast below Slow
                    exit_reason = "Fast below Slow"
                    self.last_sell_bar = self.bar_count  # Record the sell bar
                elif self.dataclose[-1] < self.ema200[-1]:  # Stop loss
                    exit_reason = "ema200 stop loss triggered"
                    self.last_sell_bar = self.bar_count  # Record the sell bar
                elif self.ema_fast[-2] > self.ema_fast[-1]:  # Fast above Fast
                    exit_reason = "Fast dropping"
                    self.last_sell_bar = self.bar_count
           
           
                # Close position if any exit condition met
                if exit_reason:
                    if DEBUG:
                        print(">>> SELL SIGNAL TRIGGERED <<<")
                        print(f"Processing bar at {current_dt}: Close={current_close:.2f}")
                        print(f"CLOSING POSITION: {exit_reason} on {self.data.datetime.datetime(-1).strftime('%m/%d/%Y')}")
                        print(f"Exit Price: {self.dataclose[-1]:.2f}")
                    self.sell_signals.append((current_dt, current_close))
                    if DEBUG:
                        print(f"Starting cooldown period of {self.p.cooldown_bars} bars")
                    self.close()
def stop(self):
        # Close any open position at the end of the backtest
        if self.position:
            if DEBUG:
                print("Closing last open position at end of backtest")
            self.close()

if __name__ == "__main__":
    
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy to Cerebro
    cerebro.addstrategy(MACDStrategy)

    # Load Data (example CSV file with timestamp, open, high, low, close, volume columns)
    data = pd.read_csv('data.csv', parse_dates=['timestamp'])
    data.set_index('timestamp', inplace=True)
    data = data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
    # Create and add data feed
    data_feed = bt.feeds.PandasDirectData(
        dataname=data,
        datetime=0,  # Auto-detect from index
        open=1,      # Use column index for OHLCV
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1  # No open interest data
    )
    cerebro.adddata(data_feed)

    # Configure the broker settings
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.002)
    
    # Add analyzers to evaluate strategy performance
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    #cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    #cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # Add observers to plot buy/sell signals, portfolio changes, and close price
    #cerebro.addobserver(bt.observers.BuySell)

    # Run the strategy
    # Run backtest with performance reporting
    initial_value = cerebro.broker.getvalue()
    print(f"\nStarting Backtest:")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    
    try:
        results = cerebro.run()
        
        final_value = cerebro.broker.getvalue()
        abs_return = final_value - initial_value
        pct_return = (abs_return / initial_value) * 100
        
        print("\nBacktest Results:")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Absolute Return: ${abs_return:.2f}")
        print(f"Percentage Return: {pct_return:.1f}%")
        
    except Exception as e:
        print(f"\nBacktest failed: {str(e)}")
        raise
    final_value = cerebro.broker.get_value()
    print(f"Final Portfolio Value: {final_value}")

    # Plot the results (including buy/sell signals and portfolio changes)
    cerebro.plot(style='candlestick')


