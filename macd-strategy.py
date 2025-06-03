import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import backtrader as bt
from ta import trend as ta_trend
from ta import volatility as ta_vol
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="ta.trend")

DEBUG = True

class MACDStrategy(bt.Strategy):

    params = (
        ('symbol', 'ETHUSD'),
        ('interval', 1),
        ('ema_stop_window_size', 30),
        ('atr_window_size', 6),
        ('risk_reward', 1.1),
        ('signal_width', 14),
        ('max_buy_per_24h', 1),
        ('macd_offset', 3),
        ('signal_offset', 0),
        # Removed duplicate 'signal_width'
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

        # Create ATR
        self.atr = bt.ind.ATR(self.data0, period=self.p.signal_width)

        
        # Set minimum period required
        minperiod = max(200, 26 + 9)
        self.addminperiod(minperiod)
        print(f"✓ Minimum period set to {minperiod}")
        
        print("\nAll indicators initialized successfully")
        
        # Track buy/sell signals
        self.buy_timestamps = []
        self.position_openings = []

    def next(self):
        # Get current datetime
        current_dt = self.data.datetime.datetime(-1)
        current_close = self.dataclose[-1]
        #print(f"\nProcessing bar at {current_dt}: Close={current_close:.2f}")
        #print(f"\nProcessing bar at {current_dt}: self.ema_fast[-1]={self.ema_fast[-1]:.2f}, self.ema_slow[-1]={self.ema_slow[-1]:.2f}")
        #print(f"\nProcessing bar at {current_dt}: self.macd_line[-1]={self.macd_line[-1]:.2f}")
        
        if (not self.position and
            self.ema_fast[-1] > self.ema_slow[-1] and
            #self.macd_line[-1] < 0 and
            self.dataclose[-1] > self.ema200[-1]
        ):

            if DEBUG:
                print(f"\nProcessing bar at {current_dt}: Close={current_close:.2f}")
                print(">>> BUY SIGNAL TRIGGERED <<<")
                print(f"Entry Price: {current_close:.2f}")
            self.buy_signals.append((current_dt, current_close))
            self.buy_timestamps.append(current_dt)
            self.position_openings.append((current_dt, current_close, 'long'))
            self.buy(size=0.5, sl=self.dataclose[-1] - 2*self.atr[-1])
            
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
                elif self.dataclose[-1] < self.ema200[-1]:  # Stop loss
                    exit_reason = "ema200 stop loss triggered"
           
           
                # Close position if any exit condition met
                if exit_reason:
                    if DEBUG:
                        print(f"CLOSING POSITION: {exit_reason} on {self.data.datetime.datetime(-1).strftime('%m/%d/%Y')}")
                        print(f"Exit Price: {self.dataclose[-1]:.2f}")
                    self.sell_signals.append((current_dt, current_close))
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
    cerebro.broker.setcommission(commission=0.0025)
    
    # Add analyzers to evaluate strategy performance
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    #cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # Add observers to plot buy/sell signals and portfolio changes
    cerebro.addobserver(bt.observers.BuySell)
    #cerebro.addobserver(bt.observers.Broker)

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


