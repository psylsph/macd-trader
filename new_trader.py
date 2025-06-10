import backtrader as bt
import pandas as pd

class SmaCross(bt.Strategy):
    params = (('fast', 50), ('slow', 200), ('exit_neg', -0.05))  # Exit if loss > 5%

    def __init__(self):
        sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast)
        sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(sma1, sma2)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy(size=1)  # Enter long position
        elif self.crossover < 0:
            self.close()  # Exit position

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

# Set initial cash
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001) # 0.1% commission

# Add analyzers
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

df = pd.read_csv('data.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'})
# Create and add data feed
df = bt.feeds.PandasDirectData(
    dataname=df,
    datetime=0,  # Auto-detect from index
    open=1,      # Use column index for OHLCV
    high=2,
    low=3,
    close=4,
    volume=5,
    openinterest=-1  # No open interest data
    )

cerebro.adddata(df)  # Add your OHLC data

# Run backtest
initial_value = cerebro.broker.getvalue()
print("\n" + "="*50)
print("STARTING BACKTEST".center(50))
print("="*50)
print(f"\nInitial Portfolio Value: ${initial_value:,.2f}")

results = cerebro.run()
final_value = cerebro.broker.getvalue()

# Get analyzer results
strat = results[0]
ret_analyzer = strat.analyzers.returns.get_analysis()
sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
trade_analyzer = strat.analyzers.trades.get_analysis()

# Print summary
print("\n" + "="*50)
print("BACKTEST SUMMARY".center(50))
print("="*50)

# Portfolio Performance
print("\nPORTFOLIO PERFORMANCE")
print("-"*25)
print(f"Final Value: ${final_value:,.2f}")
print(f"Return: {((final_value - initial_value) / initial_value * 100):,.2f}%")
print(f"Sharpe Ratio: {sharpe_analyzer['sharperatio']:,.2f}" if sharpe_analyzer['sharperatio'] else "Sharpe Ratio: N/A")
print(f"Max Drawdown: {drawdown_analyzer.max.drawdown:,.2f}%")

# Trade Statistics
print("\nTRADE STATISTICS")
print("-"*25)
try:
    total_trades = trade_analyzer.total.closed
    winning_trades = trade_analyzer.won.total
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades}")
    print(f"Win Rate: {win_rate:,.1f}%")
    
    if trade_analyzer.won.pnl.average and trade_analyzer.lost.pnl.average:
        print(f"Avg Win: ${trade_analyzer.won.pnl.average:,.2f}")
        print(f"Avg Loss: ${trade_analyzer.lost.pnl.average:,.2f}")
except:
    print("No trade statistics available")

print("\n" + "="*50)

import matplotlib.pyplot as pl
pl.style.use("default") #ggplot is also fine
pl.rcParams["figure.figsize"] = (15,12)
# Plot results
cerebro.plot()