import os
import asyncio
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.data.live import CryptoDataStream
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import logging
import keyboard  # For keyboard input monitoring
from ta import trend as ta_trend
from ta import volatility as ta_vol

from dotenv import load_dotenv
# Load environment variables from .env file at the script level
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Alpaca credentials from environment
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

class LiveMACDStrategy:
    def __init__(self, symbol='ETH/USD', timeframe='1Min', paper=True):
        """Initialize trading strategy"""
        self.symbol = symbol
        self.timeframe = timeframe
        self.paper = paper
        self.stream_task = None
        self.data_stream = None
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)
        self.bars = pd.DataFrame()
        self.positions = []
        self.ema_stop_window_size = 30
        self.atr_window_size = 6
        self.risk_reward = 1.1
        self.signal_width = 14
        self.max_buy_per_24h = 1
        self.macd_offset = 3
        self.signal_offset = 3
        self.buy_timestamps = []
        self.close_timestamps = []

    def update_indicators(self):
        """Calculate and update technical indicators"""
        if len(self.bars) < 200:  # Need enough data for indicators
            return
            
        close = self.bars['close']
        self.ema12 = ta_trend.EMAIndicator(close, window=12).ema_indicator()
        self.ema26 = ta_trend.EMAIndicator(close, window=26).ema_indicator()
        self.emaStop = ta_trend.EMAIndicator(close, window=self.ema_stop_window_size).ema_indicator()
        self.ema200 = ta_trend.EMAIndicator(close, window=200).ema_indicator()
        self.macd = self.ema12 - self.ema26
        self.signal = ta_trend.EMAIndicator(pd.Series(self.macd), window=self.signal_width).ema_indicator()
        self.atr = ta_vol.AverageTrueRange(
            high=self.bars['high'],
            low=self.bars['low'],
            close=close,
            window=self.atr_window_size
        ).average_true_range()

    async def handle_bar(self, bar):
        """Process incoming bar data"""
        try:
            # Update bars DataFrame
            new_bar = pd.DataFrame({
                'open': [bar.open],
                'high': [bar.high],
                'low': [bar.low],
                'close': [bar.close],
                'volume': [bar.volume]
            }, index=[pd.to_datetime(bar.timestamp)])
            
            self.bars = pd.concat([self.bars, new_bar]).iloc[-1000:]  # Keep last 1000 bars
            
            # Update indicators and check signals
            self.update_indicators()
            self.check_signals()
            
        except Exception as e:
            logger.error(f"Error processing bar: {str(e)}")

    def check_signals(self):
        """Check for trading signals"""
        if len(self.bars) < 200:  # Need enough data for signals
            return
            
        # Always get fresh position from API
        position = self.get_open_position()
        if position:
            logger.info(f"Current position: {position.side} {position.qty} {position.symbol} @ {position.avg_entry_price}")
        else:
            logger.info("No current position")
        
        # Entry signals
        if not position:
            self.check_entry_signals()
        else:
            self.check_exit_signals(position)

    def check_entry_signals(self):
        """Check for entry opportunities"""
        try:
            # Remove buy timestamps older than 24 hours
            current_time = pd.Timestamp.now()
            self.buy_timestamps = [t for t in self.buy_timestamps
                                 if (current_time - t).total_seconds() < 24*60*60]
                
            # Buy when MACD crosses above signal line, both are below offsets, price is above EMA200,
            # all EMAs are not trending down, and we haven't exceeded max buys in 24h
            if (self.macd.iloc[-1] > self.signal.iloc[-1] and
                self.macd.iloc[-1] < self.macd_offset and
                self.signal.iloc[-1] < self.signal_offset and
                self.bars['close'].iloc[-1] >= self.ema200.iloc[-1]*1.003 and
                self.ema12.iloc[-1] >= self.ema12.iloc[-2] and  # EMA12 not trending down
                self.ema26.iloc[-1] >= self.ema26.iloc[-2] and  # EMA26 not trending down
                self.emaStop.iloc[-1] >= self.emaStop.iloc[-2] and # emaStop not trending down
                self.ema200.iloc[-1] >= self.ema200.iloc[-2] and # ema200 not trending down
                self.signal.iloc[-1] >= self.signal.iloc[-2] and # Signal not trending down
                len(self.buy_timestamps) < self.max_buy_per_24h):
                
                self.enter_position(OrderSide.BUY)
                self.last_trade_time = pd.Timestamp.now()
                
        except Exception as e:
            logger.error(f"Entry signal error: {str(e)}")

    def check_exit_signals(self, position):
        """Check if current position should be exited"""
        try:
            exit_reason = None
            
            # Exit long positions
            if position and position.side == OrderSide.BUY:
                entry_price = float(position.entry_price)
                current_price = self.bars['close'].iloc[-1]
                tp_price = entry_price*1.0055
                
                if current_price > tp_price:
                    exit_reason = "Take profit target reached"
                elif self.macd.iloc[-1] < self.signal.iloc[-1]:
                    exit_reason = "MACD crossover below signal"
                elif current_price < self.emaStop.iloc[-1]:
                    exit_reason = "emaStop stop loss triggered"
                    
            if exit_reason:
                self.exit_position(exit_reason)
                
        except Exception as e:
            logger.error(f"Exit signal error: {str(e)}")

    def get_available_funds(self) -> float:
        """Get available trading funds in USD"""
        try:
            # Get account info without explicit type import
            account = self.trading_client.get_account()
            if not account or not hasattr(account, 'cash'):
                logger.error("No valid account information available")
                return 0.0
                
            cash_str = getattr(account, 'cash', '0')
            if not cash_str:
                logger.error("No cash balance available")
                return 0.0
                
            try:
                return float(cash_str)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid cash value '{cash_str}': {str(e)}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0.0

    def enter_position(self, side: OrderSide):
        """Enter new position with risk management"""
        try:
            current_price = self.bars['close'].iloc[-1]
            qty = 0.1  # Fixed position size for example
            required_funds = current_price * qty
            
            # Check available funds with buffer
            available_funds = self.get_available_funds()
            buffer = 1.05  # 5% buffer for price fluctuations
            if available_funds < required_funds * buffer:
                logger.warning(f"Insufficient funds for {side} order (required: {required_funds:.2f}, available: {available_funds:.2f})")
                
                # Try reducing position size if possible
                min_qty = 0.01  # Minimum trade size
                adjusted_qty = min(max(available_funds / (current_price * buffer), min_qty), qty)
                if adjusted_qty >= min_qty:
                    qty = adjusted_qty
                    required_funds = current_price * qty
                    logger.info(f"Reduced position size to {qty:.4f}")
                else:
                    logger.warning("Cannot reduce position size further - skipping trade")
                    return
                
            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order synchronously
            self.trading_client.submit_order(order_data)
            logger.info(f"Entered {side} position at {current_price}")
            
        except Exception as e:
            logger.error(f"Error entering position: {str(e)}")

    def exit_position(self, reason: str):
        """Exit current position"""
        try:
            position = self.get_open_position()
            if position:
                close_time = pd.Timestamp.now()
                self.close_timestamps.append(close_time)
                self.trading_client.close_position(self.symbol)
                logger.info(f"Exited position: {reason} on {close_time.strftime('%m/%d/%Y')}")
                
        except Exception as e:
            logger.error(f"Error exiting position: {str(e)}")

    def get_open_position(self):
        """Get current open position for the symbol from Alpaca API"""
        try:
            # Always fetch fresh position data
            positions = self.trading_client.get_all_positions()
            if not isinstance(positions, list):
                return None
                
            position = next((p for p in positions
                          if hasattr(p, 'symbol') and p.symbol == self.symbol.replace('/', '')), None)
            
            # Store position details for reference
            if position:
                self.current_position = {
                    'side': position.side,
                    'qty': position.qty,
                    'entry_price': position.avg_entry_price,
                    'current_value': position.market_value,
                    'timestamp': pd.Timestamp.now()
                }
            else:
                self.current_position = None
                
            return position
            
        except Exception as e:
            logger.error(f"Error getting position: {str(e)}")
            return None

    async def run(self):
        """Start live trading"""
        try:
            # Log initial position status
            position = self.get_open_position()
            if position:
                logger.info(f"Initial position: {position.side} {position.qty} {position.symbol} @ {position.avg_entry_price}")
            else:
                logger.info("No initial position found")
            
            # Initialize data stream with proper credentials
            api_key = os.environ['APCA_API_KEY_ID']
            secret_key = os.environ['APCA_API_SECRET_KEY']
            self.data_stream = CryptoDataStream(api_key, secret_key)
            
            if not self.data_stream:
                raise RuntimeError("Failed to initialize data stream")
                
            self.data_stream.subscribe_bars(self.handle_bar, self.symbol)
            
            # Run stream in background
            self.stream_task = asyncio.create_task(self._run_stream())
            
            # Keep strategy running until 'x' is pressed
            logger.info("Press 'x' to stop trading...")
            try:
                while True:
                    if keyboard.is_pressed('x'):
                        logger.info("Stopping trading...")
                        break
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received - stopping trading...")
                
        except Exception as e:
            logger.error(f"Strategy error: {str(e)}")
        finally:
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass

    async def _run_stream(self):
        """Wrapper for running data stream"""
        try:
            if not self.data_stream:
                raise RuntimeError("Data stream not initialized")
                
            if hasattr(self.data_stream, 'run') and callable(self.data_stream.run):
                # Run synchronous stream in executor
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.data_stream.run)
            else:
                raise RuntimeError("Data stream missing run method")
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")

if __name__ == "__main__":
    strategy = LiveMACDStrategy(
        symbol='ETH/USD',
        timeframe='1Min',
        paper=True
    )
    asyncio.run(strategy.run())