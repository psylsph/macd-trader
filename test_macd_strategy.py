import unittest
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from macd_strategy import MACDStrategy, AdaptiveMACDDeluxe
from test_data import TestData

class TestMACDStrategy(unittest.TestCase):
    def setUp(self):
        """Set up the backtest environment before each test"""
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.set_cash(100000)
        self.cerebro.broker.setcommission(commission=0.002)
        self.strategy = None
        # Override strategy settings for testing
        class TestStrategy(MACDStrategy):
            def __init__(self):
                super().__init__()
                self.buy_signals = []
                self.sell_signals = []
        self.strategy_class = TestStrategy

    def create_test_data(self, prices, base_price=100, volumes=None):
        """Create a test data feed with given prices
        
        Args:
            prices (list): List of closing prices
            volumes (list, optional): List of volumes. Defaults to None.
        """
        if volumes is None:
            volumes = [1000] * (400 + len(prices))  # Match full dataset length (warmup + test data)
            
        # Generate synthetic data with proper OHLC
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        
        # Start with enough data for indicator warmup
        start_date = datetime(2024, 1, 1) - timedelta(days=400)  # Extra days for warmup
        current_price = base_price
        prev_price = base_price
        
        for i in range(400 + len(prices)):  # 400 warmup bars + actual test data
            dates.append(start_date + timedelta(days=i))
            
            if i < 400:  # Warmup data
                # Introduce more significant volatility during warmup, especially at the start
                # Base oscillation for overall movement
                base_oscillation = 0.02 * np.sin(i / 8.0) # Faster, larger initial wave
                
                # Trend component, ensuring it's not zero at the very start for movement
                trend_component = (i + 1) * 0.00020 
                
                # Noise component
                noise_component = np.random.normal(0, 0.0025) # Slightly larger noise
                
                # Finer sine wave for texture
                fine_sine_wave = 0.0075 * np.sin(i / 18.0)
                
                current_price = base_price * (1 + trend_component + noise_component + fine_sine_wave + base_oscillation)
                current_price = max(1.0, current_price) # Ensure price is positive
            else:
                current_price = prices[i - 400]

            prev_price = current_price

            # Create OHLC data ensuring non-zero ranges and minimum volatility
            min_daily_spread = 0.005 # Ensure at least 0.5% spread between high and low
            open_price = current_price
            
            # Ensure high is always above open and close, and low is always below
            rand_high_factor = np.random.uniform(1.001, 1.01) # Random factor for high
            rand_low_factor = np.random.uniform(0.99, 0.999)   # Random factor for low
            
            high_candidate = current_price * rand_high_factor
            low_candidate = current_price * rand_low_factor
            close_candidate = current_price * (1 + np.random.normal(0, 0.0015)) # Slight variation for close

            # Ensure high > low by at least min_daily_spread
            if (high_candidate - low_candidate) / current_price < min_daily_spread:
                high_candidate = current_price * (1 + min_daily_spread / 2)
                low_candidate = current_price * (1 - min_daily_spread / 2)

            high_price = max(high_candidate, open_price, close_candidate)
            low_price = min(low_candidate, open_price, close_candidate)
            
            # Final check to ensure high > low
            if high_price <= low_price:
                high_price = low_price + (current_price * 0.001) # Add a small fixed amount if they are equal

            close_price = np.clip(close_candidate, low_price, high_price) # Ensure close is within high/low
            open_price = np.clip(open_price, low_price, high_price) # Ensure open is within high/low
            
            # Ensure OHLC relationship
            actual_high = max(open_price, high_price, low_price, close_price)
            actual_low = min(open_price, high_price, low_price, close_price)
            
            # If by any chance low >= high, adjust them
            if actual_low >= actual_high:
                actual_high = actual_low + (current_price * 0.001)


            opens.append(open_price)
            highs.append(actual_high) # Use adjusted high
            lows.append(actual_low)   # Use adjusted low
            closes.append(close_price)
            
        data = {
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }
        df = pd.DataFrame(data)
        
        # Save test data to CSV
        df.to_csv('test_data.csv', index=False)
        
        # Create data feed
        data = TestData(
            dataname='test_data.csv',
            fromdate=datetime(2024, 1, 1),
            todate=datetime(2024, 12, 31)
        )
        return data

    def test_entry_conditions(self):
        """Test if entry conditions trigger correctly"""
        # Create uptrend scenario
        # Price above EMA200, RSI oversold and turning up, MACD conditions met
        prices = []
        base_price = 100
        
        # Generate 250 days of price data
        # Hardcode rsi_period to default value (14) to avoid access issues with params object
        rsi_period = 14 # Used for initial noisy data generation phase
        
        # Define phases for price generation
        # Strategy min_period is 200. MACD warmup is ~34.
        # Target entry signal around bar 230-240.
        
        # Phase 0: Initial noisy data (first ~19 bars)
        phase0_end = rsi_period + 5 
        
        # Phase 1: Establish uptrend for EMA conditions (up to bar 199)
        # Price > EMA200, EMA50 > EMA200
        phase1_end = 199 
        
        # Phase 2: Sharp pullback for RSI oversold (e.g., 15 bars, up to bar 214)
        phase2_duration = 15
        phase2_end = phase1_end + phase2_duration
        
        # Phase 3: Bounce and recovery for RSI turn up & MACD cross (e.g., 15 bars, up to bar 229)
        phase3_duration = 15
        phase3_end = phase2_end + phase3_duration

        # Phase 4: Stabilization / slight continuation (remaining bars up to 249)

        # Store prices at phase transitions
        price_at_phase1_end = 0
        price_at_phase2_end = 0
        price_at_phase3_end = 0

        for i in range(250):
            # Phase 0: Initial noisy data
            if i < phase0_end:
                price = base_price * (1 + (i+1) * 0.001) # Slower initial rise
                if i > 0 and i % 4 == 0 : # Make every 4th bar a bit lower
                    price *= 0.995
            # Phase 1: Strong uptrend for EMA crossover
            elif i <= phase1_end:
                # Start from where phase0 left off, or a defined point
                # For simplicity, let phase1 build on base_price
                # Ensure this phase is long enough for EMAs to cross and price to be above EMA200
                # Example: 0.25% daily increase
                price = base_price * (1 + (i - phase0_end) * 0.0035) 
                if i == phase1_end:
                    price_at_phase1_end = price
            # Phase 2: Sharp pullback for RSI oversold
            elif i <= phase2_end:
                # Pullback from price_at_phase1_end
                # Example: 1.5% daily decline relative to pullback duration
                pullback_progress = (i - phase1_end) / phase2_duration
                price = price_at_phase1_end * (1 - pullback_progress * 0.20) # 20% total pullback
                if i == phase2_end:
                    price_at_phase2_end = price
            # Phase 3: Bounce and recovery
            elif i <= phase3_end:
                # Bounce from price_at_phase2_end
                # Example: 0.8% daily increase relative to bounce duration
                bounce_progress = (i - phase2_end) / phase3_duration
                price = price_at_phase2_end * (1 + bounce_progress * 0.15) # 15% total bounce from low
                if i == phase3_end:
                    price_at_phase3_end = price
            # Phase 4: Continued uptrend or stabilization
            else:
                # Continue from price_at_phase3_end
                price = price_at_phase3_end * (1 + (i - phase3_end) * 0.001) # Gentle rise
            
            # Ensure price is valid
            price = max(price, 1.0)
            prices.append(price)

        # Add test data to Cerebro with base price
        self.cerebro.adddata(self.create_test_data(prices, base_price=base_price))
        
        # Add our test strategy
        self.cerebro.addstrategy(self.strategy_class)
        
        # Run the test
        results = self.cerebro.run()
        strategy = results[0]
        
        # Assert we have at least one trade, with debug info on failure
        if len(strategy.buy_signals) == 0:
            raise AssertionError(
                f"No buy signals were generated\nBuy signals: {strategy.buy_signals}"
            )
        self.assertTrue(len(strategy.buy_signals) > 0, "No buy signals were generated")

    def test_exit_conditions(self):
        """Test if exit conditions trigger correctly"""
        # Create scenario for testing exits
        prices = []
        base_price = 100
        
        # Generate price data for exit testing
        # Hardcode rsi_period to default value (14) to avoid access issues with params object
        rsi_period = 14

        for i in range(250):
            price = base_price # Initialize price, will be overwritten based on i

            # Price pattern for exit test
            # Phase 0: Initial volatile data for indicator warmup (first rsi_period + 5 bars, e.g., 19 bars for rsi_period=14)
            if i < rsi_period + 5: 
                if i == 0:
                    price = base_price * 1.005  # Start with a small rise from base_price
                else:
                    # Subsequent prices are based on the previously appended price in the `prices` list
                    prev_close = prices[i-1] # Get the actual previous closing price
                    if i == 5 or i == 10 or i == 15: # Force a distinct drop on these specific bars
                        price = prev_close * 0.985 # Drop 1.5% from previous day's generated price
                    elif i % 3 == 1 : # Bars 1, 4, 7, 13, 16 - try to go up
                        price = prev_close * (1 + np.random.uniform(0.001, 0.004)) # Small rise
                    elif i % 3 == 2 : # Bars 2, 8, 11, 14, 17 - try to go down
                        price = prev_close * (1 - np.random.uniform(0.0005, 0.003)) # Small dip
                    else: # Bars 0 (handled), 3, 6, 9, 12, 18 - moderate rise
                        price = prev_close * (1 + np.random.uniform(0.0005, 0.0025))
                price = max(price, 1.0) # Ensure price is not zero or negative to avoid issues with OHLC generation
            # Phase 1: Strong uptrend to trigger entry (up to bar 99)
            elif i < 100:
                price = base_price * (1 + i * 0.008) # 0.8% daily increase
            # Phase 2: Consolidation/Slight further rise (50 bars)
            elif i < 150:
                start_consolidation_price = base_price * (1 + 99 * 0.008)
                price = start_consolidation_price * (1 + (i-99) * 0.001 + 0.02 * np.sin((i-99)*0.1))
            # Phase 3: Sharp rally to hit profit targets (30 bars)
            elif i < 180:
                start_rally_price = base_price * (1 + 99 * 0.008) * (1 + 49 * 0.001 + 0.02 * np.sin(49*0.1))
                price = start_rally_price * (1 + (i - 149) * 0.01) # 1% daily increase
            # Phase 4: Sharp reversal for exit signal (remaining bars)
            else:
                if i == 180: # Capture peak price
                    self._peak_price_exit = base_price * (1 + 99 * 0.008) * (1 + 49 * 0.001 + 0.02 * np.sin(49*0.1)) * (1 + 29 * 0.01)
                peak = self._peak_price_exit if hasattr(self, '_peak_price_exit') else price # Fallback if not set
                price = peak * (1 - (i - 179) * 0.015) # 1.5% daily decline
            
            # Ensure price is valid
            price = max(price, 1.0)
            prices.append(price)

        # Add test data to Cerebro
        self.cerebro.adddata(self.create_test_data(prices))
        
        # Add our test strategy
        self.cerebro.addstrategy(self.strategy_class)
        
        # Run the test
        results = self.cerebro.run()
        strategy = results[0]
        
        # Assert we have both entry and exit signals, with debug info on failure
        if len(strategy.sell_signals) == 0:
            raise AssertionError(
                f"No sell signals were generated\nBuy signals: {strategy.buy_signals}\nSell signals: {strategy.sell_signals}"
            )
        self.assertTrue(len(strategy.sell_signals) > 0, "No sell signals were generated")

    def test_position_sizing(self):
        """Test if position sizing works correctly"""
        # Create uptrend scenario with enough data for all indicators
        prices = []
        base_price = 100
        for i in range(250):
            # Steady uptrend with small random variations
            price = base_price + i * 0.2 + np.random.normal(0, 0.1)
            prices.append(max(price, 1.0))  # Ensure no negative or zero prices
        
        # Add test data to Cerebro
        self.cerebro.adddata(self.create_test_data(prices))
        
        # Set initial capital
        self.cerebro.broker.set_cash(100000)
        
        # Add our test strategy
        self.cerebro.addstrategy(self.strategy_class)
        
        # Run the test
        results = self.cerebro.run()
        strategy = results[0]
        
        # Check if position size is calculated based on risk
        if len(strategy.buy_signals) > 0:
            # Get the position size from the first trade
            size = strategy.position.size if strategy.position else 0
            self.assertTrue(size > 0, "Position size was not calculated correctly")

if __name__ == '__main__':
    unittest.main()
