import argparse
import os
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv

# Alpaca SDK imports
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError

# Load environment variables from .env file at the script level
load_dotenv()

def get_api_keys(args_api_key, args_secret_key):
    """Helper to get API keys from args or environment variables."""
    api_key = args_api_key or os.getenv("APCA_API_KEY_ID")
    secret_key = args_secret_key or os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("Error: API Key ID and Secret Key are required.")
        print("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables,")
        print("or provide them via --api-key and --secret-key arguments.")
        return None, None
    return api_key, secret_key

def parse_dates(start_date_str, end_date_str):
    """Parses date strings into timezone-aware datetime objects."""
    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"Error: Invalid start_date format. Please use YYYY-MM-DD. Got: {start_date_str}")
        return None, None

    if end_date_str:
        try:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        except ValueError:
            print(f"Error: Invalid end_date format. Please use YYYY-MM-DD. Got: {end_date_str}")
            return None, None
    else:
        end_dt = datetime.now(timezone.utc)

    if start_dt >= end_dt:
        print(f"Error: Start date ({start_date_str}) must be before end date ({end_date_str or 'now'}).")
        return None, None

    return start_dt, end_dt

def process_and_save_data(bars_response, symbol, output_file):
    """Processes API response data and saves it to CSV if requested."""
    if not bars_response or not bars_response.data or symbol.upper() not in bars_response.data:
        print(f"No data found for {symbol} in the specified date range.")
        return

    bars_df = bars_response.df
    if bars_df.empty:
        print(f"No data returned for {symbol} in the specified date range.")
        return

    if isinstance(bars_df.index, pd.MultiIndex):
        # For multi-symbol requests (though we use single symbol here),
        # or if the symbol is part of the index by default
        if symbol.upper() in bars_df.index.get_level_values(0): # Check if symbol is in the first level of MultiIndex
             bars_df = bars_df.loc[symbol.upper()]
        bars_df = bars_df.reset_index()
    else:
        bars_df = bars_df.reset_index() # 'timestamp' is usually the index

    print(f"\nSuccessfully fetched {len(bars_df)} bars for {symbol}:")
    print(bars_df.head())
    print("...")
    print(bars_df.tail())

    if output_file:
        try:
            bars_df.to_csv(output_file, index=False)
            print(f"\nData saved to {output_file}")
        except Exception as e:
            print(f"Error saving data to {output_file}: {e}")


def download_stock_hourly_data(api_key: str,
                               secret_key: str,
                               symbol: str,
                               start_dt: datetime,
                               end_dt: datetime,
                               output_file = None):
    """Downloads 1-hour historical stock data from Alpaca."""
    client = StockHistoricalDataClient(api_key, secret_key)
    print(f"Fetching 1-hour STOCK data for {symbol} from {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol.upper(),
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start_dt,
        end=end_dt,
        adjustment='all'
    )
    try:
        bars_response = client.get_stock_bars(request_params)
        process_and_save_data(bars_response, symbol, output_file)
    except APIError as e:
        print(f"Alpaca API Error (Stock): {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
    except Exception as e:
        print(f"An unexpected error occurred (Stock): {e}")


def download_crypto_data(api_key: str,
                                secret_key: str,
                                symbol: str, # e.g., "BTC/USD"
                                start_dt: datetime,
                                end_dt: datetime,
                                time_frame: TimeFrame,
                                output_file = None):
    """Downloads 1-hour historical crypto data from Alpaca."""
    client = CryptoHistoricalDataClient(api_key, secret_key)
    print(f"Fetching {time_frame} CRYPTO data for {symbol} from {start_dt.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}...")

    # Crypto symbols are often pairs like "BTC/USD". The API expects them as is.
    request_params = CryptoBarsRequest(
        symbol_or_symbols=symbol.upper(), # Ensure uppercase, e.g., BTC/USD
        timeframe=time_frame,
        start=start_dt,
        end=end_dt
        # No 'adjustment' parameter for crypto
    )
    try:
        bars_response = client.get_crypto_bars(request_params)
        # For crypto, symbol in bars_response.data might be "BTCUSD" if input was "BTC/USD"
        # The .df method usually handles this well.
        # If symbol needs to be transformed (e.g. "BTC/USD" to "BTCUSD" for key lookup),
        # this is where you'd do it, but .df is generally robust.
        process_and_save_data(bars_response, symbol.upper(), output_file)
    except APIError as e:
        print(f"Alpaca API Error (Crypto): {e}")
        print(f"Response text: {e.response.text if e.response else 'N/A'}")
    except Exception as e:
        print(f"An unexpected error occurred (Crypto): {e}")


def main():
    parser = argparse.ArgumentParser(description="Download 1-hour chart data from Alpaca for stocks or crypto.")
    parser.add_argument("symbol", help="Asset symbol (e.g., AAPL for stocks, BTC/USD for crypto).")
    parser.add_argument("start_date", help="Start date for data fetching (YYYY-MM-DD).")
    parser.add_argument("--asset-class", "-ac", choices=['stock', 'crypto'], default='crypto',
                        help="Type of asset: 'stock' or 'crypto'. Default: 'stock'.")
    parser.add_argument("--end-date", help="End date for data fetching (YYYY-MM-DD). Defaults to now.", default=None)
    parser.add_argument("--output-file", "-o", help="Path to save the data as CSV (e.g., data.csv).", default="data.csv")
    parser.add_argument("--api-key", help="Alpaca API Key ID (overrides .env).", default=None)
    parser.add_argument("--secret-key", help="Alpaca API Secret Key (overrides .env).", default=None)

    args = parser.parse_args()

    api_key, secret_key = get_api_keys(args.api_key, args.secret_key)
    if not api_key or not secret_key:
        return

    start_dt, end_dt = parse_dates(args.start_date, args.end_date)
    if not start_dt or not end_dt:
        return

    if args.asset_class == 'stock':
        download_stock_hourly_data(
            api_key=api_key,
            secret_key=secret_key,
            symbol=args.symbol,
            start_dt=start_dt,
            end_dt=end_dt,
            output_file=args.output_file
        )
    elif args.asset_class == 'crypto':
        # Ensure crypto symbols are in the format "BASE/QUOTE", e.g., "BTC/USD"
        if '/' not in args.symbol:
            print("Warning: Crypto symbol might be incorrect. Expected format like 'BTC/USD' or 'ETH/EUR'.")
            # Consider adding more robust validation or transformation if needed
        download_crypto_data(
            api_key=api_key,
            secret_key=secret_key,
            symbol=args.symbol, # Pass as is, e.g., "BTC/USD"
            start_dt=start_dt,
            end_dt=end_dt,
            time_frame=TimeFrame(1, TimeFrameUnit.Hour),
            output_file=args.output_file
        )
    else:
        print(f"Error: Unknown asset class '{args.asset_class}'. Choose 'stock' or 'crypto'.")

if __name__ == "__main__":
    main()