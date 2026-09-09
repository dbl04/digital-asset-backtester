import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class CryptoDataLoader:
    """Robust market data loader with disk persistence for quantitative backtesting.

    Fetches historical OHLCV data from Binance public API using CCXT and caches
    data locally in CSV format to optimize network requests and execution speed.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """Initializes the CryptoDataLoader.

        Args:
            data_dir (str): Relative or absolute directory path to store CSV files. Defaults to 'data'.
        """
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        self.data_dir = Path(data_dir)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        days_back: int = 365
    ) -> pd.DataFrame:
        """Downloads historical OHLCV data directly from Binance API.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): Candle timeframe (e.g., '1d', '1h'). Defaults to '1d'.
            days_back (int): Number of historical days to fetch. Defaults to 365.

        Returns:
            pd.DataFrame: DataFrame indexed by Datetime (UTC) with float columns:
                ['open', 'high', 'low', 'close', 'volume'].

        Raises:
            RuntimeError: Encapsulates CCXT network or exchange errors.
        """
        logger.info("Downloading OHLCV from Binance API: symbol=%s, timeframe=%s, days_back=%d", symbol, timeframe, days_back)

        since_datetime = datetime.now(timezone.utc) - timedelta(days=days_back)
        since_ms = int(since_datetime.timestamp() * 1000)

        raw_ohlcv = []
        limit = 1000
        current_since = since_ms

        try:
            while True:
                remaining_candles = days_back - len(raw_ohlcv) if timeframe.lower() == "1d" else limit
                if remaining_candles <= 0:
                    break

                fetch_limit = min(remaining_candles, limit)

                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=fetch_limit
                )

                if not candles:
                    break

                raw_ohlcv.extend(candles)
                last_timestamp = candles[-1][0]
                current_since = last_timestamp + 1

                if len(candles) < fetch_limit:
                    break

        except ccxt.NetworkError as ne:
            logger.error("Network error while connecting to Binance API: %s", ne)
            raise RuntimeError(f"Network error: {ne}") from ne
        except ccxt.ExchangeError as ee:
            logger.error("Binance exchange error occurred: %s", ee)
            raise RuntimeError(f"Exchange error: {ee}") from ee
        except ccxt.BaseError as be:
            logger.error("CCXT exception encountered: %s", be)
            raise RuntimeError(f"CCXT error: {be}") from be
        except Exception as exc:
            logger.error("Unexpected error fetching market data: %s", exc)
            raise RuntimeError(f"Unexpected error: {exc}") from exc

        if not raw_ohlcv:
            logger.warning("No OHLCV records returned from Binance for %s", symbol)
            empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            empty_df.index = pd.to_datetime(empty_df.index, utc=True)
            empty_df.index.name = 'datetime'
            return empty_df

        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame(raw_ohlcv, columns=columns)

        # Datetime index (UTC)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)

        # Convert price and volume columns to float
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        df[float_cols] = df[float_cols].astype(float)

        # Filter to requested date window
        df = df.loc[df.index >= since_datetime]

        logger.info("Successfully fetched %d OHLCV candles from API for %s", len(df), symbol)
        return df

    # Alias for backward compatibility
    fetch_ohlcv = fetch_historical_data

    def load_data(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        days_back: int = 365
    ) -> pd.DataFrame:
        """Loads market data from local CSV cache if present; downloads from API otherwise.

        Args:
            symbol (str): Trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): Candle timeframe (e.g., '1d'). Defaults to '1d'.
            days_back (int): Number of historical days to fetch if downloading. Defaults to 365.

        Returns:
            pd.DataFrame: DataFrame indexed by parsed Datetime with float columns.
        """
        # Replace '/' with '_' for safe file paths (e.g., 'BTC/USDT' -> 'BTC_USDT')
        clean_symbol = symbol.upper().replace('/', '_')
        file_name = f"{clean_symbol}_{timeframe}.csv"
        file_path = self.data_dir / file_name

        since_datetime = datetime.now(timezone.utc) - timedelta(days=days_back)

        # Check if local CSV file exists
        if file_path.exists():
            logger.info("Cargando datos locales desde archivo CSV: %s", file_path)
            df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = 'datetime'
            
            # Ensure float types
            float_cols = ['open', 'high', 'low', 'close', 'volume']
            df[float_cols] = df[float_cols].astype(float)

            # If cache starts within requested since_datetime (with 1-day tolerance for UTC candle alignment)
            if not df.empty and df.index[0] <= since_datetime + timedelta(days=1):
                return df.loc[df.index >= since_datetime - timedelta(days=1)]
            
            logger.info("El archivo CSV local no cubre el rango solicitado (days_back=%d). Re-descargando de la API...", days_back)

        # If file does not exist or cache incomplete, fetch from API and persist to disk
        logger.info("Descargando datos desde la API para '%s'...", symbol)
        df = self.fetch_historical_data(symbol=symbol, timeframe=timeframe, days_back=days_back)
        
        if not df.empty:
            df.to_csv(file_path)
            logger.info("Datos guardados exitosamente en archivo local: %s", file_path)

        return df


# Alias class name for backwards compatibility
BinanceDataLoader = CryptoDataLoader


if __name__ == "__main__":
    loader = CryptoDataLoader()

    symbol = "BTC/USDT"
    timeframe = "1d"
    days_back = 365

    print("\n--- Ejecución 1: Primer intento de carga (Descarga API / Creación CSV) ---")
    df_first_run = loader.load_data(symbol=symbol, timeframe=timeframe, days_back=days_back)
    print("\nHead (Ejecución 1):")
    print(df_first_run.head())

    print("\n--- Ejecución 2: Segundo intento de carga (Uso de Caché Local CSV) ---")
    df_second_run = loader.load_data(symbol=symbol, timeframe=timeframe, days_back=days_back)
    print("\nHead (Ejecución 2):")
    print(df_second_run.head())

    print("\n--- Verificación del esquema del DataFrame ---")
    print(df_second_run.info())
