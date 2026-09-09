import sys
from pathlib import Path

# Add project root to sys.path if running from root directory
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import CryptoDataLoader, BinanceDataLoader, logger

if __name__ == '__main__':
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
