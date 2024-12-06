from datetime import datetime

import pandas as pd
from binance.client import Client


def load_btcusdt_data(start_date="2022-01-01 00:00:00", interval="30m"):
    client = Client()
    klines = client.get_historical_klines("BTCUSDT", interval, start_date)

    df = pd.DataFrame(
        klines,
        columns=[
            "openTime",
            "openPrice",
            "highPrice",
            "lowPrice",
            "closePrice",
            "volume",
            "closeTime",
            "quoteAssetVolume",
            "NumberOfTrades",
            "TakerBaseVolume",
            "TakerQuoteVolume",
            "Ignore",
        ],
    )

    df["closeTime"] = pd.to_datetime(df["closeTime"], unit="ms")

    df = df.astype(
        {
            "closePrice": "float",
            "openPrice": "float",
            "highPrice": "float",
            "lowPrice": "float",
            "volume": "float",
        }
    )

    df.set_index("closeTime", inplace=True)
    df.sort_index(inplace=True)

    df["return"] = df["closePrice"].pct_change()
    df["return_next"] = df["return"].shift(-1).fillna(0)

    return df


if __name__ == "__main__":
    df = load_btcusdt_data()
    df.to_csv("data/btcusdt_30m.csv")
