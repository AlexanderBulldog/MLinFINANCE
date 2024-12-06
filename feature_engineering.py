import numpy as np
import pandas as pd
import pandas_ta as ta


def func_features(df):

    for length in [6, 12, 16, 20, 25, 30, 50]:
        df.ta(
            kind="SMA", append=True, centered=False, close="closePrice", length=length
        )
        df[f"f_SMA_{length}"] = np.where(df["closePrice"] > df[f"SMA_{length}"], -1, 1)

        df[f"f_trend_{length}"] = np.where(
            df["return"].rolling(window=length).sum() > 0, -1, 1
        )

        df.ta(
            kind="ATR",
            append=True,
            centered=False,
            high="highPrice",
            low="lowPrice",
            volume="volume",
            close="closePrice",
            length=length,
        )
        df[f"f_ATR_{length}"] = np.where(
            df[f"ATRr_{length}"] > df[f"ATRr_{length}"].rolling(window=6).mean(), 1, -1
        )

        df.ta(
            kind="RSI", append=True, centered=False, close="closePrice", length=length
        )
        df["f_rsi"] = np.where(
            df[f"RSI_{length}"] < 20, 1, np.where(df[f"RSI_{length}"] > 80, -1, 0)
        )

        df.ta(
            kind="PVT", append=True, centered=False, volume="volume", close="closePrice"
        )
        df["f_pvt"] = np.where(
            df["PVT"] > df["PVT"].rolling(window=length).mean(), 1, -1
        )

    # df.ta(kind='RSI',append=True,centered=False,close='closePrice',length=length_rsi)
    # df['f_rsi'] = np.where(df[f'RSI_{length_rsi}']<20,1,np.where(df[f'RSI_{length_rsi}']>80,-1,0))

    df.ta(
        kind="MACD",
        append=True,
        centered=False,
        high="highPrice",
        low="lowPrice",
        volume="volume",
        close="closePrice",
        length=6,
    )
    df["f_macd"] = np.where(df[f"MACDh_12_26_9"] > 0, 1, -1)

    df.ta(
        kind="OBV",
        append=True,
        centered=False,
        high="highPrice",
        low="lowPrice",
        volume="volume",
        close="closePrice",
        length=6,
    )
    df[f"f_OBV"] = np.where(df["OBV"] < df["OBV"].rolling(window=12).mean(), -1, 1)

    # df.ta(kind='MFI',append=True,centered=False,high='highPrice',low='lowPrice',volume='volume',close='closePrice',length=length)
    # df['f_mfi'] = np.where(df[f'MFI_{length}']<20,1,np.where(df[f'MFI_{length}']>80,-1,0))

    # df['return_next_2'] = np.where(df['return_next'].shift(-11).rolling(window=12).sum()>0,1,-1)

    return df


# BTC 'length_rsi': '6', 'length_pvt': '36', 'length_trend': '18', 'model_type': 'lr'


def select_features(df):
    features = [col for col in df.columns if col.startswith("f_")]
    return features


def remove_correlated_features(df, threshold=0.75):
    features = [col for col in df if col.startswith("f_")]
    corr_matrix = df[features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df, to_drop
