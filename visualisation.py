import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def func_plot(df):
    testDate = "2024-06-01 00:00:00"
    df["pnl"] = df["return_next"] * df["predict"]
    df["pnl_cumsum"] = df["pnl"].cumsum()
    df["pnl_cumsum_max"] = df["pnl_cumsum"].cummax()
    df["pnl_dd"] = df["pnl_cumsum_max"] - df["pnl_cumsum"]
    plt.figure(figsize=(20, 5))
    plt.plot(df["pnl_cumsum"], color="orange")
    plt.plot(df["pnl_cumsum_max"], color="red")
    plt.axvline(x=pd.to_datetime(testDate), color="black")
    plt.grid()
    plt.show()


def plot_correlation_heatmap(df, features):
    plt.figure(figsize=(12, 10))
    cor = df[features].corr()
    sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
    plt.title("Correlation Heatmap of Features")
    plt.show()
