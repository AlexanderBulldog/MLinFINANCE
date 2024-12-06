def func_model_performance(df):
    testDate = "2024-06-01 00:00:00"
    sharpe_train = 0
    sharpe_test = 0

    train = df[(df.index < testDate)].reindex()
    test = df[(df.index >= testDate)].reindex()

    pnl_train = round(train["pnl"].sum(), 6)
    pnl_test = round(test["pnl"].sum(), 6)

    mdd_train = round(train["pnl_dd"].max(), 6)
    mdd_test = round(test["pnl_dd"].max(), 6)

    sharpe_train = round(pnl_train / mdd_train, 6)
    sharpe_test = round(pnl_test / mdd_test, 6)

    print("            train     test")
    print("PNL      : {0: <10}".format(pnl_train), "{0: <10}".format(pnl_test))
    print("MDD      : {0: <10}".format(mdd_train), "{0: <10}".format(mdd_test))
    print("Sharpe   : {0: <10}".format(sharpe_train), "{0: <10}".format(sharpe_test))
    return sharpe_train
