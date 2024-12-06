import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    mutual_info_regression,
)
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def func_train_model(modelType, df, testDate, params=None):
    if modelType == "rf":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next_class"]

        model = Pipeline(
            [
                ("feature_selection", SelectFromModel(LinearSVC(penalty="l1"))),
                ("classification", RandomForestClassifier()),
            ]
        )
        model.fit(X_train, y_train)

        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "cb":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        model = CatBoostRegressor(
            iterations=1000,
            loss_function="RMSE",
            depth=15,
            verbose=False,
            early_stopping_rounds=10,
            random_seed=1,
        )
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "cb_gs":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        cat = CatBoostRegressor(verbose=False)
        params = {
            "iterations": [1000],
            #'depth':[5,10]
            "l2_leaf_reg": np.logspace(-20, -10, 3),
            "early_stopping_rounds": [10],
        }
        model = GridSearchCV(estimator=cat, cv=5, param_grid=params)
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "cb_w":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]

        sample_weight = np.array([0])
        dfLength = len(X_train)
        rowWeigth = 0
        for i in range(dfLength - 1):
            sample_weight = np.append(sample_weight, rowWeigth)
            rowWeigth = rowWeigth + 1 / dfLength

        model = CatBoostRegressor(
            iterations=1000,
            loss_function="RMSE",
            depth=15,
            verbose=False,
            early_stopping_rounds=10,
            random_seed=1,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "lr":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        print(train.shape, test.shape)
        X_train = train[features]
        y_train = train["return_next"]
        print(X_train.shape, y_train.shape)
        model = LinearRegression()
        model.fit(X_train, y_train)
        print(model.coef_)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "lr_sl":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        model = LinearRegression()
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )
        df["pnl"] = (df["predict"] * df["return_next"]).shift(1)
        df["predict"] = np.where(
            df["pnl"].rolling(window=100).sum() < 0, 0.5 * df["predict"], df["predict"]
        )

    if modelType == "lr_ig":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]

        # mutual_info = mutual_info_regression(X_train,y_train)
        # mutual_info = pd.Series(mutual_info)
        # mutual_info.index = features
        # mutual_info.sort_values(ascending=False)

        bestCols = SelectKBest(mutual_info_regression, k=10)
        bestCols.fit(X_train, y_train)
        features = list(X_train.columns[bestCols.get_support()])

        X_train = train[features]
        model = LinearRegression()
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "lr_rfe":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        reg = LinearRegression()
        model = RFE(reg, n_features_to_select=15)
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )
    if modelType == "lr_reg":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )
    if modelType == "lr_cor":
        features = [col for col in df if col.startswith("f_")]
        corr_matrix = df[features].corr().abs()
        corr_matrix = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_)
        )
        to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
        features = [f for f in features if f not in to_drop]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        model = LinearRegression()
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )
    if modelType == "dt":
        features = [col for col in df if col.startswith("f_")]
        train = df[df.index < testDate]
        test = df[df.index >= testDate]
        X_train = train[features]
        y_train = train["return_next_class"]
        if params is None:
            model = DecisionTreeClassifier(random_state=0)
        else:
            model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )
    if modelType == "elnet_cv":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        alpha = [0.0001, 0.001, 0.1, 1, 10, 100]
        max_iter = [1000, 10000]
        l1_ratio = np.arange(0.0, 1.0, 0.1)
        tol = [0.5]
        elnet_gscv = GridSearchCV(
            estimator=ElasticNet(),
            param_grid={
                "alpha": alpha,
                "max_iter": max_iter,
                "l1_ratio": l1_ratio,
                "tol": tol,
            },
            scoring="r2",
            cv=5,
        )
        elnet_gscv.fit(X_train, y_train)
        model = ElasticNet(
            alpha=elnet_gscv.best_params_["alpha"],
            max_iter=elnet_gscv.best_params_["max_iter"],
            l1_ratio=elnet_gscv.best_params_["l1_ratio"],
            tol=elnet_gscv.best_params_["tol"],
        )
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    if modelType == "elnet":
        features = [col for col in df if col.startswith("f_")]
        train = df[(df.index < testDate)].reindex()
        test = df[(df.index >= testDate)].reindex()
        X_train = train[features]
        y_train = train["return_next"]
        model = ElasticNet()
        model.fit(X_train, y_train)
        df["predict"] = model.predict(df[features])
        df["predict"] = np.where(
            df["predict"] > 0, 1, np.where(df["predict"] < 0, -1, 0)
        )

    df["pnl"] = df["predict"] * df["return_next"]
    df["pnl_cumsum"] = df["pnl"].cumsum()
    df["pnl_cumsum_max"] = df["pnl_cumsum"].cummax()
    df["pnl_dd"] = df["pnl_cumsum_max"] - df["pnl_cumsum"]
    return df
