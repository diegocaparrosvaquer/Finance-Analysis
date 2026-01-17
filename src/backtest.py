import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def walk_forward_validation(df, model_fn, features, target, window=252):
    preds = []
    actuals = []

    for i in range(window, len(df)):
        train = df.iloc[i-window:i]
        test = df.iloc[i:i+1]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]

        model = model_fn(X_train, y_train)
        pred = model.predict(X_test)[0]

        preds.append(pred)
        actuals.append(test[target].values[0])

    return np.array(preds), np.array(actuals)
