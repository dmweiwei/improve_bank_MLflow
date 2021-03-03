import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import mlflow.sklearn

if __name__ == "__main__":
    csv_url = "bank-full.csv"

    # Read csv file
    df = pd.read_csv(csv_url, sep=";")
    # convert y to y binaire : 0 for no and 1 for yes
    df['y_binaire'] = df['y'].apply(lambda y: 1 if y == 'yes' else 0)
    # make a copy of dataframe
    df_copy = df.copy()
    # delete column y
    df_copy = df_copy.drop(["y"], axis=1)
    # convert all categorical columns to numeric
    df_copy = pd.get_dummies(df_copy, drop_first=True)
    # prepare X and y (variables and label)
    y_binaire = df_copy.pop("y_binaire")

    # split dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(df_copy, y_binaire, test_size=0.3, random_state=42)

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]

    dict_params_rf = dict(n_estimators=n_estimators, max_depth=max_depth,
                          min_samples_split=min_samples_split,
                          min_samples_leaf=min_samples_leaf)

    print("Début de l'entraînement...")
    for n_estimator in n_estimators:
        for max_d in max_depth:
            for min_sl in min_samples_leaf:
                for min_ss in min_samples_split:
                    rf = RandomForestClassifier(n_estimators=n_estimator,
                                                max_depth=max_d,
                                                min_samples_leaf=min_sl,
                                                min_samples_split=min_ss)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    f1score = f1_score(y_test, y_pred)

                    for k, v in dict_params_rf.items():
                        mlflow.log_param(k, v)
                    mlflow.log_metric("Accuracy", accuracy)
                    mlflow.log_metric("F1_score", f1score)
                    mlflow.sklearn.log_model(rf, "bank_rf")

                    print("Accuracy:" + str(accuracy) + '\n' + "F1_score:" + str(f1score))

    print("Fin de l'entraînement")