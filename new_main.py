import mlflow
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from settings import hyperF_rl, hyperF_rf, hyperF_adab

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    print(">>> Début du programme")
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

    # with mlflow.start_run():
    #     print("Tuning Logistic Regression...")
    #     clf_rl = RandomizedSearchCV(LogisticRegression(), hyperF_rl, random_state=0)
    #     search_rl = clf_rl.fit(df_copy, y_binaire)
    #
    #     mlflow.log_param("best params", search_rl.best_params_)
    #     mlflow.log_metric("Score", search_rl.best_score_)
    #     mlflow.sklearn.log_model(clf_rl, "bank")

    with mlflow.start_run():
        print("Tuning Random Forest...")
        clf_rf = RandomizedSearchCV(RandomForestClassifier(), hyperF_rf, random_state=0)
        search_rf = clf_rf.fit(df_copy, y_binaire)

        mlflow.log_param("best params", search_rf.best_params_)
        mlflow.log_metric("Score", search_rf.best_score_)
        mlflow.sklearn.log_model(RandomForestClassifier(), "bank")

    with mlflow.start_run():
        print("Tuning AdaBoost...")
        clf_adab = RandomizedSearchCV(AdaBoostClassifier(), hyperF_adab, random_state=0)
        search_adab = clf_adab.fit(df_copy, y_binaire)

        mlflow.log_param("best params", search_adab.best_params_)
        mlflow.log_metric("Score", search_adab.best_score_)
        mlflow.sklearn.log_model(AdaBoostClassifier(), "bank")
    print("Fin du programme")
