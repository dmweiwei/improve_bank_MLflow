import mlflow
from sklearn.metrics import accuracy_score, f1_score


def train_model(dict_param: dict, cl_model, x_train, y_train, x_test, y_test):
    """
    Creat differents experiences with differents classification models and hyperparametrers
    :param dict_param: hyperparametres to tune
    :param cl_model: the classification model
    :return:
    """

    with mlflow.start_run():
        cl_model.fit(x_train, y_train)
        y_pred = cl_model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        for k, v in dict_param.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1_score", f1score)