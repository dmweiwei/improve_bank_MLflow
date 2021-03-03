
"""Hyperparameters of Logistic Regression"""
hyperF_rl = {'C': [1.00000000e-04, 7.74263683e-04, 5.99484250e-03, 4.64158883e-02,
                   3.59381366e-01, 2.78255940e+00, 2.15443469e+01, 1.66810054e+02,
                   1.29154967e+03, 1.00000000e+04],
             'solver': ['lbfgs', 'liblinear', 'sag', 'saga']}

"""Hyperparameters of Random Forest"""
n_estimators = [10, 50, 100, 300]
max_depth = [1, 3, 5, 8, 15]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 5]

hyperF_rf = dict(n_estimators = n_estimators, max_depth = max_depth,
                 min_samples_split = min_samples_split,
                 min_samples_leaf = min_samples_leaf)

"""Hyperparameters of AdaBoost"""
n_estimators = [100, 300, 500, 800]
learning_rate = [0.00001, 0.0001, 0.001]

hyperF_adab = dict(n_estimators = n_estimators, learning_rate = learning_rate)