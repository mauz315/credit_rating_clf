
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from lag_selector import select_best_lags
from granger_shift import granger_shift, n_differences, stationary_df, adf_test
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA
import hyperopt
from hyperopt import tpe, fmin, hp, space_eval
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import PredefinedHoldoutSplit

file_name = "Datos"
main_path = ("C:\\Users\\juan.estrada\\Desktop\\Temporal")
file_path = (file_name + ".xlsx")
sheet_name = "Python"

percent_train = 0.75
max_evals = 600
eval_metric = 'mse'
metrics_names = {'mse': 'neg_mean_squared_error', 'r2': 'r2'}

dataset = pd.read_excel(main_path + "\\" + file_path, sheet_name, index_col = 0, parse_dates = True)

def create_target(dataset, target_as_string, timesteps = 1):
    dataset['Target'] = dataset.loc[:, target_as_string].shift(-timesteps)
    cols = dataset.columns.values.tolist()
    cols = cols[-1:] + cols[:-1]
    dataset = dataset[cols]
    return dataset.dropna()

dataset = create_target(dataset = dataset, target_as_string = 'IGAEINDX Index', timesteps = 1)

length_train = int(len(dataset)*percent_train)

colnames = dataset.columns.values.tolist()
target = colnames[0]
predictors = colnames[1:]

validation_indices = dataset.iloc[length_train:].index.values
piter = PredefinedHoldoutSplit(validation_indices)
n_features = dataset.shape[1]

# Data preprocessing functions

def split_train_test(dataset, percent_train):
    length_train = int(len(dataset)*percent_train)
    train_set = dataset.iloc[:length_train]
    test_set = dataset.iloc[length_train:]
    return train_set, test_set

def no_transform(dataset):
    X = dataset[predictors]
    Y = dataset[target]
    return X, Y

def transform(dataset):
    train_set, test_set = split_train_test(dataset, percent_train)
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = colnames)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = colnames)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def standard_scaler(dataset):
    train_set, test_set = split_train_test(dataset, percent_train)
    scaler = StandardScaler()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = colnames)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = colnames)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def robust_scaler(dataset):
    train_set, test_set = split_train_test(dataset, percent_train)
    scaler = RobustScaler()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = colnames)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = colnames)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def quantile_transformer(dataset, quantiles):
    train_set, test_set = split_train_test(dataset, percent_train)
    scaler = QuantileTransformer(n_quantiles = quantiles)
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = colnames)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = colnames)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def power_transformer(dataset):
    train_set, test_set = split_train_test(dataset, percent_train)
    scaler = PowerTransformer()
    scaler.fit(train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(train_set), columns = colnames)
    scaled_test_set = pd.DataFrame(scaler.transform(test_set), columns = colnames)
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df[predictors]
    Y = scaled_df[target]
    return X, Y, scaler

def pca_transform(dataset, components):
    train_set, test_set = split_train_test(dataset, percent_train)
    train_set = train_set[predictors]
    test_set = test_set[predictors]
    scaler_standard = StandardScaler()
    scaler_standard.fit(train_set)
    standardized_train_set = scaler_standard.transform(train_set)
    standardized_test_set = scaler_standard.transform(test_set)
    scaler = PCA(n_components = components)
    scaler.fit(standardized_train_set)
    scaled_train_set = pd.DataFrame(scaler.transform(standardized_train_set))
    scaled_test_set = pd.DataFrame(scaler.transform(standardized_test_set))
    scaled_df = pd.concat([scaled_train_set, scaled_test_set])
    X = scaled_df
    Y = dataset[target]
    return X, Y, scaler

def inverse_transform(data, scaler, n_features, preprocessing, pca_data = None):
    data = data.reshape(-1, 1)
    for i in range(data.shape[1]):
        if preprocessing != 'PCA':
            tmp = np.zeros((data.shape[0], n_features))
        else:
            tmp = np.zeros((data.shape[0], pca_data.shape[1]))
        tmp[:, 0] = data[:, i]
        data[:, i] = scaler.inverse_transform(tmp)[:, 0]
    return data

# Hyperparameter search spaces

search_space = hp.choice('regressor',[
    {
        'model': RandomForestRegressor,
        'preprocessing': hp.choice('1_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'n_estimators': sample(scope.int(hp.quniform('n_estimators', 16, 256, 1))),
        'max_depth': sample(scope.int(hp.quniform('max_depth', 3, 30, 1))),
        'min_samples_split': sample(scope.int(hp.quniform('min_samples_split', 3, 50, 1))),
        'min_samples_leaf': sample(scope.int(hp.quniform('min_samples_leaf', 3, 50, 1))),
        'min_weight_fraction_leaf': hp.uniform('2_min_weight_fraction_leaf', 0, 0.5),
        'max_features': hp.choice('1_max_features', ['auto', 'sqrt', 'log2', None,]),
        'max_leaf_nodes': sample(scope.int(hp.quniform('max_leaf_nodes', 2, 50, 1)))}
    },
    {
        'model': AdaBoostRegressor,
        'preprocessing': hp.choice('2_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'n_estimators': sample(scope.int(hp.quniform('n_estimators', 16, 256, 1))),
        'learning_rate': hp.loguniform('1_learning_rate', np.log(0.005), np.log(0.2)),
        'loss': hp.choice('1_loss', ['linear', 'square', 'exponential']),
        'base_estimator': {'max_depth': sample(scope.int(hp.quniform('max_depth', 3, 30, 1))),
        'min_samples_split': sample(scope.int(hp.quniform('min_samples_split', 3, 50, 1))),
        'min_samples_leaf': sample(scope.int(hp.quniform('min_samples_leaf', 3, 50, 1))),
        'min_weight_fraction_leaf': hp.uniform('1_min_weight_fraction_leaf', 0, 0.5),
        'max_features': hp.choice('3_max_features', ['auto', 'sqrt', 'log2', None,])}}
    },
    {
        'model': ExtraTreesRegressor,
        'preprocessing': hp.choice('3_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'n_estimators': sample(scope.int(hp.quniform('n_estimators', 16, 256, 1))),
        'max_depth': sample(scope.int(hp.quniform('max_depth', 3, 30, 1))),
        'min_samples_split': sample(scope.int(hp.quniform('min_samples_split', 3, 50, 1))),
        'min_samples_leaf': sample(scope.int(hp.quniform('min_samples_leaf', 3, 50, 1))),
        'min_weight_fraction_leaf': hp.uniform('3_min_weight_fraction_leaf', 0, 0.5),
        'max_features': hp.choice('2_max_features', ['auto', 'sqrt', 'log2']),
        'max_leaf_nodes': sample(scope.int(hp.quniform('max_leaf_nodes', 2, 15, 1)))}
    },
    {
        'model': GradientBoostingRegressor,
        'preprocessing': hp.choice('4_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'loss': hp.choice('2_loss', ['ls', 'lad', 'huber', 'quantile']),
        'learning_rate': hp.loguniform('2_learning_rate', np.log(0.005), np.log(0.2)),
        'n_estimators': sample(scope.int(hp.quniform('n_estimators', 16, 256, 1))),
        'subsample': hp.uniform('1_subsample', 0.5, 1),
        'min_samples_split': sample(scope.int(hp.quniform('min_samples_split', 3, 50, 1))),
        'max_depth': sample(scope.int(hp.quniform('max_depth', 3, 12, 1))),
        'tol': hp.uniform('1_tol', 1e-5, 1e-2)}
    },
    {
        'model': SGDRegressor,
        'preprocessing': hp.choice('5_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'loss': hp.choice('3_loss', ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
        'penalty': hp.choice('penalty', [None, 'l1', 'l2', 'elasticnet']),
        'alpha': hp.uniform('1_alpha', 0.00001, 0.001),
        'l1_ratio': sample(scope.int(hp.uniform('l1_ratio', 0.05, 1))),
        'tol': sample(scope.int(hp.uniform('2_tol', 1e-5, 1e-2))),
        'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
        'power_t': sample(scope.int(hp.uniform('power_t', 0.3, 0.7)))}
    },
    {
        'model': ElasticNet,
        'preprocessing': hp.choice('6_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'alpha': hp.loguniform('2_alpha', np.log(0.1), np.log(30)),
        'l1_ratio': hp.uniform('l1_ratio', 0.1, 1),
        'tol': hp.uniform('3_tol', 1e-5, 1e-2)}
    },
    {
        'model': Ridge,
        'preprocessing': hp.choice('7_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'alpha': hp.loguniform('3_alpha', np.log(0.1), np.log(30)),
        'tol': hp.uniform('4_tol', 1e-5, 1e-2),
        'solver': hp.choice('solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])}
    },
    {
        'model': KNeighborsRegressor,
        'preprocessing': hp.choice('8_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'n_neighbors': sample(scope.int(hp.quniform('n_neighbors', 3, 15, 1))),
        'weights': hp.choice('1_weights', ['uniform', 'distance']),
        'algorithm': hp.choice('1_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'leaf_size': sample(scope.int(hp.quniform('leaf_size', 10, 50, 1))),
        'p': sample(scope.int(hp.quniform('p', 1, 10, 1)))}
    },
    {
        'model': GaussianProcessRegressor,
        'preprocessing': hp.choice('9_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'alpha': hp.uniform('4_alpha', 1e-12, 1e-8)}
    },
    {
        'model': SVR,
        'preprocessing': hp.choice('10_preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, len(colnames) - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, len(colnames) - 1, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 5, 15, 1))),
        'params': {'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': sample(scope.int(hp.quniform('degree', 1, 6, 1))),
        'gamma': hp.choice('1_gamma', ['auto', 'scale']),
        'tol': hp.uniform('5_tol', 1e-5, 1e-2),
        'C': hp.uniform('C', 0.3, 5),
        'shrinking': hp.choice('shrinking', [True, False])}
    },
    {
        'model': xgb,
        'preprocessing': hp.choice('preprocessing', ['NoTransform', 'MinMaxScaler',
                         'StandardScaler', 'RobustScaler', 'QuantileTransformer',
                         'PowerTransformer', 'PCA']),
        'k_features': sample(scope.int(hp.quniform('k_features', 1, dataset.shape[1] - 1, 1))),
        'n_components': sample(scope.int(hp.quniform('n_components', 1, 10, 1))),
        'quantiles': sample(scope.int(hp.quniform('quantiles', 4, 7, 1))),
        'params': {'booster': hp.choice('booster', ['gbtree']), #'dart']),
        'n_estimators': sample(scope.int(hp.quniform('n_estimators', 50, 300, 1))),
        'eta': hp.loguniform('eta', np.log(0.025), np.log(0.5)),
        'gamma': hp.uniform('gamma', 0.1, 1),
        'max_depth': sample(scope.int(hp.quniform('max_depth', 1, 10, 1))),
        'min_child_weight': hp.uniform('min_child_weight', 1, 10),
        'random_state': sample(scope.int(hp.quniform('random_state', 1, 7, 1))),
        'subsample': hp.uniform('subsample', 0.7, 1),
        'alpha': hp.uniform('alpha', 0, 5),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
        'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
        'normalize_type': hp.choice('normalize_type', ['tree', 'forest']),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'rate_drop': hp.uniform('rate_drop', 0, 1),
        'skip_drop': hp.uniform('skip_drop', 0, 1),
        'colsample_bylevel':  hp.uniform('colsample_bylevel', 0.7, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0.7, 1),
        'reg_lambda':  hp.uniform('reg_lambda', 0, 5)}
    }
])

# Model selection

def objective_function(args):
    n_components = args['n_components']
    quantiles = args['quantiles']
    if args['preprocessing'] == 'NoTransform':
        X, Y, scaler = transform(dataset)
    elif args['preprocessing'] == 'MinMaxScaler':
        X, Y, scaler = transform(dataset)
    elif args['preprocessing'] == 'StandardScaler':
        X, Y, scaler = standard_scaler(dataset)
    elif args['preprocessing'] == 'RobustScaler':
        X, Y, scaler = robust_scaler(dataset)
    elif args['preprocessing'] == 'QuantileTransformer':
        X, Y, scaler = quantile_transformer(dataset, quantiles)
    elif args['preprocessing'] == 'PowerTransformer':
        X, Y, scaler = power_transformer(dataset)
    elif args['preprocessing'] == 'PCA':
        X, Y, scaler = pca_transform(dataset, n_components)
    if args['preprocessing'] != 'PCA':
        k_features = args['k_features']
    else:
        k_features = X.shape[1]
    if args['model'] == RandomForestRegressor:
        n_estimators = args['params']['n_estimators']
        max_depth = args['params']['max_depth']
        min_samples_split = args['params']['min_samples_split']
        min_samples_leaf = args['params']['min_samples_leaf']
        min_weight_fraction_leaf = args['params']['min_weight_fraction_leaf']
        max_features = args['params']['max_features']
        max_leaf_nodes = args['params']['max_leaf_nodes']
        estimator = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth,
                  min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf,
                  max_leaf_nodes = max_leaf_nodes, min_weight_fraction_leaf = min_weight_fraction_leaf,
                  max_features = max_features, n_jobs = -1)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == AdaBoostRegressor:
        learning_rate = args['params']['learning_rate']
        n_estimators = args['params']['n_estimators']
        loss = args['params']['loss']
        max_depth = args['params']['base_estimator']['max_depth']
        min_samples_split = args['params']['base_estimator']['min_samples_split']
        min_samples_leaf = args['params']['base_estimator']['min_samples_leaf']
        min_weight_fraction_leaf = args['params']['base_estimator']['min_weight_fraction_leaf']
        max_features = args['params']['base_estimator']['max_features']
        estimator = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth = max_depth, min_samples_split = min_samples_split,
                  min_samples_leaf = min_samples_leaf, min_weight_fraction_leaf = min_weight_fraction_leaf,
                  max_features = max_features), learning_rate = learning_rate, n_estimators = n_estimators, loss = loss)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == ExtraTreesRegressor:
        n_estimators = args['params']['n_estimators']
        max_depth = args['params']['max_depth']
        min_samples_split = args['params']['min_samples_split']
        max_features = args['params']['max_features']
        min_samples_leaf = args['params']['min_samples_leaf']
        min_weight_fraction_leaf = args['params']['min_weight_fraction_leaf']
        max_leaf_nodes = args['params']['max_leaf_nodes']
        estimator = ExtraTreesRegressor(n_estimators = n_estimators, max_depth = max_depth,
                  min_samples_split = min_samples_split, max_features = max_features,
                  max_leaf_nodes = max_leaf_nodes, min_weight_fraction_leaf = min_weight_fraction_leaf,
                  min_samples_leaf = min_samples_leaf, n_jobs = -1)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == GradientBoostingRegressor:
        loss = args['params']['loss']
        learning_rate = args['params']['learning_rate']
        n_estimators = args['params']['n_estimators']
        subsample = args['params']['subsample']
        min_samples_split = args['params']['min_samples_split']
        max_depth = args['params']['max_depth']
        tol = args['params']['tol']
        estimator = GradientBoostingRegressor(loss = loss, n_estimators = n_estimators,
                  subsample = subsample, min_samples_split = min_samples_split, learning_rate = learning_rate,
                  max_depth = max_depth, tol = tol)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == SGDRegressor:
        loss = args['params']['loss']
        penalty = args['params']['penalty']
        alpha = args['params']['alpha']
        l1_ratio = args['params']['l1_ratio']
        tol = args['params']['tol']
        learning_rate = args['params']['learning_rate']
        power_t = args['params']['power_t']
        estimator = SGDRegressor(loss = loss, penalty = penalty, alpha = alpha, max_iter = 13000,
                  l1_ratio = l1_ratio, tol = tol, learning_rate = learning_rate, power_t = power_t)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == ElasticNet:
        alpha = args['params']['alpha']
        l1_ratio = args['params']['l1_ratio']
        tol = args['params']['tol']
        estimator = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, tol = tol)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == Ridge:
        alpha = args['params']['alpha']
        tol = args['params']['tol']
        solver = args['params']['solver']
        estimator = Ridge(alpha = alpha, tol = tol, solver = solver)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == KNeighborsRegressor:
        n_neighbors = args['params']['n_neighbors']
        weights = args['params']['weights']
        algorithm = args['params']['algorithm']
        leaf_size = args['params']['leaf_size']
        p = args['params']['p']
        estimator = KNeighborsRegressor(n_neighbors = n_neighbors, weights = weights,
                                        algorithm = algorithm, leaf_size = leaf_size, p = p, n_jobs = -1)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == GaussianProcessRegressor:
        alpha = args['params']['alpha']
        estimator = GaussianProcessRegressor(alpha = alpha)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == SVR:
        kernel = args['params']['kernel']
        if kernel == 'poly':
            degree = args['params']['degree']
        else:
            degree = 3
        if kernel == 'rbf' or 'poly' or 'sigmoid':
            gamma = args['params']['gamma']
        else:
            gamma = 'auto'
        tol = args['params']['tol']
        C = args['params']['C']
        shrinking = args['params']['shrinking']
        estimator = SVR(kernel = kernel, degree = degree, gamma = gamma, tol = tol, C = C, shrinking = shrinking)
        reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False)
    elif args['model'] == xgb:
        booster = args['params']['booster']
        eta = args['params']['eta']
        gamma = args['params']['gamma']
        max_depth = args['params']['max_depth']
        n_estimators = args['params']['n_estimators']
        min_child_weight = args['params']['min_child_weight']
        subsample = args['params']['subsample']
        alpha = args['params']['alpha']
        random_state = args['params']['random_state']
        colsample_bytree = args['params']['colsample_bytree']
        colsample_bylevel = args['params']['colsample_bylevel']
        colsample_bynode = args['params']['colsample_bynode']
        reg_lambda = args['params']['reg_lambda']
        grow_policy = args['params']['grow_policy']
        if booster == 'dart':
            sample_type = args['params']['sample_type']
            normalize_type = args['params']['normalize_type']
            rate_drop = args['params']['rate_drop']
            skip_drop = args['params']['skip_drop']
        if args['preprocessing'] != 'PCA':
            k_features = args['k_features']
        else:
            k_features = sample(scope.int(hp.quniform('k_features', 1, X.shape[1], 1)))
        if booster == 'gbtree':
            estimator = xgb.XGBRegressor(booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                              min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                              colsample_bytree = colsample_bytree, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                              colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
            reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False, scoring = metrics_names[eval_metric])
        elif booster == 'dart':
            num_round = 50
            estimator = xgb.XGBRegressor(booster = booster, eta = eta, gamma = gamma, max_depth = max_depth, n_estimators = n_estimators,
                              min_child_weight = min_child_weight, subsample = subsample, alpha = alpha, random_state = random_state,
                              colsample_bytree = colsample_bytree, sample_type = sample_type, normalize_type = normalize_type,
                              rate_drop = rate_drop, skip_drop = skip_drop, colsample_bylevel = colsample_bylevel, grow_policy = grow_policy,
                              colsample_bynode = colsample_bynode, reg_lambda = reg_lambda, n_jobs = -1)
            reg = SFS(estimator, cv = 4, k_features = k_features, forward = True, floating = False, scoring = metrics_names[eval_metric])
    if eval_metric == 'mse':
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1 - percent_train, random_state = 1, shuffle = False)
        sfsl = reg.fit(X, Y)
        x_sfs = sfsl.transform(X)
        x_train_sfs = x_sfs[:length_train]
        x_test_sfs = x_sfs[length_train:]
        estimator.fit(x_train_sfs, y_train)
        if args['model'] == xgb:
            if booster == "gbtree":
                y_pred = estimator.predict(x_test_sfs)
            elif booster == "dart":
                y_pred = estimator.predict(x_test_sfs, ntree_limit = num_round)
        else:
            y_pred = estimator.predict(x_test_sfs)
        if args['preprocessing'] != 'NoTransform':
            predictions = y_pred.reshape(-1, 1)
            for i in range(predictions.shape[1]):
                if args['preprocessing'] != 'PCA':
                    tmp = np.zeros((predictions.shape[0], n_features))
                else:
                    tmp = np.zeros((predictions.shape[0], X.shape[1]))
                tmp[:, 0] = predictions[:, i]
                predictions[:, i] = scaler.inverse_transform(tmp)[:, 0]
            mse = mean_squared_error(dataset[target][length_train:], predictions)
            print('mse value: {}, model: {}'.format(mse, args['model']))
            return mse
        else:
            mse = mean_squared_error(dataset[target][length_train:], y_pred)
            print('mse value: {}, model: {}'.format(mse, args['model']))
            return mse
    else:
        reg.fit(X, Y)
        print('Model: {}, r2 value: {}, Selected variables {}'.format(args['model'], reg.k_score_, reg.k_feature_names_))
        loss_function = 1 - reg.k_score_
        return loss_function
        #loss_function = np.average(cross_val_score(estimator, X, Y, cv = 3))
        #print('Model: {}, r2 value: {}'.format(args['model'], loss_function))
        #return 1 - loss_function

def select_model(space):
    best_regressor = fmin(objective_function, space, algo = tpe.suggest, max_evals = max_evals)
    print(hyperopt.space_eval(space, best_regressor))

select_model(search_space)
