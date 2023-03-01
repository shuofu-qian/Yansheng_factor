import numpy as np
import sklearn
import lightgbm
import xgboost
import cuml

class MyDTRegressor(sklearn.tree._classes.DecisionTreeRegressor):
    def __init__(self, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0):
        super().__init__(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha)

class MyRFRegressor(sklearn.ensemble.RandomForestRegressor):
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)

class MyRFRegressor_Gpu(cuml.ensemble.randomforestregressor.RandomForestRegressor):
    def __init__(self, *, split_criterion=2, accuracy_metric='r2', handle=None, verbose=False, output_type=None, **kwargs):
        super().__init__(split_criterion=split_criterion, accuracy_metric=accuracy_metric, handle=handle, verbose=verbose, output_type=output_type, **kwargs)

class MyLGBMRegressor(lightgbm.sklearn.LGBMRegressor):
    def __init__(self, boosting_type: str = 'gbdt', num_leaves: int = 31, max_depth: int = -1, learning_rate: float = 0.1, n_estimators: int = 100, subsample_for_bin: int = 200000, objective=None, class_weight = None, min_split_gain: float = 0.0, min_child_weight: float = 0.001, min_child_samples: int = 20, subsample: float = 1.0, subsample_freq: int = 0, colsample_bytree: float = 1.0, reg_alpha: float = 0.0, reg_lambda: float = 0.0, random_state = None, n_jobs: int = -1, silent = 'warn', importance_type: str = 'split', **kwargs):
        super().__init__(boosting_type=boosting_type, num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, subsample_for_bin=subsample_for_bin, objective=objective, class_weight=class_weight, min_split_gain=min_split_gain, min_child_weight=min_child_weight, min_child_samples=min_child_samples, subsample=subsample, subsample_freq=subsample_freq, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=random_state, n_jobs=n_jobs, silent=silent, importance_type=importance_type, **kwargs)

class MyXGBRegressor(xgboost.sklearn.XGBRegressor):
    def __init__(self, *, objective = 'reg:squarederror', **kwargs):
        super().__init__(objective = objective, **kwargs)

class MyDTRegressor2(sklearn.tree._classes.DecisionTreeRegressor):
    def __init__(self, *args):
        super().__init__(*args)

# class MyLGBMRegressor:
#     def __init__(self, **kwargs):
#         # self.params = params
#         pass

#     def fit(self,x,y):
#         self.__model__ = lightgbm.train(self.__dict__,lightgbm.Dataset(x,y))
#         return self

#     def predict(self,x):
#         y = self.__model__.predict(x)
#         return y

#     def get_params(self):
#         return self.__dict__


class Feature_lightgbm():
    pass
