import numpy as np
import math
import sklearn
import cuml


class MyLinear(sklearn.linear_model._base.LinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)

class MyRidge(sklearn.linear_model._ridge.Ridge):
    def __init__(self, alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver, positive=positive, random_state=random_state)

class MyLasso(sklearn.linear_model._coordinate_descent.Lasso):
    def __init__(self, alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, precompute=precompute, copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)

class MyElasticNet(sklearn.linear_model._coordinate_descent.ElasticNet):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, precompute=precompute, max_iter=max_iter, copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)

class MyLogic(sklearn.linear_model._logistic.LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

class MyLinear_Gpu(cuml.linear_model.linear_regression.LinearRegression):
    def __init__(self, *, algorithm='eig', fit_intercept=True, normalize=False, handle=None, verbose=False, output_type=None):
        super().__init__(algorithm=algorithm, fit_intercept=fit_intercept, normalize=normalize, handle=handle, verbose=verbose, output_type=output_type)

class MyRidge_Gpu(cuml.linear_model.ridge.Ridge):
    def __init__(self, *, alpha=1.0, solver='eig', fit_intercept=True, normalize=False, handle=None, output_type=None, verbose=False):
        super().__init__(alpha=alpha, solver=solver, fit_intercept=fit_intercept, normalize=normalize, handle=handle, output_type=output_type, verbose=verbose)

class MyLasso_Gpu(cuml.linear_model.lasso.Lasso):
    def __init__(self, *, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=0.001, solver='cd', selection='cyclic', handle=None, output_type=None, verbose=False):
        super().__init__(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=tol, solver=solver, selection=selection, handle=handle, output_type=output_type, verbose=verbose)

class MyElasticNet_Gpu(cuml.linear_model.elastic_net.ElasticNet):
    def __init__(self, *, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, max_iter=1000, tol=0.001, solver='cd', selection='cyclic', handle=None, output_type=None, verbose=False):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, tol=tol, solver=solver, selection=selection, handle=handle, output_type=output_type, verbose=verbose)

class MyLogic_Gpu(cuml.linear_model.logistic_regression.LogisticRegression):
    def __init__(self, *, penalty='l2', tol=0.0001, C=1.0, fit_intercept=True, class_weight=None, max_iter=1000, linesearch_max_iter=50, verbose=False, l1_ratio=None, solver='qn', handle=None, output_type=None):
        super().__init__(penalty=penalty, tol=tol, C=C, fit_intercept=fit_intercept, class_weight=class_weight, max_iter=max_iter, linesearch_max_iter=linesearch_max_iter, verbose=verbose, l1_ratio=l1_ratio, solver=solver, handle=handle, output_type=output_type)