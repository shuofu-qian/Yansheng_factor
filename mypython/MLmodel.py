import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from cuml.model_selection import GridSearchCV as GridSearchCV_gpu

import pickle

class MLmodel:

    def __init__(self, x:np.array, y:np.array, model, name:str='model'):
        self.x,self.y = x,y
        self.model = model

        self.x_train,self.y_train = None,None
        self.x_test, self.y_test  = None,None
        self.y_pred = None

        self.name = name


    def split_train_test(self, test_size:int or float=0.3, random:bool=True, random_state:int=None):
        assert len(self.x) == len(self.y), 'Error:The lenth of x and y is not matched'
        assert test_size >= 0 and test_size <= len(self.x), 'Error: The value of test_size is not sensible'

        if random:
            self.x_train, self.x_test, self.y_train, self.y_test =train_test_split(self.x,self.y,test_size=test_size,random_state=random_state)
        else:
            indx = type(test_size)==float and math.floor(len(self.x)*(1-test_size)) or int(len(self.x)-test_size)
            self.x_train,self.y_train = self.x[:indx],self.y[:indx]
            self.x_test, self.y_test  = self.x[indx:],self.y[indx:]

        return self


    def fit_model(self):
        self.model.fit(self.x_train,self.y_train)

        return self


    def predict(self,scaling:bool=False):
        self.y_pred = self.model.predict(self.x_test)
        
        if scaling:                                                                    
            self.y_pred = self.y_pred.argsort().argsort()/len(self.y_pred)
        
        return self


    def evaluate_model(self,print_:bool=True) -> tuple:
        mse  = metrics.mean_squared_error(self.y_test,self.y_pred)
        rmse = math.sqrt(mse)
        mae  = metrics.mean_absolute_error(self.y_test,self.y_pred)
        # r2   = metrics.r2_score(self.y_test,self.y_pred)

        if print_:
            print('{:<10}:  MSE:{:.2f}, RMSE:{:.2f}, MAE:{:.2f}'.format(self.name,mse,rmse,mae))

        return mse,rmse,mae


    def get_param_dict(self) -> dict:
        return self.model.get_params()


    def get_best_param_dict(self, param_grid:dict or list(dict), scoring=None, cv=None, gpu=True):
        """use GridSearchCV to search the best param from param_grid, return a GridSearchCV object,
        and then you can use .best_params_ to get the dict of the best param"""

        if gpu:
            GridSearchCV_model = GridSearchCV_gpu(self.model, param_grid, scoring=scoring, cv=cv)
        else:
            GridSearchCV_model = GridSearchCV(self.model, param_grid, scoring=scoring, cv=cv)
            
        GridSearchCV_model.fit(self.x_train,self.y_train)
        return GridSearchCV_model
    
    def set_param(self,param_dict:dict):
        self.model.__dict__.update(param_dict)
        return self

# def save_model(model,path):
#     pickle.dumps(model)

def load_data(x_path='/home/qianshuofu/factor_qianshuofu/Data/data_feature.npy',
              y_path='/home/qianshuofu/factor_qianshuofu/Data/data_label.npy',
              reshape:bool = False, sample = False) -> tuple:
    x,y = np.load(x_path),np.load(y_path)

    if sample: x,y = x[:3000,:50], y[:3000]
    if reshape: y = y[:,np.newaxis]

    x = x.astype(np.float32)
    y = y.astype(np.float32)
        
    return x,y

    
def tran(strr):
    result = []
    list = strr.split(' ')
    for item in list:
        list2 = item.split('=')
        result.append('{0}={0},'.format(list2[0]))
    
    for i in result:
        print(i,end=' ')


def get_boost_para():
    params = {'max_bin': 63,
            'num_leaves': 255,
            'min_data_in_leaf': 106, # 每个叶子节点中的数据
            'max_depth': -1, # -1 ： 不限制深度
            "boosting_type": "gbdt", # 'dart', 'goss', 'rf'
            "metric": 'auc', # 衡量标准
            'random_state': 66, # 随机种子    
            'learning_rate': 0.1,
            'tree_learner': 'serial',
            'task': 'train',
            'is_training_metric': 'false',
            'min_data_in_leaf': 1,
            'min_sum_hessian_in_leaf': 100,
            'ndcg_eval_at': [1, 3, 5, 10]
            }
    return params
    

# 降维PCA,ICA
def PCA(x):
    pass
