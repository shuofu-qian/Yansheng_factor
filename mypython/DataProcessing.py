import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import math
from scipy.stats.mstats import winsorize
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


class FeatureDataFrame:
    def __init__(self,data:pd.DataFrame):
        self.data = data

    def filter_asset(self,asset_pool:pd.DataFrame):
        """Filter the asset in self using the data in asset_pool
        
        asset_pool:     A dataframe whose first column is asset code
        """

        asset_pool = asset_pool.iloc[:,0:1].rename(columns = {asset_pool.columns[0]:"asset"})
        asset_pool['asset'] = asset_pool['asset'].apply(lambda x: str(x))
        self.data = pd.merge(self.data,asset_pool,how='inner',on = 'asset').sort_values(['asset','datetime'])

        return self


    def drop_samevalue(self):
        """Drop the column from the third one if all the data in this column is the same or all of them is NaN"""

        drop_list = [column for column in self.data.columns[2:] if not np.nanstd(self.data[column]) > 0]
        self.data = self.data.drop(columns = drop_list)

        return self


    def __unstack(self):
        try:
            self.data['minute'] = self.data['datetime'].apply(lambda x: x[-8:])
            self.data['datetime'] = self.data['datetime'].apply(lambda x: x[:10])
        except:
            raise Exception("The datatime column is not suitable to be unstacked or the data has reached the required shape")

        self.data = self.data.sort_values(['asset','datetime','minute']).set_index(['asset','datetime','minute']).unstack(2)
        self.data.columns = ["_".join(tuple) for tuple in self.data.columns]
        self.data.reset_index(inplace=True)

        return self

    def __stack(self):
        try:
            self.data.set_index(["asset","datetime"],inplace=True)
            column_list = ["asset","datetime"] + list(dict.fromkeys([column[:-9] for column in self.data.columns]))
            self.data.columns = pd.MultiIndex.from_tuples([(column[:-9],column[-8:]) for column in self.data.columns],names = ("","minute"))
            self.data = self.data.stack(1).reset_index()
        except:
            raise Exception("The column nams are not suitable to be stacked or the data has reached the required shape")

        self.data["datetime"] = self.data["datetime"].str.cat(self.data["minute"],sep = " ")
        self.data = self.data.drop(columns=['minute'])[column_list]

        return self

    def change_shape(self,unstack:bool=True):
        """Suppose there are n stocks, s time intervals in one day, m features,
        change the data shape from (n*s,m) to (n,m*s) if unstack == True, otherwise the opposite"""
        
        if unstack:
            self.__unstack()
        else:
            self.__stack()
            
        return self

    def fill_na(self):
        columns = self.data.columns[2:]
        tmp = self.data.groupby('datetime')[columns].transform('mean')
        self.data.iloc[:,2:] = self.data.iloc[:,2:].fillna(tmp)
        self.data = self.data.fillna(0)
        return self

    def get_zscore(self):
        columns = self.data.columns[2:]
        def scale(x):
            x.iloc[:,:] = preprocessing.scale(x)
            return x
        self.data.iloc[:,2:] = self.data.groupby('datetime')[columns].apply(lambda x: scale(x))
        return self




class LabelDataFrame:
    def __init__(self,data:pd.DataFrame):
        self.data = data
    
    def get_return(self, adj = False, return_frequency = "1d", return_type = 'return_oo'):
        """Sort the data by code and time, append a new column naming 'return' in the last with NaN being filled with 0
        
        adj:                    whether to calculate using adj price, default == False
        return_frequency:       the lenth of time when calculating return, can be chosen from '1d','2d','4m' etc. default =='1d'
        close_open:             whether to calculate return using today's close price and open price
        """

        adj_str = adj and "adj_" or ""                          
        self.data.sort_values(["symbol_id","trade_date"],inplace=True)
        fre_num = int(return_frequency[:-1])

        self.data['today_return_oc'] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(0)/\
                                               self.data.groupby("symbol_id")[adj_str+"open_price"].shift(0) )
        self.data['today_return_co'] = np.log( self.data.groupby("symbol_id")[adj_str+"open_price"].shift(0)/\
                                               self.data.groupby("symbol_id")[adj_str+"close_price"].shift(1) )
        # if close_open:
        #     self.data["return"] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
        #                                   self.data.groupby("symbol_id")[adj_str+"open_price"].shift(-1) )
        # else:
        #     self.data['return'] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
        #                                   self.data.groupby("symbol_id")[adj_str+"close_price"].shift(0) )
            
        if return_type == 'return_oc':
            self.data['return'] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
                                          self.data.groupby("symbol_id")[adj_str+"open_price"].shift(-1) )
        elif return_type == 'return_cc':
            self.data['return'] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
                                          self.data.groupby("symbol_id")[adj_str+"close_price"].shift(0) )        
        elif return_type == 'return_oo':
            self.data['return'] = np.log( self.data.groupby("symbol_id")[adj_str+"open_price"].shift(-fre_num-1)/\
                                          self.data.groupby("symbol_id")[adj_str+"open_price"].shift(-1) )
        elif return_type == 'return_co':
            self.data['return'] = np.log( self.data.groupby("symbol_id")[adj_str+"open_price"].shift(-fre_num)/\
                                          self.data.groupby("symbol_id")[adj_str+"close_price"].shift(0) )

            # if return_frequency[-1] == "d":
            #     self.data["return"] = np.log( self.data.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
            #                                   self.data.groupby("symbol_id")[adj_str+"close_price"].shift(0) )
            # elif return_frequency[-1] == "m":
            #     self.data["year_month"] = self.data["trade_date"].apply(lambda x: x[0:6])
            #     temp_df = self.data[["symbol_id",adj_str+"close_price","year_month"]].groupby(["symbol_id","year_month"]).last().reset_index()
            #     temp_df["return"] = np.log( temp_df.groupby("symbol_id")[adj_str+"close_price"].shift(-fre_num)/\
            #                                 temp_df.groupby("symbol_id")[adj_str+"close_price"].shift(0) )
            #     self.data = pd.merge(self.data,temp_df[["symbol_id","year_month","return"]], how = "left", on = ["symbol_id","year_month"])          
            #     self.data = self.data.drop(columns = ["year_month"]) 
            # else:
            #     raise Exception("The value of return_frequency is set wrong")         
                     
        self.data["today_return_oc"] = self.data["today_return_oc"].fillna(0)
        self.data['today_return_co'] = self.data["today_return_co"].fillna(0)
        self.data["return"] = self.data["return"].fillna(0)
        # self.data["rank_return"] = self.data["return"]
        return self

    def rank_norm(self):
        self.data['return'] = (self.data.groupby(['trade_date'])['return'].rank(pct=True) - 0.5)*3.4624
        return self



class ProcessArray:
    def __init__(self,data:pd.DataFrame or np.ndarray):
        self.array = np.array(data)

    def __call__(self):
        return self.array

    # Actually, when there is no extreme value, e.g.[0,1,1,1,1,1,1], using median to transform will get [1,1,1,1,1,1,1]
    # But it is not advisable for the std after transform is zero not one!

    # Solution one: transform the extreme value according to their percentile
    # Solution two: use (0.5,99.5) data to calculate mean and std

    def fill_na(self,strategy='mean',fill_value=None, keep_empty_features=True):
        """strategy:'mean','median','most_frequency','constant'"""

        transformer = SimpleImputer(strategy=strategy, fill_value=fill_value, keep_empty_features=keep_empty_features)
        self.array = transformer.fit(self.array).transform(self.array)

        return self


    def get_zscore(self,limit_extreme_value=True,limits=[0.005,0.005],robust=False,feature_range=(25,75),std_times=3):
        if limit_extreme_value:
            self.array = winsorize(self.array,limits=limits,axis=0).data

            # median = np.nanmedian(self.array,axis=0)
            # median_std = np.nanmedian(np.abs(self.array-median),axis=0)
            # self.array = np.clip(self.array,median-std_times*median_std,median+std_times*median_std)

            # mean = np.nanmean(self.array,axis=0)
            # mean_std = np.nanstd(self.array,axis=0)
            # self.array = np.clip(self.array,mean-std_times*mean_std,mean+std_times*mean_std)

        if robust:
            self.array = preprocessing.robust_scale(self.array,feature_range=feature_range)
        else:
            self.array = preprocessing.scale(self.array)

        return self

    def get_scale(self):
        self.array = (self.array - self.array.min()) / (self.array.max() - self.array.min())
        return self


def merge_feature_label(df_feature,df_label) -> pd.DataFrame:
    df_feature['trade_date'] = df_feature['datetime'].apply(lambda x: x[:10].replace("-",""))
    df_feature['symbol_id'] = df_feature['asset']

    df_merge = pd.merge(df_feature,df_label[["trade_date","symbol_id","today_return_oc","today_return_co","return"]],how = "inner",on = ["trade_date","symbol_id"])
    df_merge = df_merge.drop(columns = ["trade_date","symbol_id"]).reset_index(drop = True)
    df_merge = df_merge.sort_values(by=['datetime','asset'])

    return df_merge

def split_index_feature_label(df_merge) -> tuple:
    """Split the merged data into three parts: pa_index,pa_feature,pa_label. whose indexs are matched"""
    
    data_index = ProcessArray(df_merge.iloc[:,:2])
    data_feature = ProcessArray(df_merge.iloc[:,2:-3])
    data_label = ProcessArray(df_merge.iloc[:,-3:])
    data_column = ProcessArray(df_merge.columns[2:-3])

    return data_index, data_feature, data_label, data_column



def return_vwap(df_vwap,df_index):
    # df_vwap = pd.read_feather('/home/qianshuofu/factor_qianshuofu/Data/30min_kbars_adjuested_1335.ftr')
    # df_index = pd.DataFrame(np.load('/home/qianshuofu/factor_qianshuofu/Data/data_index.npy',allow_pickle=True),columns=['code','datetime'])

    df_vwap = df_vwap.rename(columns={'asset':'code'})
    df_vwap['minute'] = df_vwap['datetime'].apply(lambda x: str(x)[11:])
    df_vwap['datetime'] = df_vwap['datetime'].apply(lambda x: str(x)[:10])
    df_vwap = df_vwap.dropna(how='all',subset=['Open','High','Low','Close','VWAP'])
    df_vwap = df_vwap.sort_values(['code','datetime','minute'])

    df_return_oc = pd.DataFrame(np.log(df_vwap.groupby(['code','datetime'])['VWAP'].last() / df_vwap.groupby(['code','datetime'])['VWAP'].first())).reset_index()
    df_return_oc.columns = ['code','datetime','return_oc']
    df_return_co = df_vwap.groupby(['code','datetime'])['VWAP'].aggregate(['first','last']).reset_index()
    df_return_co['last'] = df_return_co.groupby(['code'])['last'].shift(1)
    df_return_co['return_co'] = np.log(df_return_co['first']/df_return_co['last'])

    df_return_oc['return_oc'] = df_return_oc['return_oc'].fillna(0)
    df_return_co['return_co'] = df_return_co['return_co'].fillna(0)

    df_return_oc = pd.merge(df_index,df_return_oc,how='left',on=['code','datetime'])
    df_return_co = pd.merge(df_index,df_return_co,how='left',on=['code','datetime'])

    np.save('/home/qianshuofu/factor_qianshuofu/Data/data_return_oc_vwap.npy',df_return_oc['return_oc'])
    np.save('/home/qianshuofu/factor_qianshuofu/Data/data_return_co_vwap.npy',df_return_co['return_co'])



def main(return_type='return_oo', return_frequency='1d'):
    """
    return_type: ['return_co','return_cc','return_oo','return_oc']
    return_frequency: '1d'
    """

    close_open_string = return_type[-2:]
    fre_num = int(return_frequency[:-1])

    path_dict = {"asset_pool_path":"/sda/intern_data_shuofu/industry_1070.csv",
                "feature_path": "/home/qianshuofu/factor_qianshuofu/Data/30_minutes_data.feather",
                "label_path": "/home/qianshuofu/factor_qianshuofu/Data/adj_prices.feather",

                "index_save_path": "/home/qianshuofu/factor_qianshuofu/Data/data_index.npy",
                "feature_save_path": "/home/qianshuofu/factor_qianshuofu/Data/data_feature.npy",
                "label_save_path": "/home/qianshuofu/factor_qianshuofu/Data/data_label_{}_{}.npy".format(close_open_string,fre_num),
                "column_save_path":"/home/qianshuofu/factor_qianshuofu/Data/data_column.npy"}

    df_asset_pool = pd.read_csv(path_dict['asset_pool_path'])
    df_feature = FeatureDataFrame(pd.read_feather(path_dict["feature_path"])).filter_asset(df_asset_pool).change_shape(unstack=True).fill_na().get_zscore() #.drop_samevalue()
    df_label = LabelDataFrame(pd.read_feather(path_dict['label_path'])).get_return(return_frequency=return_frequency,return_type=return_type).rank_norm()
    df_merge = merge_feature_label(df_feature.data,df_label.data)

    data_index,data_feature,data_label,data_column = split_index_feature_label(df_merge)
    # data_feature.fill_na().get_zscore(limit_extreme_value=False)
    # data_label.get_zscore(limit_extreme_value=False)

    np.save(path_dict['index_save_path'],data_index.array)
    np.save(path_dict['feature_save_path'],data_feature.array)
    np.save(path_dict['label_save_path'],data_label.array[:,-1])
    np.save(path_dict['column_save_path'],data_column.array)

    np.save("/home/qianshuofu/factor_qianshuofu/Data/data_return_oc.npy",data_label.array[:,0])
    np.save("/home/qianshuofu/factor_qianshuofu/Data/data_return_co.npy",data_label.array[:,1])

if __name__ == '__main__':
    main()