import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def strategy_return(y_pred,y_return,groups,cost_ratio=0.003,long_ratio=0.1,short_ratio=0.1):
#     u,ind = np.unique(groups, return_index=True)
#     ind = np.append(ind,len(groups))
#     daily_return = []

#     for i in range(len(u)):
#         tmp_ind = list(range(ind[i],ind[i+1]))
#         y_pred_rank = y_pred[tmp_ind].argsort().argsort() / len(y_pred[tmp_ind])
#         long_ind, short_ind = y_pred_rank > (1-long_ratio), y_pred_rank < short_ratio
#         nums = long_ind.sum() + short_ind.sum()
#         tmp_return = (np.dot(y_return[tmp_ind],long_ind)-np.dot(y_return[tmp_ind],short_ind)) / nums - cost_ratio
#         daily_return.append(tmp_return)

#     u = u[1:]
#     daily_return = daily_return[:-1]
#     cum_return = np.cumsum(daily_return)
#     riskfree_rate = 0.0001

#     return_mean = np.mean(daily_return)
#     return_std  = np.std(daily_return)
#     return_sharpe = (return_mean-riskfree_rate)/return_std*np.sqrt(252)
#     return_maxdown = (cum_return - np.maximum.accumulate(cum_return)).min()

#     fig, ax = plt.subplots(figsize=(10,6))
#     ax.plot(pd.to_datetime(u), 1+cum_return)
#     ax.set_title('PNL from {} ~ {} with cost_ratio {}'.format(u[0],u[-1],cost_ratio))
#     ax.text(0.02,0.85,'daily_mean:   {:.2%}\ndaily_std:       {:.2%}\nsharpe:          {:.2f}\nmaxdown:     {:.2%}'.format(return_mean,return_std,return_sharpe,return_maxdown),
#             bbox=dict(facecolor='yellow', alpha=0.5),transform=ax.transAxes)
            
#     return u,daily_return

def strategy_return(Y,y_pred,groups,hold_days,ahead_steps,close_open,long_ratio=0.1,short_ratio=0.1,cost_ratio=0.003):
  df = pd.DataFrame({'code':groups[:,0],'datetime':groups[:,1],'y_return_oc':Y[:,0],'y_return_cc':Y[:,1],'y_pred':y_pred,'buy_status':0,'hold_status':0})
  df['y_pred'] = df.groupby(['code'])['y_pred'].shift(1)
  df['y_pred_rank'] = df.groupby(['datetime'])['y_pred'].rank(pct=True)

  def buy_status(x):
      x['buy_status'] = x['buy_status'] + (x['y_pred_rank'] > (1-long_ratio)).astype('int')
      x['buy_status'] = x['buy_status'] - (x['y_pred_rank'] < short_ratio).astype('int')
      return x
  def hold_status(x):
      x['hold_status'] = 0
      i = 1
      while i < hold_days:
          x['hold_status'] = x['hold_status'] + x['buy_status'].shift(i)
          x = x.fillna(0)
          i += 1
      return x

  df['buy_status'] = df.groupby(['datetime']).apply(lambda x: buy_status(x))['buy_status']
  df['hold_status'] = df.groupby(['code']).apply(lambda x: hold_status(x))['hold_status']

  if close_open:
    df['daily_return'] = df['buy_status']*df['y_return_oc'] + df['hold_status']*df['y_return_cc']
  else:
    df['daily_return'] = df['buy_status']*df['y_return_cc'] + df['hold_status']*df['y_return_cc']

  df_return = df.groupby(['datetime']).apply(lambda x: x['daily_return'].sum()/(abs(x['buy_status']).sum()+abs(x['hold_status']).sum()))
  df_return = df_return - cost_ratio/hold_days

  u,daily_return = np.array(df_return.index), list(df_return.fillna(0))

  cum_return = np.cumsum(daily_return)
  riskfree_rate = 0.0001

  return_mean = np.mean(daily_return)
  return_std  = np.std(daily_return)
  return_sharpe = (return_mean-riskfree_rate)/return_std*np.sqrt(252)
  return_maxdown = (cum_return - np.maximum.accumulate(cum_return)).min()

  fig, ax = plt.subplots(figsize=(10,6))
  ax.plot(pd.to_datetime(u),1+cum_return)
  ax.set_title('PNL from {} ~ {}'.format(u[0],u[-1]))
  ax.text(0.02,0.85,'daily_mean:    {:.2%}\ndaily_std:        {:.2%}\nsharpe:           {:.2f}\nmaxdown:      {:.2%}'.format(return_mean,return_std,return_sharpe,return_maxdown),bbox=dict(facecolor='yellow', alpha=0.5),transform=ax.transAxes)
  ax.text(0.02,0.78,'{} days ahead predict'.format(ahead_steps),bbox=dict(facecolor='red', alpha=0.5),transform=ax.transAxes)
  ax.text(0.02,0.71,'Holding for {} days     '.format(hold_days),bbox=dict(facecolor='green', alpha=0.5),transform=ax.transAxes)
  ax.text(0.02,0.64,'Close_open: {}      '.format(close_open),bbox=dict(facecolor='orange', alpha=0.5),transform=ax.transAxes)
  ax.text(0.02,0.57,'Paras:{}'.format((long_ratio,short_ratio,cost_ratio)),bbox=dict(facecolor='grey', alpha=0.5),transform=ax.transAxes)


  return u, daily_return


def model_strategy_return():
    pass


def backtest(y_pred,groups,return_oc,return_co,return_type,long_ratio=0.1,short_ratio=0.1,cost_ratio=0.0035):
    def _calculate(y_pred,groups,return_oc,return_co,return_type,long_ratio,short_ratio,cost_ratio):
      df = pd.DataFrame({'code':groups[:,0],'datetime':groups[:,1],'return_oc':return_oc,'return_co':return_co,'y_pred':y_pred,'oc_status':0,'co_status':0})
      df['y_pred'] = df.groupby(['code'])['y_pred'].shift(1)
      df['y_pred'] = df.groupby(['datetime'])['y_pred'].rank(pct=True)

      def oc_status(x):
        x['oc_status'] = x['oc_status'] + (x['y_pred'] > (1-long_ratio)).astype('int')
        x['oc_status'] = x['oc_status'] - (x['y_pred'] < short_ratio).astype('int')
        return x
      
      def co_status(x):
        x['co_status'] = x['co_status'] + (x['y_pred'] > (1-long_ratio)).astype('int')
        x['co_status'] = x['co_status'] - (x['y_pred'] < short_ratio).astype('int')
        return x
        
      df['oc_status'] = df.groupby(['datetime']).apply(lambda x: oc_status(x))['oc_status']
      df['co_status'] = df.groupby(['datetime']).apply(lambda x: co_status(x))['co_status']

      if return_type == 'return_co':
        df['return'] = df['co_status']*df['return_co']
        df_return = df.groupby(['datetime']).apply(lambda x: x['return'].sum()/abs(x['co_status']).sum())
      elif return_type == 'return_cc':
        df['return'] = df['co_status']*df['return_co'] + df['oc_status']*df['return_oc']
        df_return = df.groupby(['datetime']).apply(lambda x: x['return'].sum()/abs(x['co_status']).sum())
      elif return_type == 'return_oc':
        df['return'] = df['oc_status']*df['return_oc']
        df_return = df.groupby(['datetime']).apply(lambda x: x['return'].sum()/abs(x['oc_status']).sum())
      elif return_type == 'return_oo':
        df['co_status'] = df.groupby(['code'])['co_status'].shift(1)
        df['co_status'] = df['co_status'].fillna(0)
        df['return1'] = df['co_status']*df['return_co']
        df['return2'] = df['oc_status']*df['return_oc']
        df_return = df.groupby(['datetime']).apply(lambda x: x['return1'].sum()/abs(x['co_status']).sum()+x['return2'].sum()/abs(x['oc_status']).sum())
      else:
        raise 'return_type is set wrong'

      df_return = df_return - cost_ratio
      u,daily_return = np.array(df_return.index), np.array(df_return.fillna(0))
      return u, daily_return
     
    u,daily_return = _calculate(y_pred,groups,return_oc,return_co,return_type,long_ratio,short_ratio,cost_ratio)
    u,daily_return_long = _calculate(y_pred,groups,return_oc,return_co,return_type,long_ratio,0,cost_ratio)
    u,daily_return_short = _calculate(y_pred,groups,return_oc,return_co,return_type,0,short_ratio,cost_ratio)

    daily_return_long = np.array(daily_return_long)*(long_ratio/(long_ratio+short_ratio))
    daily_return_short = np.array(daily_return_short)*(short_ratio/(long_ratio+short_ratio))

    cum_return = np.cumsum(daily_return)
    cum_return_long = np.cumsum(daily_return_long)
    cum_return_short = np.cumsum(daily_return_short)
    riskfree_rate = 0.0001

    return_mean = np.mean(daily_return)
    return_std  = np.std(daily_return)
    return_sharpe = (return_mean-riskfree_rate)/return_std*np.sqrt(252)
    return_maxdown = (cum_return - np.maximum.accumulate(cum_return)).min()

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(pd.to_datetime(u),cum_return,color='black')
    ax.plot(pd.to_datetime(u),cum_return_long,color='red')
    ax.plot(pd.to_datetime(u),cum_return_short,color='green')
    ax.legend(['long_short','long','short'],loc='upper right',)
    ax.set_title('PNL from {} ~ {}'.format(u[0],u[-1]))
    ax.text(0.02,0.85,'daily_mean:    {:.2%}\ndaily_std:        {:.2%}\nsharpe:           {:.2f}\nmaxdown:      {:.2%}'.format(return_mean,return_std,return_sharpe,return_maxdown),bbox=dict(facecolor='yellow', alpha=0.5),transform=ax.transAxes)
    # ax.text(0.02,0.78,'{} days ahead predict'.format(ahead_steps),bbox=dict(facecolor='red', alpha=0.5),transform=ax.transAxes)
    # ax.text(0.02,0.71,'Holding for {} days     '.format(hold_days),bbox=dict(facecolor='green', alpha=0.5),transform=ax.transAxes)
    ax.text(0.02,0.78,'return_type: {}'.format(return_type),bbox=dict(facecolor='orange', alpha=0.5),transform=ax.transAxes)
    ax.text(0.02,0.71,'Paras:{}'.format((long_ratio,short_ratio,cost_ratio)),bbox=dict(facecolor='grey', alpha=0.5),transform=ax.transAxes)

    return u, daily_return
      
