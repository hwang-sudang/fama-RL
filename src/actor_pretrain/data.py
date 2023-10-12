
import os
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

# # import pyportfolio to make labels
# from pypfopt import risk_models, expected_returns, EfficientSemivariance
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt.discrete_allocation import DiscreteAllocation # 그리디 방법으로 사후처리해야함.
# from pypfopt.risk_models import CovarianceShrinkage


from util.configures import *
warnings.filterwarnings('ignore')


'''
# 모듈의 목적
- pretrain을 위한 라벨링, 2D tensor를 3D텐서로 바꿔줌.

# 모듈의 순서
1. load data
2. data preprocessing --> class 
3. main()

'''




class Timeseries3D():
    '''
    Module화
    DataFrame 경로들을 넣으면, 3d tensor로 바꿔줌
    '''
    def __init__(self) -> None:
        pass

    @staticmethod
    def make_2Ddata(data_dir:str, 
                    ticker_file:str, 
                    initial_date: str,
                    final_date: str,
                    to_numpy=False):
        ''' 
        Args
        data_dir : 자산 데이터가 있는 위치
        stocks_subset : 포트폴리오 자산명이 들어있는 리스트

        Out
        2d numpy size : (time, asset*features)

        '''
        initial_date = datetime.strptime(initial_date, '%Y-%m-%d')
        final_date = datetime.strptime(final_date, '%Y-%m-%d')

        # ticker 정보가 있는 txt파일을 열어서, 편입될 종목의 이름 수집
        with open(ticker_file) as f:
            stocks_subset = f.read().splitlines()
            stocks_subset = [ticker for ticker in stocks_subset]

        data_container = {}
        y_container = {}

        for ticker in stocks_subset:
            data = pd.read_csv(data_dir+f'/{ticker}.csv', parse_dates=['date'], index_col=0)
            if 'market cap' in data.columns:
                data.drop('market cap', axis=1, inplace=True)
            
            data.columns = [f'open_{ticker}', f'high_{ticker}', 
                            f'low_{ticker}', f'close_{ticker}', 
                            f'volume_{ticker}', f'BTM_{ticker}',f'market_cap_{ticker}']

            # 데이터 티커 정보 추가하고 데이터 컨테이너에 임시저장
            # data['ticker'] = ticker
            data_container[ticker] = data
            y_container[ticker] = data.iloc[:,3] # close_{ticker}

        # Make dict data to the dataframe
        final_data = pd.DataFrame()
        y_data = pd.DataFrame(y_container)

        for tic in data_container:
            # 옆으로 붙여야 의미가 맞지 않을까?
            # key = next(iter(tic.keys()))
            key = tic
            final_data = pd.concat([final_data, data_container[tic]], axis=1, names=key)
        
        final_data = final_data.loc[initial_date : final_date]
        
        # if numpy array option True
        if to_numpy:
            final_data = final_data[data.columns].to_numpy()
            y_data = y_data.to_numpy()

        return final_data, y_data



    def make_2D_pretrain_data(pretrain_json:str,
                            ticker_file:str,
                            initial_date: str,
                            final_date: str,
                            save_option = False,
                            country = 'USA',
                            network = 'actor',
                            window=40,
                            init_portfolio = 10000,
                            to_numpy = False):
        ''' 
        Args
        data_dir : 자산 데이터가 있는 위치
        stocks_subset : 포트폴리오 자산명이 들어있는 리스트
        network : 'actor' or 'critic'

        Out
        2d numpy size : (time, asset*features)

        '''
        initial_date = datetime.strptime(initial_date, '%Y-%m-%d')
        final_date = datetime.strptime(final_date, '%Y-%m-%d')
        # ticker 정보가 있는 txt파일을 열어서, 편입될 종목의 이름 수집
        with open(ticker_file) as f:
            stocks_subset = f.read().splitlines()
            stocks_subset = [ticker for ticker in stocks_subset]

        with open(pretrain_json, 'r') as j:
            pretrain_json = json.load(j)

        if network == 'actor':
        # define data container
            data_container = {}
            y_container = {}

            for ticker in stocks_subset:
                for key, value in pretrain_json["pretrain"].items() :
                    if ticker in value:
                        data = pd.read_csv(f'{DATA_DIR}pretrain/preprocess/{country}/{key}.csv',
                                            parse_dates=['Date'], index_col=0)
                        data.columns = [f'close_{ticker}', f'open_{ticker}', f'high_{ticker}', f'low_{ticker}', 
                                        f'volume_{ticker}', f'chg%_{ticker}']
                        
                        # calculate total tradiing volume : instead of marketcap
                        # some reasons should be explained why trading volume is used instead of marketcap data  
                        data[f'tradingvol_{ticker}'] = data[f'close_{ticker}'] * data[f'volume_{ticker}']
                        data_container[ticker] = data

                        # making up and down labels (y_data)
                        next_p = data[f'close_{ticker}'].shift(-1).fillna(method='ffill')
                        _r = next_p/data[f'close_{ticker}']-1
                        y_container[ticker] = pd.DataFrame([1 if i>=0 else 0 for i in _r ], index=data.index) ## 정답지
                        
                        print(ticker, data[f'close_{ticker}'].index[0], data[f'close_{ticker}'].index[-1],
                        len(data[f'close_{ticker}']), len(next_p), len(_r), len(y_container[ticker]))
                
            
            # Make data to the dataframe
            final_data = pd.DataFrame()
            y_data = pd.DataFrame()

            for tic in data_container:
                # 옆으로 붙여야 의미가 맞지 않을까?
                final_data = pd.concat([final_data, data_container[tic]], axis=1, join='outer')
                y_data = pd.concat([y_data, y_container[tic]], axis=1, join='outer')
            
            # set the labels
            y_data.columns = stocks_subset
            
            
            final_data.sort_index(ascending=True, inplace=True)
            y_data.sort_index(ascending=True, inplace=True)
            final_data.fillna(method='ffill', inplace=True)
            y_data.fillna(method='ffill', inplace=True)
            #---------------------------------------------------------------------------------------------------
        
        else :
            # critic network
            final_data, y_data = Timeseries3D._make_critic_label(stocks_subset, pretrain_json, window=window, init_portfolio=10000)

        
        if save_option:
            final_data.to_csv(f'{DATA_DIR}pretrain/{country}/etf_2d_{network}.csv')
            y_data.to_csv(f'{DATA_DIR}/pretrain/{country}/etf_2d_y_{network}.csv')
            
        # to numpy?
        if to_numpy:
            final_data = final_data[data.columns].to_numpy()
            y_data = y_data.to_numpy()

        return final_data, y_data

    

    @staticmethod
    def make_3Ddata(data_2d, y_data, ticker_file:str or list, time_first=True):
        '''
        Args
        - Same as Timeseries3D.make_2Ddata
        - data_2d : pd.DataFrame, size(T, assets*features)
        Out
        - data_3d : np.array()
            - (6, 30, 4536)
            - 차원 (assets, T, features) --> channel 축이 features수
        '''
        if type(ticker_file) ==  str:
            with open(ticker_file) as f:
                stocks_subset = f.read().splitlines()
                stocks_subset = [ticker for ticker in stocks_subset]
        else :
            stocks_subset = ticker_file
        
        # chg dat 2d to 3d
        # 원래대로면 stocks_subset=30 이어야하는데 29, 28로 다르므로...
        stock_number = len(stocks_subset)
        data_3d_arr = np.empty([len(data_2d),stock_number,int(len(data_2d.columns)/stock_number)])

        if time_first == True :
            data_3d_arr = np.empty([len(data_2d),stock_number,int(len(data_2d.columns)/stock_number)])

            for idx, ticker in enumerate(stocks_subset):
                temp1 = []
                for col in data_2d.columns:
                    if ticker == col.split('_')[-1] :
                        # for close price
                        temp1.append(data_2d[col])
                try: 
                    data_3d_arr[:,idx,:] = np.array(temp1).T ####
                except:
                    print(ticker, col, " data doesn't exist.")
                    pass
        
        else:
            # asset first
            data_3d_arr = np.empty([stock_number,len(data_2d),int(len(data_2d.columns)/stock_number)])
            
            for idx, ticker in enumerate(stocks_subset):
                temp1 = []
                for col in data_2d.columns:
                    if ticker == col.split('_')[-1] :
                        # for close price
                        temp1.append(data_2d[col])
                try: 
                    data_3d_arr[idx,:,:] = np.array(temp1).T
                except:
                    print(ticker, col, " data doesn't exist.")
                    pass

        if type(y_data) != np.array:
            y_data = np.array(y_data) # 이것도 30, 4356

        return data_3d_arr, y_data



    @staticmethod
    def make_window_data(x_data, window:int=40):
        """
        make window data: 윈도우 사이즈만큼의 데이터가 있음.
        Args

        Out
        - window_x_data
            - 2차원의 경우 (T-window, 40, Asset*features)
            - 3차원의 경우 (T-window, 40, Asset, features)
        
        - y_data : answer for pretrain
        """
        # window size만큼의 시계열 조각 데이터 셋 만들기
        temp = []

        if type(x_data) == pd.DataFrame:
            # dataset이 2차원 dataframe일때
            for i in range(len(x_data)-window):
                temp.append(x_data.iloc[i:i+window, :].to_numpy())
            
            window_x_data = np.array(temp)
        
        
        else:
            # data type numpy
            if x_data.ndim == 3:
                # ㅠㅠ asset이 먼저오면 이것도 조건문화 시켜야함
                for i in range(window, x_data.shape[0]):
                    # temp.append(x_data[:,:,i-window:i])
                    temp.append(x_data[i-window:i,:,:])
                
                # make array : (T, asset, timeseries, features)
                window_x_data = np.array(temp)
                # window_x_data = window_x_data.transpose(0,2,3,1)
                
            else :
                # 2차원에서는 t, 30*features 이렇게 옴...
                for i in range(window, len(x_data)):
                    temp.append(x_data[i-window:i]) 
                
                # make array : (T, window, features*asset)
                window_x_data = np.array(temp)
        
        print("window_x_data.shape : ", window_x_data.shape) 


        return window_x_data





class NDStandardScaler(TransformerMixin):
    '''
    If you want to scale each feature differently, use this scaler
    https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix

    
    '''
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X




def main():
    # for code test & make new etf data
    # option setting
    country = "USA" #### should add options
    pretrain = True
    initial_date = '2000-01-01'
    final_date = '2021-12-31'
    window = 40

    # directory setting
    # basic_dir = '/home/ubuntu2010/바탕화면/'
    tickers_subset = f"{BASE_DIR}/portfolios_and_tickers/ticker_diy.txt"
    pretrain_subset = f"{DATA_DIR}/pretrain/pretrain_description.json"

    if pretrain:
        data_dir = f'{DATA_DIR}/pretrain/'
    else: 
        data_dir = f'{DATA_DIR}/stockdata/30stocks_2000/'


    # pretrain data 완료
    pretrain_df, pretrain_y = Timeseries3D.make_2D_pretrain_data(pretrain_subset, 
                                                    tickers_subset, 
                                                    initial_date, 
                                                    final_date,
                                                    country = 'USA',
                                                    network = 'critic',
                                                    window = window,
                                                    init_portfolio = 10000,
                                                    save_option = True,
                                                    to_numpy=False)
    # import ipdb
    # ipdb.set_trace()

    pretrain_3d, pretrain_y_3d = Timeseries3D.make_3Ddata(pretrain_df, pretrain_y, tickers_subset)
    print(pretrain_3d.shape) # (feature, assets, times)

    test = Timeseries3D.make_window_data(pretrain_3d, 40)
    print(test.shape)
    # done!
    pass





if __name__ == "__main__":
    main()