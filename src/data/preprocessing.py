import os
import re
import numpy as np
import pandas as pd
import json
from datetime import date, datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# inner module import
from util.configures import *


class DataPreprocessing():
    def __init__(self, args, base_dir, data_dir):
        self.args = args
        self.base_dir = base_dir
        self.data_dir = data_dir
        self.startdate = args.initial_date
        self.enddate = args.final_date
        self.window = args.window
        self.country = args.country


    def load_data(
                self,
                BASE_DIR:str,
                DATA_DIR:str, 
                country:str,) -> pd.DataFrame:
        '''load and concat'''
        asset_name = re.sub(r'[0-9]+', '', country)
        datadir = f'{DATA_DIR}/stockdata/final_with_factor/{asset_name}/{FACTOR}'
        ticker_file = BASE_DIR+f'/portfolio_info/{country}/snp_portfolio_smb.json'
        with open(ticker_file, 'r') as f:
            portfolio_info = json.load(f)
            self.stocks_subset = list(portfolio_info["market_cap"].keys())
            
        # ohlcv, factor dataset 분리해서 넣기 : 전처리 때문
        ohlcv_df = pd.DataFrame()
        factor_df = pd.DataFrame()
        for ticker in self.stocks_subset:
            data = pd.read_csv(datadir+f'/{ticker}.csv', parse_dates=['date'], index_col=0)
            data.rename(columns={"marketCap_new" : "market_cap", "MarketCap" : "market_cap"}, inplace=True)
            data.columns = [i+f"_{ticker}" for i in data.columns] # ex: 'high_{ticker}'
            ohlcv_df = pd.concat([ohlcv_df, data.iloc[:, :5]], axis=1)
            factor_df = pd.concat([factor_df, data.iloc[:, 5:-1]], axis=1) ###
        
        if FACTOR == 'rep':
            self.rep_factor_name = self.factor_name = ["market_cap", "BTM", "std"]
        else :
            self.rep_factor_name = ["market_cap", "BTM", "std"]
            self.factor_name = [i.split("_")[0] for i in data.columns[5:-1]]
        return ohlcv_df, factor_df


    def data_split(self, data, 
                start_date:str,
                end_date:str,):
        df = data[data.index>=start_date]
        df = df[df.index<=end_date]        
        df = df.fillna(method='ffill')
        
        # check there's any nan data
        print("dataframe number of nan  :", df.isnull().sum().sum())
        return df
    

    
    def minmax_scale(self, data:pd.DataFrame, axis=1, 
                     min=None, max=None, 
                     feature_range=(0,1), reverse=False) -> np.array:
        '''
        Factor 지표에 대해서 스케일링 처리해줌
        Output
            - data(pd.DataFrame) : minmax 스케일링 할 데이터 
            - axis(int) : 스케일링을 적용할 축 
                - Factor 의 경우 가로축 데이터를 대상으로 할때는 axis=1)
            - min(np.array) :
            - max(np.array) :
            - feature_range(tuple of 2) :
            - reverse(bool) : 

        Output
            - scaled_data(pd.DataFrame) : 스케일링 적용이 완료된 데이터
            - data_min, data_max (np.array) : 민맥스 스케일링 정보 담은 리스트
        '''
        out_df = pd.DataFrame()
        feat_min, feat_max = feature_range

        # Grouping with same factors from each stock TI data
        f_dict = {f:[] for f in self.factor_name}
        for f in self.factor_name:
            for col in data:
                if f in col: f_dict[f].append(data[col])
            
            # Grouping Factors
            X = pd.concat(f_dict[f], axis=1)

            # SEet Low-volatility, small-cap first
            if f == 'market_cap' or f == 'std':
                X = X *(-1)

            # minmax scaling
            min = np.expand_dims(X.values.min(axis=axis), axis=-1)
            max = np.expand_dims(X.values.max(axis=axis), axis=-1)
            X_std = (X-min)/(max - min) 
            X_scaled = X_std*(feat_max-feat_min)+feat_min
            out_df = pd.concat([out_df, X_scaled], axis=1) #names=key

        return out_df #, min, max




    def score_scale(self, data:pd.DataFrame, axis=1, 
                     feature_range=(0,1)) -> pd.DataFrame:
        '''
        Factor 지표에 대해서 스케일링 처리해줌
        Output
            - data(pd.DataFrame) : minmax 스케일링 할 데이터 
            - axis(int) : 스케일링을 적용할 축 
                - Factor 의 경우 가로축 데이터를 대상으로 할때는 axis=1)
            - min(np.array) : 최소값, 기존 스케일링 범위가 있는 경우 지정
            - max(np.array) : 최대값, 기존 스케일링 범위가 있는 경우 지정
            - feature_range(tuple of 2) : 값의 범위

        Output
            - scaled_data(pd.DataFrame) : 스케일링 적용이 완료된 데이터
            - data_min, data_max (np.array) : 민맥스 스케일링 정보 담은 리스트
        '''
        out_df = pd.DataFrame()
        feat_min, feat_max = feature_range

        # data에서 같은 factor끼리 묶어야함..
        f_dict = {f:[] for f in self.factor_name}

        for f in self.factor_name:
            for col in data:
                if f in col: f_dict[f].append(data[col])
            # grouping factors 
            X = pd.concat(f_dict[f], axis=1)
            ######
            if f == 'market_cap' or f == 'std':
                # 저변동성, 소형주 우선
                X = X *(-1)
            
            # 1st : scoring (ranking)
            rank_X = np.empty_like(X.values)
            for l in range(len(X.values)):
                temp = np.argsort(X.values[l])
                rank_X[l, temp] = np.arange(len(X.values[l]))
            
            # minmax scaling
            min = np.expand_dims(rank_X.min(axis=axis), axis=-1)
            max = np.expand_dims(rank_X.max(axis=axis), axis=-1)
            X_std = (rank_X-min)/(max - min) 
            X_scaled = X_std*(feat_max-feat_min)+feat_min
            X_scaled = pd.DataFrame(X_scaled, columns = X.columns, index=X.index) # for pandas concat
            out_df = pd.concat([out_df, X_scaled], axis=1) #names=key
        return out_df #, min, max




    def log_scale(self, data) -> pd.DataFrame:
        '''
        모든 데이터를 log 변화율로 변환한다: 근데.. 이거 팩터 정보를 손상시킬 수도 있겠는데.

        여기서 만들어져야 하는 데이터
        - close price
        - log close
        - all log df
        '''
        close_p = pd.DataFrame([data[i] for i in data.columns if "close" in i]).T
        log_df = np.log(data/data.shift(1).fillna(method='bfill'))
        log_df = log_df.fillna(method='ffill')
        # inf값을 어떻게 하면 더 효율적으로 바꿀 수 있을까...
        log_df = log_df.replace([np.inf, -np.inf], [log_df.max(axis=0), log_df.min(axis=0)])
        log_close = pd.DataFrame([log_df[i] for i in log_df.columns if "close" in i]).T  

        return log_df, log_close, close_p
    
    

    def make3Ddata(self,
                data_2d:pd.DataFrame,
                stocks_subset:list,) -> np.array:
        '''
        Args
        - Same as Timeseries3D.make_2Ddata
        - data_2d : pd.DataFrame, size(T, assets*features)
        - close_p : pd.DataFrame
        Out
        - data_3d : np.array()
            - input : (6, 30, 4536)
            - output : 차원 (T, assets, features) --> channel 축이 features수
        - y_data : np.array()
        '''
        # chg data 2d to 3d
        stock_number = len(stocks_subset)
        data_3d_arr = np.empty([len(data_2d),stock_number,int(len(data_2d.columns)/stock_number)])

        for idx, ticker in enumerate(stocks_subset):
            temp1 = []
            for col in data_2d.columns:
                if ticker == col.split('_')[-1] : temp1.append(data_2d[col]) # for close price    
            try: 
                data_3d_arr[:,idx,:] = np.array(temp1).T 
            except:
                print(ticker, col, " data doesn't exist.")
        return data_3d_arr


    def make_window_data(self, x_data, window:int=40):
        """
        # make window data: window size만큼의 시계열 조각 데이터 셋 만들기
        Args
            - x_data : 윈도우 사이즈 별로 시계열을 잘라 데이터 셋을 만든다.
        Out
            - window_x_data
                - 2차원의 경우 (T-window, 40, Asset*features)
                - 3차원의 경우 (T-window, 40, Asset, features)
            - y_data : answer for pretrain
        """
        temp = []
        if type(x_data) == pd.DataFrame:
            # dataset이 2차원 dataframe일때
            for i in range(len(x_data)-window):
                temp.append(x_data.iloc[i:i+window, :].to_numpy())
            window_x_data = np.array(temp)
        else:
            # data type numpy
            if x_data.ndim == 3:
                for i in range(window, x_data.shape[0]):
                    temp.append(x_data[i-window:i,:,:])
                window_x_data = np.array(temp) # (T, asset, timeseries, features)
            else :
                for i in range(window, len(x_data)):
                    temp.append(x_data[i-window:i]) 
                window_x_data = np.array(temp) #  (T, window, features*asset)
        print("window_x_data.shape : ", window_x_data.shape) 
        return window_x_data


    def __call__(self):
        '''
        Output
            - log_series (timelength-window, window, assets, features) : 4D data, 모든 팩터 데이터 + window 단위, 로그변화율로 변환된 데이터셋 (for LSTM model)
            - log_df_3d (timelength, assets, features) : 3D data, 모든 팩터 데이터 + 1일단위 로그변화율로 변환된 데이터셋
            - log_close_series : Only 종가 로그 변화율 데이터 (for pretrain) 
            - close_p: 종가 원본 데이터 
            - y_label: 다음날의 로그 변화율 데이터, pretrain용 데이터셋
        '''
        # 여기서 factor와 ohlcv 나누기
        ## 그러려면 우선 factor와 ohlcv를 합쳐야함
        ohlcv_df, factor_df = self.load_data(self.base_dir, self.data_dir, self.country)
        ohlcv_df = self.data_split(ohlcv_df, self.startdate, self.enddate)
        factor_df  = self.data_split(factor_df, self.startdate, self.enddate)
        
        # 전처리
        log_df, log_close, close_p = self.log_scale(ohlcv_df)
        # scaled_factor = self.minmax_scale(factor_df, axis=1, feature_range=(-1,1)) 
        # scaled_factor = self.score_scale(factor_df, axis=1, feature_range=(-1,1)) 
        ## what about mixing rankscale * minmaxscale?
        scaled_factor = self.minmax_scale(factor_df, axis=1, feature_range=(-1,1)) * \
                        self.score_scale(factor_df, axis=1, feature_range=(-1,1)) 

        # Concatenate log return and min-maxed factor data
        concat_df = pd.concat([log_df, scaled_factor], axis=1)  # 이거 data 범위 생각해보기
        concat_df = concat_df.clip(-3, 3, axis=0) 
        log_df_3d = self.make3Ddata(data_2d=concat_df, stocks_subset=self.stocks_subset) #(timelength, assets, features)

        # make window dataset
        log_series = self.make_window_data(log_df_3d, window=self.window) #(timelength-window, window, assets, features)
        y_label = log_close.shift(-1).fillna(method='ffill')
        return log_series, log_df_3d, log_close, close_p, y_label






















class DataPreprocess1():
    def __init__(self):
        pass

    @staticmethod
    def load_data(BASE_DIR:str,
                DATA_DIR:str, 
                country:str,) -> pd.DataFrame:
        '''load and concat'''

        ticker_file = BASE_DIR+f'/portfolio_info/{country}/snp_portfolio_smb.json'
        with open(ticker_file, 'r') as f:
            portfolio_info = json.load(f)
            stocks_subset = list(portfolio_info["market_cap"].keys())

        data_container = {}
        for ticker in stocks_subset:
            data = pd.read_csv(DATA_DIR+f'/{ticker}.csv', parse_dates=['date'], index_col=0)
            if 'market cap' in data.columns:
                data.drop('market cap', axis=1, inplace=True)
            data.columns = [f'open_{ticker}', f'high_{ticker}', 
                            f'low_{ticker}', f'close_{ticker}', 
                            f'volume_{ticker}', f'BTM_{ticker}',f'market_cap_{ticker}']
            # 데이터 티커 정보 추가하고 데이터 컨테이너에 임시저장
            data_container[ticker] = data

        # Make dict data to the dataframe
        final_data = pd.DataFrame()
        for tic in data_container:
            key = tic
            final_data = pd.concat([final_data, data_container[tic]], axis=1, names=key)
        return final_data


    @staticmethod
    def data_split(data, 
                start_date:str,
                end_date:str,):
        # start_date = datetime.strptime(start_date, '%Y-%m-%d')
        # end_date = datetime.strptime(end_date, '%Y-%m-%d')
        df = data[data.index>=start_date]
        df = df[df.index<=end_date]        
        # check there's any nan data
        print("dataframe number of nan  :", df.isnull().sum().sum())
        return df
    
    
    @staticmethod
    def log_scale(data) -> pd.DataFrame:
        '''
        모든 데이터를 log 변화율로 변환한다: 근데.. 이거 팩터 정보를 손상시킬 수도 있겠는데.

        여기서 만들어져야 하는 데이터
        - close price
        - log close
        - all log df
        '''
        close_p = pd.DataFrame([data[i] for i in data.columns if "close" in i]).T
        log_df = np.log(data/data.shift(1).fillna(method='bfill'))
        log_df = log_df.fillna(method='ffill')
        log_close = pd.DataFrame([log_df[i] for i in log_df.columns if "close" in i]).T        
        return log_df, log_close, close_p
    
    
    @staticmethod
    def make3Ddata(data_2d:pd.DataFrame,
                   stocks_subset:list,) -> np.array:
        '''
        Args
        - Same as Timeseries3D.make_2Ddata
        - data_2d : pd.DataFrame, size(T, assets*features)
        - close_p : pd.DataFrame
        Out
        - data_3d : np.array()
            - input : (6, 30, 4536)
            - output : 차원 (T, assets, features) --> channel 축이 features수
        - y_data : np.array()
        '''
        # chg data 2d to 3d
        stock_number = len(stocks_subset)
        data_3d_arr = np.empty([len(data_2d),stock_number,int(len(data_2d.columns)/stock_number)])

        for idx, ticker in enumerate(stocks_subset):
            temp1 = []
            for col in data_2d.columns:
                if ticker == col.split('_')[-1] : temp1.append(data_2d[col]) # for close price    
            try: 
                data_3d_arr[:,idx,:] = np.array(temp1).T ####
            except:
                print(ticker, col, " data doesn't exist.")
                pass
        return data_3d_arr


    @staticmethod
    def make_window_data(x_data, window:int=40):
        """
        # make window data: window size만큼의 시계열 조각 데이터 셋 만들기
        Args
            - x_data : 윈도우 사이즈 별로 시계열을 잘라 데이터 셋을 만든다.
        Out
            - window_x_data
                - 2차원의 경우 (T-window, 40, Asset*features)
                - 3차원의 경우 (T-window, 40, Asset, features)
            - y_data : answer for pretrain
        """
        
        temp = []
        if type(x_data) == pd.DataFrame:
            # dataset이 2차원 dataframe일때
            for i in range(len(x_data)-window):
                temp.append(x_data.iloc[i:i+window, :].to_numpy())
            window_x_data = np.array(temp)
        else:
            # data type numpy
            if x_data.ndim == 3:
                for i in range(window, x_data.shape[0]):
                    temp.append(x_data[i-window:i,:,:])
                window_x_data = np.array(temp) # (T, asset, timeseries, features)
            else :
                for i in range(window, len(x_data)):
                    temp.append(x_data[i-window:i]) 
                window_x_data = np.array(temp) #  (T, window, features*asset)
        print("window_x_data.shape : ", window_x_data.shape) 
        return window_x_data









