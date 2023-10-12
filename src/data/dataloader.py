import os
import numpy as np
import pandas as pd
import json
import torch
from datetime import date, datetime, timedelta
from torch.utils.data import Dataset


# inner module import
from util.configures import *
from util.decompostion import Decomposition as EMD
from data.preprocessing import DataPreprocessing as DP


class PortfolioDataset():
    '''
    For Custom dataset split, prepare data as torch Dataset.
    Args:
        - x_data(pd.DataFrame)
        - y_data(pd.Series)

    Out: Dataset 객체 
        - series(np.array) : 윈도우 길이만큼의 모든 시계열 정보 #(timelength-window, window, assets, features)
        - series_t(np.array) : t시점의 모든 시계열 정보 #(timelength, assets, features)
        - close_x(np.array) : 윈도우 길이만큼의 close정보, 시계열 정보
        - p_t(pd.DataFrame) : 로그 변환을 위한 마지막 윈도우의 종가
        - y(pd.DataFrame) (target) : pretrain을 위한 정답 logreturn (timelength-window, assets)
    '''
    def __init__(self,
                args:dict,
                log_scale:int = 1,
                emd_mode:bool=True,
                ):
        super(PortfolioDataset, self).__init__()
        self.mode = args.mode # train or test
        self.country = args.country
        self.prewindow = args.prewindow
        self.window = args.window
        self.scale = log_scale
        self.emd_mode = emd_mode
        self.initial_date = args.initial_date
        self.final_date = args.final_date
        self.dataprocess = DP(args, BASE_DIR, DATA_DIR)
        self.log_series, self.log_series_t, self.log_close_series, self.close_p, self.y_label = self.dataprocess() # 첫번째만 window로 잘려있음
        self.dates = self.close_p.index # 이것도 40일 밀림
        self._decompose(self.log_close_series, self.close_p, self.y_label)
        
    

    def _decompose(self, x_data, p_data, y_data):
        # 우선 emd파일이 있는지 확인하기
        emd_filedir = f'{BASE_DIR}/emd_data/{self.country}/{self.mode}/{EMD_FILE}_w{self.prewindow}_{self.initial_date}_{self.final_date}.npz'
        if os.path.isfile(emd_filedir):
            print("Loading EMD decomposition npz file....")
            npz = np.load(emd_filedir)
            x_array = npz['x_data']
            price_array = npz['price_data']
        
        else:
            os.makedirs(os.path.dirname(emd_filedir), exist_ok=True) # 저장파일 만들 경로 설정
            x_array = []
            price_array = []
            x_data = np.array(x_data) # only close
            p_data = np.array(p_data)
            y_data = np.array(y_data) # shape 확인, 데이터 확인


            if self.emd_mode: 
                print("start EMD mode...") 
            for i in range(self.prewindow, len(x_data)):
                # 애초에 앞의 40일 데이터가 들어오면
                # x_data가 (40, 30)
                if self.emd_mode: 
                    x_ = EMD.emd_data_transform2D(x_data[i-self.prewindow:i])
                    # print(x_)
                else : x_ = x_data[i-self.prewindow:i]
                x_array.append(x_)
                price_array.append(p_data[i-1])
            
            # Save numpy data
            x_array = np.array(x_array)
            price_array = np.array(price_array)
            # print(x_array)
            np.savez(emd_filedir, x_data=x_array, price_data=price_array)

        
        # Array to Tensor
        self.x_data = torch.FloatTensor(x_array*self.scale) # torch.Size([4496, self.prewindow, 30])
        self.price_data = torch.FloatTensor(price_array) # torch.Size([4496, 30]) 
        self.y_data = torch.FloatTensor(np.array(y_data[self.prewindow:])) # torch.Size([4496, 30])
        self.log_series = torch.FloatTensor(np.array(self.log_series)) # torch.Size([4496, self.window, 30, 9])
        self.log_series_t = torch.FloatTensor(np.array(self.log_series_t[self.window:])) # torch.Size([4496, (1), 30, 9])
        
        assert len(self.x_data)==len(self.y_data), "number of x_data and y_data set doesnt match..."
        assert len(self.x_data)==len(self.price_data), "number of x_data and price_data set doesnt match..."
        print("EMD Dataset Done.")
        
 
    def getdata(self):
        '''
        
        '''
        self.data = {"dates": self.dates,
                    "series" : self.log_series, # prestate
                    "series_t" : self.log_series_t,
                    "log_close": self.x_data,
                    "close_t": self.price_data,
                    "y": self.y_data}
        
        return self.data


    def __getitem__(self, index):
        '''
        - series : 윈도우 길이만큼의 모든 시계열 정보
        - series_t : t시점의 모든 시계열 정보
        - log_close : 윈도우 길이만큼의 close의 log return정보, 시계열 정보
        - close_t : 로그 변환을 위한 마지막 윈도우의 종가
        - y(target) : pretrain을 위한 정답 logreturn
        '''

        # help indexing: 0번째는 x 데이터, 1번째는 y
        # dictionary 형태로 바꾸면 유지보수 편함
        # metric에서는  price*pred, price*y를 비교함
        self.dic = {"dates": self.dates[index],
                    "series" : self.log_series[index],
                    "series_t" : self.log_series[index, -1], # self.log_series_t 같나?
                    "log_close": self.x_data[index],
                    "close_t": self.price_data[index],
                    "y": self.y_data[index]}
        
        return self.dic['series'], self.dic['series_t'], self.dic['log_close'], self.dic['close_t'], self.dic['y']

    
    def __len__(self):
        # length func
        return self.y_data.shape[0]









class PretrainDataset(Dataset):
    '''
    For Custom dataset split, prepare data as torch Dataset.
    Args:
        - x_data(pd.DataFrame)
        - y_data(pd.Series)

    Out: Dataset 객체 
        - x(series) : 윈도우 길이만큼의 close정보, 시계열 정보
        - price : 로그 변환을 위한 마지막 윈도우의 종가
        - y(target) : 정답 logreturn
    '''

    def __init__(self, 
                dataset,
                ACTION_SPACE_DIMENSION,
                window,
                log_scale:int = 1,
                emd_mode:bool=True,
                ):

        super(PretrainDataset, self).__init__()        

        # 2. WINDOW_SIZE씩 묶어서 처리하기
        # 처음에는 dataset(close, logr, label)을 포함하는 데이터 프레임이 들어올 것. 
        self.window = window
        self.scale = log_scale
        self.dataset = dataset
        self.emd_mode = emd_mode
        self.action_space_dimension = ACTION_SPACE_DIMENSION

        x_array = []
        price_array = []
        x_data = np.array(self.dataset["log_p"])
        p_data = np.array(self.dataset["close"])
        y_data = np.array(self.dataset["log_label"])

        # input 크기에 따라
        if x_data.shape[0] == self.action_space_dimension:
            # x_data.shape가 asset, Time, features 일때...
            for i in range(self.window, len(x_data)):
                if self.emd_mode: x_ = EMD.emd_data_transform_per_batch(x_data[:,i-self.window:i,:])
                else : x_ = x_data[:,i-self.window:i,:]
                x_array.append(x_) 
                price_array.append(p_data[:,i-1,:])
            self.x_data = torch.FloatTensor(np.array(x_array)*self.scale) # torch.Size([4496, 30, 40, 6])
            self.price_data = torch.FloatTensor(np.array(price_array))

        else : 
            # data shape : (Time, asset, features)
            for i in range(self.window, len(x_data)):
                # 애초에 앞의 40일 데이터가 들어오면
                if self.emd_mode: x_ = EMD.emd_data_transform1D(x_data[i-self.window:i])
                else : x_ = x_data[i-self.window:i]
                x_array.append(x_)
                price_array.append(p_data[i-1])
            self.x_data = torch.FloatTensor(np.array(x_array)*self.scale) # torch.Size([4496, 40, 30, 6])
            self.price_data = torch.FloatTensor(np.array(price_array)) # torch.Size([4496, 30]) 

        self.y_data = torch.FloatTensor(np.array(y_data[self.window:])) # torch.Size([4496, 30]) 
        
        assert len(self.x_data)==len(self.y_data), "number of x_data and y_data set doesnt match..."
        assert len(self.x_data)==len(self.price_data), "number of x_data and price_data set doesnt match..."
        print("EMD Dataset Done.")

    
    def __getitem__(self, index):
        # help indexing: 0번째는 x 데이터, 1번째는 y
        # dictionary 형태로 바꾸면 유지보수 편함
        # metric에서는  price*pred, price*y를 비교함
        self.dic = {"x": self.x_data[index],
                    "price": self.price_data[index],
                    "y": self.y_data[index]}
        # return self.dic
        return self.x_data[index], self.price_data[index], self.y_data[index]

    
    def __len__(self):
        # length func
        return self.y_data.shape[0]