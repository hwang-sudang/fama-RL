#https://www.kaggle.com/code/szaitseff/online-learning-of-time-series-with-lstm-rnn

import os
import torch
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Tuple
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


from util.configures import *
from actor_pretrain.network import Decomposition as EMD


##### 괜히 만들었다... #######



class PretrainSlidingDataset(Dataset):
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

        super(PretrainSlidingDataset, self).__init__()        

        # 2. WINDOW_SIZE씩 묶어서 처리하기
        # 처음에는 dataset(close, logr, label)을 포함하는 데이터 프레임이 들어올 것. 
        self.window = window
        self.scale = log_scale
        self.dataset = dataset
        self.emd_mode = emd_mode
        self.action_space_dimension = ACTION_SPACE_DIMENSION
        self._preprocessing()
        self._tvt_split()

        
    def _preprocessing(self):
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



    def __getitem__(self, index:int, iter:int) -> torch.tensor:
        '''
        Args
        - index(int): 데이터 인덱스
        - iter(int): 슬라이딩 윈도우를 세는 단위의 인덱스


        # help indexing: 0번째는 x 데이터, 1번째는 y
        # dictionary 형태로 바꾸면 유지보수 편함
        # metric에서는  price*pred, price*y를 비교함

        ## 예상 사용법
        dataset.__getitem__()
        '''

        self.dic = {"x": self.x_data[index],
                    "price": self.price_data[index],
                    "y": self.y_data[index]}
        
        # return self.dic
        return self.x_data[index], self.price_data[index], self.y_data[index]

    
    def __len__(self):
        # length func
        return self.y_data.shape[0]
    
    
    def _tvt_split(self, period:int = 300):
        '''
        Args
        - period : 슬라이딩 윈도우 기간을 적용할(tvt를 할) 기간.
        
        1. 데이터를 일정 기간(period)으로 자른다
        2. period 일자씩 자른 후에 7:1:2로 데이터를 나눈다.
        3. period 수 별로 tvt정리
        '''
        # x_data = np.array(self.dataset["log_p"]) > self.x_data
        # p_data = np.array(self.dataset["close"]) > self.p_data
        # y_data = np.array(self.dataset["log_label"]) > self.y_data

        slide_dic = {}
        n_period = len(self.y_data)//period +1 
        
        # 1. 데이터를 일정 기간(period)으로 자른다
        for i in range(1, n_period+1):
            x = self.x_data[(i-1)*n_period:i*n_period]
            y = self.y_data[(i-1)*n_period:i*n_period]
            p = self.p_data[(i-1)*n_period:i*n_period]

            # 2. 자른 데이터에서 7:1:2로 데이터를 나눈다.
            assert len(x) == len(y) == len(p), 'Each Data Length are different: sliding_loader 133 line'
            n_data = len(x)

            # 3. period 수 별로 tvt정리
            slide_dic[i] = {
                            "train": {"x": x[:int(n_data*0.7)],
                                      "y": y[:int(n_data*0.7)],
                                      "p": p[:int(n_data*0.7)]},

                            "val": {"x": x[int(n_data*0.7):int(n_data*0.8)],
                                      "y": y[int(n_data*0.7):int(n_data*0.8)],
                                      "p": p[int(n_data*0.7):int(n_data*0.8)]},

                            "test": {"x": x[:int(n_data*0.7)],
                                      "y": y[:int(n_data*0.7)],
                                      "p": p[:int(n_data*0.7)]},
                            }

            self.slide_dic = slide_dic
            pass



    
    def input_size(self):
        # input_size 
        return self.x_data.shape



@dataclass
class SlidingData:
    '''
    해보고 못쓰겠으면 dict로 바꾸기
    '''
    #test 
    id: int
    name: str
    birthdate: date
    admin: bool = False

    # Real
    n_period : int
    x : List[torch.tensor] # self.x_data
    y : List[torch.tensor] # self.y_data
    p : List[torch.tensor] # self.p_data




