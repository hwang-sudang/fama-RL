# !pip install PyPortfolioOpt
# '/home/ubuntu2010/바탕화면/DEV/trading12/'
# '/nas3/hwang/pretrains/pretrain_RL'

import sys
import os
import torch
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


# inner module import
# from src.util.data import *
from util.configures import *
# from network import Decomposition as EMD
from actor_pretrain.network import Decomposition as EMD




class PretrainUtilites:
    '''
    Pretrain에서 사용할 기타 함수들(utilities)
    
    # Function List
    - data_split
    - normalizer
    - saveModel
    '''

    @staticmethod
    def data_split(data, 
                    START_DATE, TRAIN_END, VAL_END, TEST_END, 
                    window=40,
                    y_data:bool=False):
        '''
        '''
        train_df = data[data.index>=START_DATE]
        train_df = train_df[train_df.index<TRAIN_END]
        val_df = data[data.index>=TRAIN_END]
        val_df = val_df[val_df.index<VAL_END]    
        test_df = data[data.index>=VAL_END]
        test_df = test_df[test_df.index<=TEST_END]

        return train_df, val_df, test_df

    @staticmethod
    def normalizer(train_df:pd.DataFrame, 
                val_df:pd.DataFrame, 
                test_df:pd.DataFrame,
                normalizer:str = None):
        ''' 
        정규화를 위한 함수
        다음단계에 있을 라벨데이터 생성과 torch.dataset으로의 변환을 위해서
        출력물은 모두 데이터 프레임으로
        '''
        print(f"Scaler Mode : {normalizer}")
        
        if normalizer == "log":
            train = (np.log(train_df) - np.log(train_df.shift(1)))
            val = (np.log(val_df) - np.log(val_df.shift(1)))
            test = (np.log(test_df) - np.log(test_df.shift(1)))

            # chg는 log 변환시 정보 삭제 : 그런데 이걸 이렇게 처리해도되나?

            for col in train_df.columns:
                if 'chg' in col:
                    # train[col] = train_df[col]
                    # val[col] = val_df[col]
                    # test[col] = test_df[col]
                    train = train.drop(col, axis=1)
                    val = val.drop(col, axis=1)
                    test = test.drop(col, axis=1)

            # train["Change %"] = train_df["Change %"]
            # val["Change %"] = val_df["Change %"]
            # test["Change %"] = test_df["Change %"]
            train = train.fillna(method='bfill', limit=1)
            val.iloc[0] = train.iloc[-1]
            test.iloc[0] = val.iloc[-1]

        else :
            scaler = MinMaxScaler()
            train = scaler.fit_transform(train_df)
            val = scaler.transform(val_df)
            test = scaler.transform(test_df)

            train = pd.DataFrame(train, columns=train_df.columns)
            val = pd.DataFrame(val, columns=val_df.columns)
            test = pd.DataFrame(test, columns=test_df.columns)
        return train, val, test

    @staticmethod
    def saveModel(model, save_directory, network_name):
        '''Function to save the model'''
        # save_path = f"{base_dir}/actor_pretrain/bestmodel/weights/%s" % (network_name)
        save_path = f"{save_directory}/logs/bestmodel/weights/{network_name}"
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        path = f"{save_path}/BestModel_{network_name}.pth"
        torch.save(model.state_dict(), path)
        # print(f"---------------------------------SAVED MODEL AT {path} -------------------------------\n")







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
    

    def input_size(self):
        # input_size 
        return self.x_data.shape






class MakeDataset(Dataset):
    '''
    For Custom dataset split, prepare data as torch Dataset.
    Args:
    - x_data(pd.DataFrame)
    - y_data(pd.Series)

    Out: Dataset 객체 
    -  iter(Dataset객체) 한 후 for문 돌려서 나오는 각 값은
    - Out[0] : x_data (input feature)
    - Out[1] : y_data (label)
    '''

    def __init__(self, x_data:np.array, y_data:np.array,
                ACTION_SPACE_DIMENSION, WINDOW_SIZE, emd_mode:bool=True):
        '''
        수정사항 : 바로 데이터 프레임으로 받으면 안되나?
        '''
        super(MakeDataset, self).__init__()
        # WINDOW_SIZE 40개씩 묶어서 처리하기
        x_array = []
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # input 크기에 따라
        if x_data.shape[0] == ACTION_SPACE_DIMENSION:
            # x_data.shape가 asset, Time, features 일때...
            for i in range(WINDOW_SIZE, len(x_data)):
                if emd_mode: x_ = EMD.emd_data_transform_per_batch(x_data[:,i-WINDOW_SIZE:i,:])
                else : x_ = x_data[:,i-WINDOW_SIZE:i,:]
                x_array.append(x_) 
            self.x_data = torch.FloatTensor(np.array(x_array)) # torch.Size([4496, 30, 40, 6])

        else : 
            for i in range(WINDOW_SIZE, len(x_data)):
                # 애초에 앞의 40일 데이터가 들어오면
                if emd_mode: x_ = EMD.emd_data_transform1D(x_data[i-WINDOW_SIZE:i])
                else : x_ = x_data[i-WINDOW_SIZE:i]
                x_array.append(x_)
            self.x_data = torch.FloatTensor(np.array(x_array)) # torch.Size([4496, 40, 30, 6])

        self.y_data = torch.FloatTensor(np.array(y_data[WINDOW_SIZE:])) # torch.Size([4496, 30]) 
        assert len(self.x_data)==len(self.y_data), "number of x_data and y_data set doesnt match..."
        print("EMD Dataset Done.")


    def __getitem__(self, index):
        # help indexing: 0번째는 x 데이터, 1번째는 y
        # dictionary 형태로 바꾸면 유지보수 편함
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        # length func
        return self.y_data.shape[0]
    
    def input_size(self):
        # input_size 
        return self.x_data.shape






class CriticDataset(Dataset):
    '''
    For Custom dataset split, prepare data as torch Dataset.
    Args:
    - x_data(pd.DataFrame)
    - y_data(pd.Series)

    Out: Dataset 객체 
    -  iter(Dataset객체) 한 후 for문 돌려서 나오는 각 값은
    - Out[0] : x_data (input feature)
    - Out[1] : y_data (label)
    '''

    def __init__(self, x_data:np.array, y_data:np.array, action_data:np.array,
                ACTION_SPACE_DIMENSION, WINDOW_SIZE):
        '''
        수정사항 : 바로 데이터 프레임으로 받으면 안되나?
        '''
        super(CriticDataset, self).__init__()
        # 40개씩 묶어서 처리하기
        x_array =  []

        # input 크기에 따라
        if x_data.shape[0] == ACTION_SPACE_DIMENSION:
            # x_data.shape가 asset, Time, features 일때...
            for i in range(WINDOW_SIZE, len(x_data)):
                x_array.append(x_data[:,i-WINDOW_SIZE:i,:])
            self.x_data = torch.FloatTensor(np.array(x_array)) # torch.Size([4496, 30, 40, 6])

        else : 
            # data shape : (Time, asset, features)
            for i in range(WINDOW_SIZE, len(x_data)):
                x_array.append(x_data[i-WINDOW_SIZE:i])
            self.x_data = torch.FloatTensor(np.array(x_array)) # torch.Size([4496, 40, 30, 6])

        # action이 문젠데
        import ipdb
        ipdb.set_trace()
        self.y_data = torch.FloatTensor(y_data[WINDOW_SIZE:]) # torch.Size([4496, 30])
        self.action_data = torch.FloatTensor(action_data[WINDOW_SIZE:]) # toch.Size([###, 30])



    def __getitem__(self, index):
        # help indexing: 0번째는 x 데이터, 1번째는 y
        return {'state':self.x_data[index], 'action':self.action_data[index]}, self.y_data[index]
    
    def __len__(self):
        # length func
        return self.y_data.shape[0]
    
    def input_size(self):
        # input_size 
        return self.x_data.shape








if __name__ =='main':
    import pandas as pd
    


