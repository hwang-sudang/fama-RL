import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from PyEMD import EMD, EEMD, CEEMDAN 

'''
- SNP는 dropout 50이 제일 나았음.
'''


class Decomposition():
    @staticmethod
    def emd_data_transform1D(data, mode:str='emd') -> List[np.ndarray]:
        # mode에 따라서 데이터를 다르게hidden1 분해한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()hidden1
        
        if len(data.shape) == 2: data = np.squeeze(data, axis=-1)
        emd = EMD()
        IMFs = emd(data)
        data_res = data-IMFs[0]-IMFs[1]
        data_mf = np.stack((IMFs[0], IMFs[1], data_res), axis=-1)
        return data_mf  # 차원 확인

    @staticmethod
    def emd_data_transform2D(data, mode:str='emd') -> List[np.ndarray]:
        # mode에 따라서 데이터를 다르게hidden1 분해한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()
        assert len(data.shape)==2 , "Your data dimension is %d. Use another function." % (data.ndim)
        ## 수정 필요
        time_steps, data_dims = data.shape
        
        data_mf1 = []
        data_mf2 = []
        data_res = []

        for j in range(data_dims):
            S = np.ravel(data[:, j])
            emd = EMD()
            IMFs = emd(S)
            data_mf1.append(IMFs[0].tolist())
            data_mf2.append(IMFs[1].tolist())
            data_res.append((S-(IMFs[0]+IMFs[1])).tolist())

        data_mf1 = np.array(data_mf1).transpose([1, 0]) #(batch, timeseries, asset)
        data_mf2 = np.array(data_mf2).transpose([1, 0]) #(batch, timeseries, asset)
        data_res = np.array(data_res).transpose([1, 0]) 
        data_mf = np.stack((data_mf1, data_mf2, data_res), axis=-1)
        return data_mf  # 차원 확인

    @staticmethod
    def emd_data_transform_per_batch(data, mode:str='emd') -> List[np.ndarray]:
        # 3차원 이상부터 써도 될듯
        # mode에 따라서 데이터를 다르게 미분한다.
        # if mode == 'emd': decom = EMD()
        # elif mode == 'eemd' : decom = EEMD()
        # else : decom = CEEMDAN()
        
        #무조건 3차원 인풋으로와야함
        samples, time_steps, data_dims = data.shape
        data_mf1 = []
        data_mf2 = []
        for i in range(samples):
            sample_1 = []
            sample_2 = []
            for j in range(data_dims):
                S = np.ravel(data[i, :, j])
                emd = EMD()
                IMFs = emd(S)
                sample_1.append(IMFs[0].tolist())
                sample_2.append((S - IMFs[0]).tolist())
            data_mf1.append(sample_1)
            data_mf2.append(sample_2)
        data_mf1 = np.array(data_mf1).transpose([0, 2, 1]) #(batch, timeseries, asset)
        data_mf2 = np.array(data_mf2).transpose([0, 2, 1]) #(batch, timeseries, asset)
        
        data_mf = np.stack((data_mf1, data_mf2), axis=-1)
        return data_mf  # 차원 확인


class Attention3D(nn.Module):
    def __init__(self, timesteps, attention_dim=1):
        super(Attention3D, self).__inhidden1it__()
        # attention_dim 은 timesteps가 되어야함
        # 4D version도 생각해보기! --> (B, T, A, I)에서 T 축으로 통과
        self.timesteps = timesteps
        self.attention = nn.Sequential(nn.Linear(self.timesteps, self.timesteps),
                                       nn.Softmax())
        # self.attention = nn.Sequential(nn.Linear(self.timesteps, attention_dim),
        #                                nn.Linear(attention_dim, self.timesteps),
        #                                nn.Softmax())
    
    def forward(self, input):
        # input data dim : (batch, time, asset, features)
        if input.dim() == 4:
            # change dimension to 2d (except for the batch side)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
        x = torch.permute(input, (0,2,1)) # (B, A*I, T)
        x = self.attention(x) #B, T, ahidden1ttention dim
        x_probs = torch.permute(x, (0,2,1)) 
        output_attention_mul = torch.mul(input, x_probs)
        
        return output_attention_mul #(B, Time, Asset*Inputs)
    


class Attention4D(nn.Module):
    def __init__(self, timesteps, attention_dim=1):
        super(Attention4D, self).__init__()
        # attention_dim 은 timesteps가 되어야함
        # 4D version도 생각해보기! --> (B, T, A, I)에서 T 축으로 통과
        self.timesteps = timesteps
        self.attention = nn.Sequential(nn.Linear(self.timesteps, self.timesteps),
                                       nn.Softmax())
        # self.attention = nn.Sequential(nn.Linear(self.timesteps, attention_dim),
        #                                nn.Linear(attention_dim, self.timesteps),
        #                                nn.Softmax())
    
    def forward(self, input):
        # input data dim : (batch, time, asset, features)
        assert input.dim()==4, "input dimension should be 4: (batch, time, asset, features)"
        x = torch.permute(input, (0,2,3,1)) # (B, T, A, I) -> (B, A, I, T)
        x = self.attention(x) #B, T, attention dim
        x_probs = torch.permute(x, (0,3,1,2)) # (B, A, I, T) -> (B, T, A, I)
        output_attention_mul = torch.mul(input, x_probs) #(B, Time, Asset, Inputs)
        output_attention_mul = output_attention_mul.view(input.shape[0], input.shape[1], -1)
        return output_attention_mul #(B, Time, Asset*Inputs) for LSTM

            
class Network(nn.Module):
    '''
    Abstract class to be inherited by the various critic and actor classes.
    '''
    def __init__(self,
                network_name: str,
                checkpoint_directory_networks: str,
                device: str = 'cpu',
                ) -> None :
        """
        Args:
            input_shape (tuple) : state space의 차원 수 
            layer_neurons (int) : 네트워크 레이어 안의 뉴런 수 ###
            name (str): 네트워크 이름
            checkpoint_directory_networks (str = 'saved_networks'): base directory for the checkpoints
        
        Returns:
            no value
        """
        super(Network, self).__init__()
        self.network_name = network_name
        self.checkpoint_directory_networks = checkpoint_directory_networks
        self.checkpoint_directory_pretrain = checkpoint_directory_networks+f"/pretrain"
        self.checkpoint_file_network = os.path.join(self.checkpoint_directory_networks, f'{self.network_name}.pth')
        self.pretrain_net_dir = checkpoint_directory_networks+f"/pretrain/price_{network_name}/BestModel_price_{network_name}.pth"
        self.device = device

        # if torch.cuda.device_count() > 1:
        #     # gpu 병렬처리 해라
        #     self = torch.nn.DataParallel(self)
        # self.to(self.device)


    def forward(self, 
                *args, 
                **kwargs, ) -> torch.tensor:
        raise NotImplementedError


    def save_network_weights(self) -> None:
        '''Save checkpoint, used in training mode'''
        torch.save(self.state_dict(), self.checkpoint_file_network) ##


    def load_network_weights(self) -> None:
        '''Save checkpoint, used in test mode'''
        self.load_state_dict(torch.load(self.checkpoint_file_network, map_location=self.device))




class StackedLSTM2(nn.Module):
    def __init__(self, 
                input_shape:tuple, 
                hidden_dim:Union[int, list] = [32,64,16], 
                out_dim = 1):
        super(StackedLSTM2, self).__init__()
        # timelength, data_dim1, data_dim2, hidden_dim : Union[list, int] 
        
        self.input_dim = input_shape[-1] if len(input_shape) == 2 else input_shape[-1]*input_shape[-2]
        self.window = input_shape[0]

        if isinstance(hidden_dim, list):
            assert len(hidden_dim) == 3, 'The hidden dim length should be 3 but you have [{0}]'.format(len(hidden_dim))
            self.hidden1 = hidden_dim[0]
            self.hidden2 = hidden_dim[1]
            self.hidden3 = hidden_dim[2]
        else :
            self.hidden1 = self.hidden2 = self.hidden3 = hidden_dim

        # Model layer define
        self.lstm1 = nn.LSTM(input_size=1, 
                    hidden_size = self.hidden1, #self.input_dim, 
                    num_layers = 1, 
                    dropout = 0.4, #0.25
                    batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.4,
                        batch_first=True) 
        
        self.lstm3 = nn.LSTM(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.4,
                        batch_first=True)
        
        # self.fc1 = nn.Linear(self.input_dim, out_dim)
        self.fc1 = nn.Linear(self.hidden1*3, self.hidden1*3//2)
        self.fc2 = nn.Linear(self.hidden1*3//2, 1)
        # self.fc3 = nn.Linear(self.hidden1, 1)
        self.batchnorm = nn.BatchNorm1d(num_features=self.window)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_ = self.batchnorm(x)

        # 시계열 분리
        imf1, imf2, imf3 = x_[:,:,0], x_[:,:,1], x_[:,:,-1]

        imf1, _ = self.lstm1(imf1.unsqueeze(-1))
        imf1 = self.relu(imf1[:, -1, :])        # out = self.tanh(x_)
        # print("self.fc1(x_) & tanh: ",x_.sum())
        imf3, _ = self.lstm3(imf3.unsqueeze(-1))
        imf3 = self.relu(imf3[:, -1, :])

        # 각 시계열 분해요소 concat
        imfs = torch.cat((imf1, imf2, imf3), -1) # shape 확인
        out = self.fc1(imfs)
        out = self.fc2(self.tanh(out))
        return out # 64,1





class StackedGRU(nn.Module):
    def __init__(self, 
                input_shape:tuple, 
                hidden_dim:Union[int, list] = [32,64,16], 
                out_dim = 1):
        super(StackedGRU, self).__init__()
        # timelength, data_dim1, data_dim2, hidden_dim : Union[list, int] 
        
        self.input_dim = input_shape[-1] if len(input_shape) == 2 else input_shape[-1]*input_shape[-2]
        self.window = input_shape[0]

        if isinstance(hidden_dim, list):
            assert len(hidden_dim) == 3, 'The hidden dim length should be 3 but you have [{0}]'.format(len(hidden_dim))
            self.hidden1 = hidden_dim[0]
            self.hidden2 = hidden_dim[1]
            self.hidden3 = hidden_dim[2]
        else :
            self.hidden1 = self.hidden2 = self.hidden3 = hidden_dim

        # Model layer define
        self.lstm1 = nn.GRU(input_size=1, 
                    hidden_size = self.hidden1, #self.input_dim, 
                    num_layers = 1, 
                    dropout = 0.25, #0.5
                    batch_first=True)
        
        self.lstm2 = nn.GRU(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.25,
                        batch_first=True) 
        
        self.lstm3 = nn.GRU(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.25,
                        batch_first=True)
        
        # self.fc1 = nn.Linear(self.input_dim, out_dim)
        self.fc1 = nn.Linear(self.hidden1*3, self.hidden1*3//2)
        self.fc2 = nn.Linear(self.hidden1*3//2, 1)
        # self.fc3 = nn.Linear(self.hidden1, 1)
        self.batchnorm = nn.BatchNorm1d(num_features=self.window)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_ = self.batchnorm(x)

        # 시계열 분리
        imf1, imf2, imf3 = x_[:,:,0], x_[:,:,1], x_[:,:,-1]

        imf1, _ = self.lstm1(imf1.unsqueeze(-1))
        imf1 = self.relu(imf1[:, -1, :])
        imf2, _ = self.lstm2(imf2.unsqueeze(-1))
        imf2 = self.relu(imf2[:, -1, :])
        imf3, _ = self.lstm3(imf3.unsqueeze(-1))
        imf3 = self.relu(imf3[:, -1, :])

        # 각 시계열 분해요소 concat
        imfs = torch.cat((imf1, imf2, imf3), -1) # shape 확인
        out = self.fc1(imfs)
        out = self.fc2(self.tanh(out))
        return out.squeeze() # 64,1
    



class StackedLSTM(nn.Module):
    def __init__(self, 
                input_shape:tuple, 
                hidden_dim:Union[int, list] = [32,64,16], 
                out_dim = 1):
        super(StackedLSTM, self).__init__()
        # timelength, data_dim1, data_dim2, hidden_dim : Union[list, int] 
        
        self.input_dim = input_shape[-1] if len(input_shape) == 2 else input_shape[-1]*input_shape[-2]
        self.window = input_shape[0]

        if isinstance(hidden_dim, list):
            assert len(hidden_dim) == 3, 'The hidden dim length should be 3 but you have [{0}]'.format(len(hidden_dim))
            self.hidden1 = hidden_dim[0]
            self.hidden2 = hidden_dim[1]
            self.hidden3 = hidden_dim[2]
        else :
            self.hidden1 = self.hidden2 = self.hidden3 = hidden_dim

        # Model layer define
        self.lstm1 = nn.LSTM(input_size=self.input_dim, 
                    hidden_size = self.hidden1, 
                    num_layers = 1, 
                    dropout = 0.25, #0.25
                    batch_first=True)
        
        self.lstm2 = nn.LSTM(input_size = self.hidden1, 
                        hidden_size = self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.1,
                        batch_first=True)
        
        self.fc1 = nn.Linear(self.input_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(num_features=self.window)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # out = self.model(x)
        x_ = self.batchnorm(x)
        x_1, _ = self.lstm1(x_)
        x_2,_ = self.lstm2(x_1)

        out = self.fc1(x_2)
        # out = self.tanh(x_)
        # print("self.fc1(x_) & tanh: ",x_.sum())
        return out




    

