import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List



class StackedGRU(nn.Module):
    def __init__(self, 
                input_shape:tuple, 
                hidden_dim:Union[int, List] = [32,64,16], 
                pretrain:bool = True,
                out_dim = 1,
                ):
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
                    dropout = 0.5, #0.25
                    batch_first=True)
        
        self.lstm2 = nn.GRU(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.5,
                        batch_first=True) 
        
        self.lstm3 = nn.GRU(input_size = 1, 
                        hidden_size = self.hidden1, # self.input_dim, #self.hidden2, 
                        num_layers = 1, 
                        dropout = 0.5,
                        batch_first=True)
        # self.fc1 = nn.Linear(self.input_dim, out_dim)
        self.fc1 = nn.Linear(self.hidden1*3, self.hidden1*3//2)
        self.fc2 = nn.Linear(self.hidden1*3//2, 1)
        # self.fc3 = nn.Linear(self.hidden1, 1)
        self.batchnorm = nn.BatchNorm1d(num_features=self.window) ##
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        if x.ndim != 3:
            x = x.unsqueeze(dim=0) # 1개씩 들어오는 경우, 배치를 1로 만들어준다. (1, window, imfs)
        x_ = self.batchnorm(x) #batchnorm의 채널은 두번째 차원? #####
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
        return out.squeeze() # 64,1 (Batch, 1)
    

    def init_weights(self):
        # initialize the weights 
        for layer_n in self.state_dict():
            param = self.state_dict()[layer_n]
            if 'weight_ih' in layer_n:
                torch.nn.init.orthogonal_(param)
            elif 'weight_hh' in layer_n:
                weight_hh_data_ii = torch.eye(self.hidden1,self.hidden1) #H_W_reset
                weight_hh_data_if = torch.eye(self.hidden1,self.hidden1)#H_W_update
                weight_hh_data_ic = torch.eye(self.hidden1,self.hidden1)#H_W_out
                # weight_hh_data_io = torch.eye(self.hidden1,self.hidden1)#H_Wio
                weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic], dim=0)
                weight_hh_data = weight_hh_data.view(self.hidden1*3,self.hidden1)
                param.data.copy_(weight_hh_data)
            elif 'bias' in layer_n:
                torch.nn.init.constant_(param, val=0)
                param.data[self.hidden1:self.hidden1*2].fill_(1)
            elif ('fc' or 'weight') in layer_n:
                torch.nn.init.xavier_uniform_(param)