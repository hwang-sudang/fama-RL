
import numpy as np
import os
import torch
import torch.nn as nn
from typing import Tuple
from model.networks.net import Network, DynamicEncoder # 왜 안되노

class MLPCritic2(Network):
    '''
    크리틱 네트워크 정의, state, action 쌍의 Q 밸류 결과 추정
    '''
    def __init__(self, 
                lr_Q: float, 
                action_space_dimension: Tuple, 
                initializer='xavier',
                *args,
                **kwargs, ) -> None:
        """
        Args:
            lr_Q (float): learning rate for the gradient descent 
            action_space_dimension (Tuple): dimension of the action space
        Returns:
            no value        
        """
        super(MLPCritic2, self).__init__(*args, **kwargs)
        self.feature_dim = self.input_shape[-1]+1

        self.window = self.input_shape[0]
        self.action_space_dimension = action_space_dimension #30
        # self.actionlayer = torch.nn.Linear(action_space_dimension, 16).to(self.device, dtype=torch.float32)
        self.statelayer =  torch.nn.Linear(self.feature_dim, 1).to(self.device, dtype=torch.float32)
        self.Q = torch.nn.Sequential(torch.nn.Linear(2*action_space_dimension, 24),
                                     torch.nn.Linear(24, 1),
                                     ).to(self.device, dtype=torch.float32)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q, ) #weight_decay=1e-6
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # self.lrscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)


    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                action: torch.tensor,
                ) -> torch.tensor:
        """
        Implement the feedforward of the net.

        Args:
            state, prestate (torch.tensor): input state 
            action (torch.tensor): input action
            
        Returns:
            value attributed to the (state, action) pair        
        """
        state = torch.cat((state, prestate.unsqueeze(dim=-1)), dim=-1) # (B, 30, 9)
        state = self.statelayer(state).squeeze() # for mlp  #32,30,1
        # action = self.actionlayer(action) # 32,30
        x = torch.cat((self.sigmoid(state), action), axis=-1)
        q = self.Q(x) #(batch, 1)
        return q







class MLPCritic3(Network):
    '''
    크리틱 네트워크 정의, state, action 쌍의 Q 밸류 결과 추정
    '''
    def __init__(self, 
                lr_Q: float, 
                action_space_dimension: Tuple, 
                initializer='xavier',
                *args,
                **kwargs, ) -> None:
        """
        Args:
            lr_Q (float): learning rate for the gradient descent 
            action_space_dimension (Tuple): dimension of the action space
        Returns:
            no value        
        """
        super(MLPCritic3, self).__init__(*args, **kwargs)
        self.feature_dim = self.input_shape[-1]
        self.window = self.input_shape[0]
        self.stock_space_dimension = self.input_shape[1]
        self.action_space_dimension = action_space_dimension #30
        self.input_dim = self.action_space_dimension * self.window


        # Embedding layer
        self.ohlcv_feat = nn.Sequential(
                            nn.LayerNorm(5*self.stock_space_dimension),
                            nn.GRU(5*self.stock_space_dimension, self.action_space_dimension, batch_first=True),
                            )
        self.factor_feat = nn.Sequential(nn.Linear(self.feature_dim-5, (self.feature_dim-5)//2), 
                                         nn.Linear((self.feature_dim-5)//2, 1),
                                         nn.Flatten(start_dim=1),
                                         nn.Linear(self.stock_space_dimension, self.action_space_dimension)
                                         )
        self.prestate_feat = nn.Linear(self.stock_space_dimension, self.action_space_dimension)
        self.state_enc = DynamicEncoder(input_shape = self.action_space_dimension*2)
        self.state_fc = nn.Linear(self.action_space_dimension*2, self.action_space_dimension)
        self.Q = nn.Sequential(
                                nn.LayerNorm(self.action_space_dimension),
                                nn.Linear(self.action_space_dimension, 1),
                                nn.ReLU()
                                ).to(self.device, dtype=torch.float32)
        
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q, weight_decay=1e-6) 
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        # self.lrscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)



    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                action: torch.tensor,
                ) -> torch.tensor:
        """
        Implement the feedforward of the net.

        Args:
            state, prestate (torch.tensor): input state 
            action (torch.tensor): input action
            
        Returns:
            value attributed to the (state, action) pair        
        """
        ohlcv = state[:,:,:,:5]
        factor = state[:,-1,:,5:]
        
        # feature extraction of states
        ohlcv, _ = self.ohlcv_feat(torch.flatten(ohlcv, start_dim=-2)) # (B,1,A)
        ohlcv = self.relu(ohlcv)
        factor = self.relu(self.factor_feat(factor)) # (B,A,1)
        prestate = self.relu(self.prestate_feat(prestate))
        #######

        # compute the attention(?) two state features
        state, state_loss = self.state_enc(torch.hstack([ohlcv[:, -1], factor.squeeze(dim=-1)])) # (B, A, 2) -> (B, A, 2)
        state = self.state_fc(state)
        weighted_a = torch.mul(prestate, action) # always (Batch, Asset+1)
        x = weighted_a * state  #(B, A) * (B, A) -> (B, A)

        q = self.Q(x) #(batch, 1) ## 여기가 고민이긴 하네 # ㅋㅋㅋㅋ 개웃기네 state는 쓰지도 않아 ㅋㅋㅋㅋㅋ
        self.loss_term = state_loss
        self.critic_attention = self.state_enc.wm.detach().cpu().numpy() #(B, asset*2)
        return q





class TwinnedQNet(Network):
    def __init__(self,                
                 lr_Q: float, 
                 action_space_dimension: Tuple,
                 initializer='xavier',
                 *args, 
                 **kwargs):
        super(TwinnedQNet, self).__init__(*args, **kwargs)

        self.Q1 = MLPCritic3(lr_Q=lr_Q,
                            action_space_dimension=action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_neurons,
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device,
                            network_name='Q1').to(self.device)
        
        self.Q2 = MLPCritic3(lr_Q=lr_Q,
                            action_space_dimension=action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_neurons,
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device,                            
                            network_name='Q2').to(self.device)
        
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q, weight_decay=1e-6)
    
    def forward(self, state, pre_state, action):
        q1 = self.Q1(state, pre_state, action)
        q2 = self.Q2(state, pre_state, action)
        self.loss_term = (q1.loss_term+q2.loss_term)/2
        return q1, q2





#-----------------------------------------------------------------------------------------------------------------------

class ManagerCritic(Network):
    '''
    크리틱 네트워크 정의, state, action 쌍의 Q 밸류 결과 추정
    '''
    
    def __init__(self, 
                lr_Q: float, 
                action_space_dimension: Tuple, 
                *args,
                **kwargs, ) -> None:

        """
        Args:
            lr_Q (float): learning rate for the gradient descent 
            action_space_dimension (Tuple): dimension of the action space
            
        Returns:
            no value        
        """
        super(ManagerCritic, self).__init__(*args, **kwargs)
        # import ipdb
        # ipdb.set_trace()
        if len(self.input_shape) == 3:
            # input 3차원이라면
            self.feature_dim = self.input_shape[1]*self.input_shape[2]
        else :
            self.feature_dim = self.input_shape[2]

        self.market_dim = 3 # feature 수
        self.window = self.input_shape[0]


        self.action_space_dimension = action_space_dimension #30
        self.actionlayer = torch.nn.Linear(action_space_dimension, self.layer_neurons).to(self.device, dtype=torch.float32)
        self.stocklayer = torch.nn.Sequential(
                            torch.nn.Linear(self.feature_dim, 128).to(self.device, dtype=torch.float32),
                            torch.nn.ReLU(inplace=True),
                            torch.nn.Linear(128, 1).to(self.device, dtype=torch.float32),
                            )
        self.marketlayer = torch.nn.Sequential(
                            torch.nn.Linear(self.market_dim, 1).to(self.device, dtype=torch.float32),
                            torch.nn.ReLU(inplace=True),
                            # torch.nn.Linear(128, 1).to(self.device, dtype=torch.float32),
                            )
        self.layer1 = torch.nn.Linear(self.layer_neurons + self.window,  self.layer_neurons).to(self.device, dtype=torch.float32)
        self.layer2 = torch.nn.Linear(self.layer_neurons, self.layer_neurons).to(self.device, dtype=torch.float32)
        self.Q = torch.nn.Linear(self.layer_neurons, 1).to(self.device, dtype=torch.float32)
        self.relu = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_Q)
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')



    def forward(self,
                stock_obs, market_obs,
                action: np.ndarray,
                ) -> torch.tensor:
        """
        Implement the feedforward of the net.

        Args:
            state (np.array): input state 
            action (np.array): input action
            
        Returns:
            value attributed to the (state, action) pair        
        """

        # state 인풋에 따라서 + self.window
        if len(state.shape) != 4:
            # 3차원이 아닐 때에
            stock_obs = stock_obs.view(stock_obs.shape[0], -1)
            # state = self.statelayer(state)
        else:
            stock_obs = self.stocklayer(stock_obs.view(stock_obs.shape[0], -1, self.feature_dim)) # for mlp
            market_obs = self.marketlayer(market_obs) # check the dimension of market_obs
            state = stock_obs * market_obs
            state = state.squeeze()
        
        state = self.relu(state)
        action = self.actionlayer(action) # 32,22
        x = self.layer1(torch.cat([state, action], dim=1))

        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.sigmoid(x)
        action_value = self.Q(x)
        x = torch.nn.functional.relu(x)

        return action_value
