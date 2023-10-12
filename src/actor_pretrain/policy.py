import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Union


# from model.networks.net import Network
# from src.networks.tcn import TCN
# from src.networks.attention import *
from actor_pretrain.network import *




# 그냥 nn.Modules로 상속받고 학습시켜도 될듯..
class SimpleGRU(Network):
    def __init__(self,
                lr_pi: float,
                input_shape:Tuple,
                layer_neurons:int=4, 
                *args,
                **kwargs,
                ) -> None:

        super(SimpleGRU, self).__init__(*args, **kwargs)
        # 공통 세팅
        # self.input_shape : 
        self.input_shape = input_shape
        self.window = self.input_shape[0]
        self.n_feature = self.input_shape[-1]
        if len(self.input_shape)==3:
            self.asset_num = self.input_shape[1] #(40,30,7)

        self.lr_pi = lr_pi
        self.layer_neurons = layer_neurons

        # Layers used in model
        self.model = StackedGRU(input_shape=self.input_shape,
                                hidden_dim=layer_neurons, ###
                                out_dim=1
                                )

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.model(x)




# 그냥 nn.Modules로 상속받고 학습시켜도 될듯..
class SimpleLSTM(Network):
    def __init__(self,
                lr_pi: float,
                input_shape:Tuple,
                layer_neurons:int = 8, 
                *args,
                **kwargs,
                ) -> None:

        super(SimpleLSTM, self).__init__(*args, **kwargs)
        
        
        # 공통 세팅
        self.input_shape = input_shape
        self.window = self.input_shape[0]
        self.n_feature = self.input_shape[-1]
        if len(self.input_shape)==3:
            self.asset_num = self.input_shape[1] #(40,30,7)

        self.lr_pi = lr_pi
        self.layer_neurons = layer_neurons #8

        # Layers used in model
        self.model = StackedLSTM2(input_shape=self.input_shape,
                                hidden_dim=layer_neurons, ###
                                out_dim=1
                                )
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()


    def forward(self, x):
        return self.model(x)





class LSTMA_Actor(Network):
    def __init__(self,                 
                lr_pi: float,
                max_actions: int, # 안씀
                action_space_dimension: Tuple,
                log_sigma_min: float = -20.0,
                log_sigma_max: float= 2.0,
                window: int = 40,
                *args,
                **kwargs,) -> None:
        super(LSTMA_Actor, self).__init__(*args, **kwargs)

        # self.input_shape = obs_space_dimension
        self.max_actions = max_actions
        self.n_feature = self.input_shape[2] #(40,30,7)
        self.asset_num = self.input_shape[1]
        self.action_space_dimension = action_space_dimension
        self.window = window

        # layers of the main model
        self.att_3d = Attention3D(timesteps=self.window, attention_dim=32)
        self.LSTM = StackedLSTM(input_shape=self.input_shape,
                                hidden_dim=[32,64,32], ###
                                # out_dim=self.asset_num,
                                out_dim=self.layer_neurons)
        
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        ## for sampling function of SAC Actor
        self.mu = nn.Linear(self.layer_neurons, self.action_space_dimension) # 1,21을 원하는 중..mu :  torch.Size([1, 21, 22])
        self.log_sigma = nn.Linear(self.layer_neurons, self.action_space_dimension)

        self.layer_mu = nn.Linear(self.action_space_dimension*self.window, self.action_space_dimension)
        self.layer_sig = nn.Linear(self.action_space_dimension*self.window, self.action_space_dimension)
        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi)
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        

    def forward(self, state):
        # x(state) shape :(batch, time, asset, features) --> [32, 40, 30, 1?]
        batch = state.shape[0]
        timestep, input_dim = state.shape[1], state.shape[2]*state.shape[3] 
        state = state.contiguous()
        att_x = self.att_3d(state)
        lstm_x = self.LSTM(att_x.view(batch, timestep, -1))

        # mu
        mu = self.mu(lstm_x) # batch, window, featureㅜㅠ 
        mu = mu.view(mu.shape[0], -1)
        mu = self.layer_mu(mu) # 여기서 nan인 행들이 생기네
        
        # sig 
        log_sigma = self.log_sigma(lstm_x)
        log_sigma = torch.flatten(log_sigma,1)
        log_sigma = self.layer_sig(log_sigma)
        log_sigma = torch.clamp(log_sigma, min = self.log_sigma_min, max=self.log_sigma_max)
        sigma = (log_sigma).exp()

        return mu, sigma


    def sample(self, 
                state: np.ndarray,
                reparameterize: bool = True,) -> Tuple[torch.tensor]:
        '''
        Sample from the Normal Dist, output of feedforward method, to give an action
        
        Args:
            state (np.array): state of the environment in which the actor has to pick an action
            reparameterize (bool): whether one should sample using the reparemeterization trick or not
        
        Returns:
            action sampled from Normal disribution, as well as log probability of the given action        
        '''

        ## distributional RL 구성방법이로구나
        mu, sigma = self.forward(state)
        normal = torch.distributions.Normal(mu, sigma) ####

        if reparameterize:
            actions = normal.rsample()
        else:
            actions = normal.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_actions).to(self.device)

        log_probabilities = normal.log_prob(actions)
        log_probabilities -= torch.log(1 - action.pow(2) + self.noise + 1e-6) 
        
        log_probabilities = log_probabilities.sum(1, keepdim=True) ##
        # action = nn.Softmax(action)

        return action, log_probabilities



#------------------------------------------------------------------------

