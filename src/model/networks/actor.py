import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# inner module import
from model.networks.net import *
# from env.environment import PortfolioEnv
torch.autograd.set_detect_anomaly(True)


class DDPGActor(Network):
    def __init__(self,
                lr_pi: float,
                max_actions: np.ndarray,
                action_space_dimension: Tuple,
                # log_sigma_min: float = -20.0,
                # log_sigma_max: float= 2.0,
                window: int = 40,
                *args,
                **kwargs,) -> None:
        super(DDPGActor, self).__init__(*args, **kwargs)
        '''
        항상 배치를 포함한 차원으로 넣어줄 것. (1,30,9)
        '''
        # 공통 세팅
        self.action_space_dimension = action_space_dimension
        self.stock_space_dimension = self.input_shape[-2]
        self.max_actions = torch.tensor(max_actions).to(self.device)
        self.noise = 1e-6
        self.window = window
        self.input_size = self.input_shape[-1] 
        self.lr_pi = lr_pi

        # forward 모델 구성 layers
        # self._input_shape = self.input_size+1
        ## feature extractions
        self.feat_layer = Actor7(
                                action_space_dimension= self.action_space_dimension,
                                input_shape=self.input_shape,
                                layer_neurons=self.layer_neurons,
                                )
        
        # set optimizer for networks
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi) # RMSprop
        # self.lrscheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                # state_factor : Union(np.ndarray, torch.tensor),
                ) -> List[torch.tensor]:
        '''
        Input
        - state : Stock OHLCV timeseries (Batch, Timeseries, asset, input_size) 
        - state_factor : Stock Technical Indicators indicating Factors at t-1 (Batch, 1, asset, TI)
        Output
        - mu : action
        '''
        mu, _ = self.feat_layer(state, prestate)
        self.attention = self.feat_layer.attention
        return F.softmax(mu, dim=-1)





class SACActor(Network):
    def __init__(self,
                lr_pi: float,
                max_actions: np.ndarray,
                action_space_dimension: Tuple,
                log_sigma_min: float = -20.0,
                log_sigma_max: float= 2.0,
                window: int = 40,
                *args,
                **kwargs,) -> None:
        super(SACActor, self).__init__(*args, **kwargs)
        '''
        항상 배치를 포함한 차원으로 넣어줄 것. (1,30,9)
        '''
        # 공통 세팅
        self.action_space_dimension = action_space_dimension
        self.stock_space_dimension = action_space_dimension
        self.max_actions = torch.tensor(max_actions).to(self.device)
        self.noise = 1e-6
        self.window = window
        self.input_size = self.input_shape[-1]
        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min

        # forward 모델 구성 layers
        ## feature extractions
        self.feat_layer = Actor7(
                                action_space_dimension= self.action_space_dimension,
                                input_shape=self.input_shape,
                                layer_neurons=self.layer_neurons,
                                )
        # for sample() 구성 
        self.layer_pfratio = nn.Sequential(nn.Linear(self.action_space_dimension, self.action_space_dimension),
                                            nn.Softmax(dim=-1))
        # set optimizer for networks
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_pi) # RMSprop  weight_decay=1e-6
        # self.lrscheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        self.lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')


    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                # state_factor : Union(np.ndarray, torch.tensor),
                ) -> List[torch.tensor]:
        '''
        Input
        - state : Stock OHLCV timeseries (Batch, Timeseries, asset, input_size) 
        - state_factor : Stock Technical Indicators indicating Factors at t-1 (Batch, 1, asset, TI)
        Output
        Expectation and standard deviation of a Normal distribution
        - mu
        - sigma
        '''
        # mu and sigma calculation for SAC action sampling
        # 계산 과정에서 nan 나오는 이유?
        assert not torch.isnan(state).any()
        assert not torch.isnan(prestate).any()
                       
        mu, log_sigma = self.feat_layer(state, prestate)
        log_sigma = torch.clamp(log_sigma, min = self.log_sigma_min, max=self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        self.attention = self.feat_layer.attention
        # print('mu.shape: ', mu.shape, 'log_sigma.shape: ', sigma.shape)
        return mu, sigma
    

    def sample(self, 
                state: torch.tensor,
                prestate: torch.tensor,
                reparameterize: bool = True,) -> Tuple[torch.tensor]:
        '''
        Sample from the Normal Dist, output of feedforward method, to give an action
        
        Args:
            state (np.array): state of the environment in which the actor has to pick an action
            reparameterize (bool): whether one should sample using the reparemeterization trick or not
        
        Returns:
            action sampled from Normal disribution, as well as log probability of the given action        
        '''
        # Make Normal distributions for SAC sampling
        state = torch.nan_to_num(state, nan=0.0, neginf=-3, posinf=3).to(self.device)
        mu, sigma = self.forward(state, prestate)
        normal = torch.distributions.Normal(mu, sigma) ####

        # SAC sample reparameterize
        if reparameterize: actions = torch.tanh(normal.rsample())
        else: actions = torch.tanh(normal.sample())

        # Portfolio action and logprobability for SAC entropy compute
        action_ratio = self.layer_pfratio(actions)
        log_probabilities = normal.log_prob(actions)
        log_probabilities -= torch.log(1 - action_ratio.pow(2) + self.noise) 
        log_probabilities = log_probabilities.sum(1, keepdim=True) ##entropy
        log_probabilities = F.tanh(log_probabilities) ###
        log_probabilities  = torch.clamp(log_probabilities, min = self.log_sigma_min, max=self.log_sigma_max)
        

        '''
        # v2. hanojaa
        # /home/ubuntu2010/바탕화면/DEV/trading14/job_.sh
        action_ratio = self.layer_pfratio(actions)
        log_probabilities = normal.log_prob(actions)
        squash = torch.log(1 - action_ratio.pow(2) + self.noise) #squash : adjusting the action bound
        squash = squash.sum(-1, keepdim=True) # 주석달아서도 해보고 아니고도 해보고
        log_probabilities -= squash 
        log_probabilities.sum(-1, keepdim=True) # 
        log_probabilities = F.tanh(log_probabilities)
        log_probabilities  = torch.clamp(log_probabilities, min = self.log_sigma_min, max=self.log_sigma_max) # it's getting bigger..
        '''

        return action_ratio, log_probabilities





#------------------------------------- Actor Networks -------------------------------------------


class Actor7(nn.Module):
    '''
    성수오빠 의견 반영

    '''
    def __init__(self,
            input_shape: Tuple,
            action_space_dimension: int,
            layer_neurons:int,
            window: int = 40,
            network_name:str = "Actor6"
            ) -> None:
        super(Actor7, self).__init__()
        '''
        항상 배치를 포함한 차원으로 넣어줄 것. (1,30,9)
        
        # input size
        state = (batch, assets, 5+factors)
        prestate = (batch, assets)

        ohlcv = (batch, assets, 5)
        factors = (batch, assets, factors(3))
        prestate = (batch, assets)

        '''
        # 공통 세팅
        self.action_space_dimension = action_space_dimension
        self.stock_space_dimension = input_shape[-2]
        self.input_size = input_shape[-1]
        self.batch_size = input_shape[0]
        self.noise = 1e-6
        self.window = window
        self.time = 1
        self.network_name = network_name
        self.layer_neurons = layer_neurons

        # num of features 
        self.factor_c = input_shape[-1]-5
        self.ohlcv_c = 5 
        self.rf = 0.001


        # --------------------  feature extractions -------------------- 
        self.lstm_ohlcv = nn.Sequential(
                                nn.LayerNorm(self.stock_space_dimension*self.ohlcv_c),
                                nn.LSTM(input_size=self.stock_space_dimension*self.ohlcv_c, 
                                hidden_size=self.stock_space_dimension, num_layers=2, batch_first=True),
                                ) # out : (B, asset, 1) or (B, asset)
        
        self.fc_fac = nn.Sequential(
                                    nn.LayerNorm(self.factor_c),
                                    nn.Flatten(start_dim=1),
                                    nn.Linear(in_features=self.factor_c*self.stock_space_dimension, 
                                              out_features=self.factor_c*2),
                                    nn.Linear(in_features=self.factor_c*2, 
                                              out_features=self.factor_c)
                                    ) # out : (B, factor) 
                                    # 엄.. 근데 이것도 asset 만큼 나와야하는 거 아닌가 (B, asset, factor)
        
        self.fc_pred = nn.Sequential(
                                    nn.Linear(in_features=self.stock_space_dimension, 
                                              out_features=self.stock_space_dimension),
                                    # nn.Sigmoid()
                                    ) # out : (B, asset, 1) or (B, asset)


        # --------------------  layers after concatenation -------------------- 
        params = {"in_c": self.ohlcv_c+self.factor_c+1, 
                  "layer1":(self.ohlcv_c+self.factor_c+1)//2, #self.layer_neurons,
                  "layer2":self.layer_neurons*2, 
                  "out_c":self.stock_space_dimension}
        
        self.encoder = DynamicEncoder(input_shape=params["in_c"], dim=-1)      ##
        # self.encoder = Attention(enc_dim=params["in_c"], hid_dim=layer_neurons, att_type='scaled')
        self.fc1 = nn.Sequential(
                                nn.Flatten(),
                                nn.Linear(in_features=params["in_c"]*self.stock_space_dimension, 
                                          out_features=params["layer1"]),
                                nn.Tanh(), ## after 12:13 
                                # nn.Linear(in_features=params["layer1"], out_features=params["layer2"]),
                                nn.Linear(in_features=params["layer1"], out_features=self.action_space_dimension),
                                 ) 
       
        # -------------------- layers for adjusting the risk assessment part  -------------------- 
        # self.risk_adj_layer = nn.Sequential(nn.Linear(in_features=4, out_features=1),
        #                                     nn.Sigmoid()
        #                                     )
        # self.risk_adj_layer2 = nn.Sequential(nn.Linear(in_features=self.stock_space_dimension, out_features=self.layer_neurons),
        #                                     nn.ReLU()
        #                                     )

        # self.action = torch.full(size=(self.batch_size, self.action_space_dimension), 
        #                          fill_value=1/self.action_space_dimension,
        #                          requires_grad=True) #.cuda() # action container

        # activation Functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # activation scaling 
        self.act_norm = nn.LeakyReLU() #nn.Sigmoid()

        
        # for compute mu and sigma
        self.mu = nn.Sequential(nn.Linear(self.layer_neurons, self.action_space_dimension) 
                                ) #28*cnn_params["out_c"]
        self.log_sigma = nn.Sequential(nn.Linear(self.layer_neurons, self.action_space_dimension),
                                       )
        

    def _stocks_allocate(self,
                ohlcv: torch.tensor,
                prestate: torch.tensor,
                factor: torch.tensor,
                # state_factor : Union(np.ndarray, torch.tensor),
                ):
        '''
        # First, Allocate risky assets
        
        ## Input
        - ohlcv(torch.tensor, [batch, window, asset, 5])
        - prestate(torch.tensor, [batch, asset])
        - factor(torch.tensor, [batch, asset, # of factors])

        ## Output
        - action (torch.tensor, [batch, asset])
        '''
        # concat the information &  calculate loss term
        '''
        ohlcv(마지막 날) --> [1, 30, 5]
        prestate --> [1, 30, 1]
        factor --> [1, 30, 15]
        '''

        x = torch.cat([ohlcv[:,-1,:], prestate, factor], dim=-1) #(B, asset, 5+factor+1) # 차원: torch.Size([1, 30, 21])
        x, self.loss_term = self.encoder(x) # x: torch.Size([1, 30, 21])  # self.loss_term: [B,] 
        risky_action = self.fc1(x) #(B, 30, 30)인데... (B, 30)
        return risky_action


    def _risk_allocation(self,
                    risky_action: torch.tensor,
                    prestate: torch.tensor,
                    close: torch.tensor
                    # state_factor : Union(np.ndarray, torch.tensor),
                    ) -> List[torch.tensor]:
        '''
        Branch, 나중에 결과 보기
        # get the ratio of risky assets and risk-free asset..
        # motivated by CAL(Capital Allocation Line) theory
        
        '''
        # ratio-weighted expected return 
        weighted_r = risky_action*prestate #(B, assets) # # 여기서 미리 self.rf를 뺄까?
        mu_w = torch.mean(weighted_r, axis=-1, keepdim=True) 
        std_w = torch.std(weighted_r, axis=-1, keepdim=True)
        
        # 과거 mu and std of ohlcv timeseries 
        # 고민 중인게 지금 여기를 그냥 다 mean으로 뭉뚱그려도되나?
        mu_state = torch.mean(torch.mean(close, axis=1), axis=-1, keepdim=True) #[batch, window, asset, input_size] -> (B, assets)?
        std_state = torch.mean(torch.std(close, axis=1), axis=-1, keepdim=True) # -> (B, assets)? 
        
        # risk-free rate should be added
        x_CAL = torch.cat([mu_w, std_w, mu_state, std_state], dim=-1)

        
        # -------------------- calculate the risk allocation rate y  -------------------- 
        # method 1 : 최대한 CAL 이론 살리기
        risk_ratio = self.risk_adj_layer(x_CAL) # (B, 1)      # y = sigma_all/sigma_risky
        adj_risky_action = risk_ratio*risky_action
        adj_rf_action = 1-torch.sum(adj_risky_action, axis=-1) # (B, 1)
        mu = torch.cat([adj_risky_action, adj_rf_action.unsqueeze(dim=-1)], dim=-1) #(B, asset+1=action dim)
        log_sigma = risk_ratio / torch.mean(torch.std(close, axis=1)) 
        
        # method 2 : 기존 통계 수치 계산해서 neural net에 맡겨버리기
        # risk_ratio = self.risk_adj_layer(x_CAL) # (B, 1)     # y = sigma_all/sigma_risky
        # adj_risky_action = risk_ratio*risky_action #(B, assets)
        # x = self.risk_adj_layer2(adj_risky_action) # output은 layer node dim 
        # mu = self.mu(x) # action
        # log_sigma = self.log_sigma(x)


        # # method 3 : 걍 neural net한테 알아서 하라고 맡겨버려
        # x = self.risk_adj_module(weighted_r, close) # self.rf도 넣던지
        # mu = self.mu(x) # action
        # log_sigma = self.log_sigma(x)

        # method 4 : 포트폴리오의 RSI 계산? --> ENV에서 Reward에 반영
        return mu, log_sigma



    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                risk_free: float = 0.001,
                # state_factor : Union(np.ndarray, torch.tensor),
                ) -> List[torch.tensor]:
        '''
        Input
        - state(torch.tensor, [batch, window, asset, input_size])
            - ohlcv # torch.Size([1, 20, 30, 5])
            - factor # time series cutting  # torch.Size([1, 30, 15])
            - close # torch.Size([1, 20, 30])
        - prestate(torch.tensor, [batch, asset])
        
        Output
        Expectation and standard deviation of a Normal distribution
        - mu (B, Asset)
        - sigma (B, Asset)
        '''

        ohlcv = state[:,:,:,:5] # torch.Size([1, 20, 30, 5])
        factor = state[:,-1,:,5:] # time series cutting  # torch.Size([1, 30, 15])
        # close = ohlcv[:,:,:,3] # torch.Size([1, 20, 30])

        # get the risky assets' allocation action
        mu = self._stocks_allocate(ohlcv, prestate.unsqueeze(dim=-1), factor) #(B, 31)????
        self.attention = self.encoder.wm.detach().cpu().numpy() #(B, 30, 21)

        # risk-allocation branch
        # mu, log_sigma = self._risk_allocation(risky_action, prestate, close)

        # mu and sigma calculation for SAC action sampling        
        return mu, 0








class Actor6(nn.Module):
    '''
    '''
    def __init__(self,
            input_shape: Tuple,
            action_space_dimension: int,
            layer_neurons:int,
            window: int = 40,
            network_name:str = "Actor6"
            ) -> None:
        super(Actor6, self).__init__()
        '''
        항상 배치를 포함한 차원으로 넣어줄 것. (1,30,9)
        
        # input size
        state = (batch, assets, 5+factors)
        prestate = (batch, assets)

        ohlcv = (batch, assets, 5)
        factors = (batch, assets, factors(3))
        prestate = (batch, assets)

        '''
        # 공통 세팅
        self.action_space_dimension = action_space_dimension
        self.stock_space_dimension = input_shape[-2]
        self.noise = 1e-6
        self.window = window
        self.input_size = input_shape[-1]
        self.time = 1
        self.network_name = network_name
        self.layer_neurons = 16 #layer_neurons

        # num of features 
        self.factor_c = input_shape[-1]-5
        self.ohlcv_c = 5 


        # self._input_shape = self.input_size+1
        ## feature extractions
        self.lstm_ohlcv = nn.Sequential(
                                nn.LayerNorm(self.stock_space_dimension*self.ohlcv_c),
                                nn.LSTM(input_size=self.stock_space_dimension*self.ohlcv_c, 
                                hidden_size=self.action_space_dimension, num_layers=2, batch_first=True),
                                )
        self.enc_ohlcv = DynamicEncoder(n_feature=self.action_space_dimension, dim=-1)
        self.fc_fac = nn.Sequential(
                                    nn.Linear(in_features=self.factor_c, out_features=self.factor_c),
                                    DynamicEncoder(n_feature=self.action_space_dimension, dim=1)
                                    )
        self.fc_pred = nn.Sequential(
                                    nn.Linear(in_features=self.action_space_dimension, 
                                              out_features=self.action_space_dimension),
                                    nn.Sigmoid()
                                    )


        # layers after concatenation
        params = {"in_c":self.factor_c+2, "layer1":16, "layer2":16, "out_c":self.layer_neurons}
        # self.att_concat = DynamicEncoder(input_shape=(self.batch_size, self.action_space_dimension, params["out_c"]))
        self.fc1 = nn.Sequential(
                                nn.Linear(in_features=params["in_c"], out_features=1),
                                nn.Flatten(),
                                nn.Tanh(), ## after 12:13 
                                nn.Linear(in_features=self.action_space_dimension, out_features=params["out_c"]),
                                # nn.Linear(in_features=params["layer1"], out_features=params["out_c"]),
                                 )
        self.loss_w = nn.Sequential(nn.Linear(in_features=2, out_features=2),
                                    nn.Sigmoid())  # DynamicEncoder(n_feature=2, dim=-1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        
        ## for compute mu and sigma
        self.mu = nn.Sequential(nn.Linear(self.layer_neurons, self.action_space_dimension) 
                                ) #28*cnn_params["out_c"]
        self.log_sigma = nn.Sequential(nn.Linear(self.layer_neurons, self.action_space_dimension),
                                       )
        


    def forward(self,
                state: torch.tensor,
                prestate: torch.tensor,
                # state_factor : Union(np.ndarray, torch.tensor),
                ) -> List[torch.tensor]:
        '''
        Input
        - state(torch.tensor, [batch, window, asset, input_size])
        - prestate(torch.tensor, [batch, asset])
        
        Output
        Expectation and standard deviation of a Normal distribution
        - mu
        - sigma
        '''
        ohlcv = state[:,:,:,:5]
        factor = state[:,-1,:,5:] # time series cutting

        # get the stocks score of each states
        ohlcv, _= self.lstm_ohlcv(torch.flatten(ohlcv, start_dim=-2))
        ohlcv, ohlcv_loss = self.enc_ohlcv(ohlcv)
        factor, factor_loss = self.fc_fac(factor)
        prestate_ = self.fc_pred(prestate)

        # concat the information
        x = torch.cat([ohlcv[:,-1,:].unsqueeze(-1), factor, prestate_.unsqueeze(-1)], dim=-1) #(B, asset, 5+factor+1) # 차원확인
        x = self.fc1(x)

        # mu and sigma calculation for SAC action sampling
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)

        # calculate loss term : weighted loss
        loss_weight = self.loss_w(torch.stack((ohlcv_loss, factor_loss),dim=-1))
        olw, flw = loss_weight[:,0], loss_weight[:,1]
        self.loss_term = (ohlcv_loss*olw + factor_loss*flw) #(batch_size)

        return mu, log_sigma
    



'''
---------------------------------------------------------------------------------------
        self.lstm_ohlcv = nn.Sequential(
                                nn.LayerNorm(self.action_space_dimension*self.ohlcv_c),
                                nn.LSTM(input_size=self.action_space_dimension*self.ohlcv_c, 
                                hidden_size=self.action_space_dimension, num_layers=2, batch_first=True),
                                )
        self.enc_ohlcv = DynamicEncoder(n_feature=self.action_space_dimension, dim=-1)
        self.fc_fac = nn.Sequential(
                                    nn.Linear(in_features=self.factor_c, out_features=self.factor_c),
                                    DynamicEncoder(n_feature=self.action_space_dimension, dim=1)
                                    )
        self.fc_pred = nn.Sequential(
                                    nn.Linear(in_features=self.action_space_dimension, 
                                              out_features=self.action_space_dimension),
                                    nn.Sigmoid()
                                    )

'''