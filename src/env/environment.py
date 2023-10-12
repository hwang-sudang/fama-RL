import os
import re
import gym
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
import warnings

from util.configures import *
from data.dataloader import PortfolioDataset
from model.networks.pretrain_net import StackedGRU
os.environ['CUDA_LAUCH_BLOCKING'] = "1"

warnings.filterwarnings('ignore')



class PortfolioEnv(gym.Env):
    """
        stock_market_history (df): (time_horizon * number_stocks) format ***** 변경
        initial_portfolio (dict): 포트폴리오 초기 구조 (현금, 보유 주식수)
        trade_cost: fees in percentage for buying a stock
        bank_rate: 현금 연 이자 수익률 (risk free rate)
        window: int = 20,
        agent_type : size, value, vol, default(기본 리워드)
        use_3denv : make 3d env(T) or 2d env(F)
    """
    def __init__(self,
                initial_portfolio: dict,
                args:dict,
                checkpoint_directory:str,
                # trade_cost: float = 0.00015,
                # window: int = 40,
                # agent_type: str = 'default',
                device : str = 'cpu',
                )-> None:
        super(PortfolioEnv, self).__init__()
        #----------------------- stock dataframe과 상관 없는 영역 --------------------
        # multi agent type
        # agent_type : whole(multi-agent 통괄모드), size, value, vol, default(기본 리워드)
        self.device = device
        self.agent_type = args.ffactor
        self.ffactor = args.ffactor
        self.prewindow = args.prewindow
        self.window = args.window
        self.pretrain = args.pretrain
        self.checkpoint_directory = checkpoint_directory

        # define risk-free dataframe
        self.riskfree_df = pd.read_csv(f"{DATA_DIR}/factor_pre/Risk_Free.csv", parse_dates=['Unnamed: 0'], index_col=0)
        self.riskfree_df.index = self.riskfree_df.index.astype(str)
        self.riskfree_df.index = pd.to_datetime(self.riskfree_df.index)

        # 매매와 은행 이자율과 관련된 attributes
        self.trade_cost = args.trade_cost
        self.initial_portfolio = initial_portfolio
        self.past_portfolio_value= initial_portfolio["Bank_account"]

        #initializing the state of the environment : 환경정보 초기화 
        self.current_step = None
        self.cash_in_bank = None
        self.stock_prices = None
        self.number_of_shares = None

        #----------------------- stock dataframe 차원과 관련된 영역 --------------------
        dataset = PortfolioDataset(args)
        data_dict = dataset.getdata()
        
        # stock market_history : (T, window, asset, features) OR (T, window=1, asset, features)
        self.stock_market_history = data_dict["series"].to(self.device) #cuda를 해주긴 해야하는데
        self.close_p = data_dict["close_t"].to(self.device)
        self.log_close = data_dict["log_close"].to(self.device)
        self.dates = data_dict["dates"]
        self.stocks_subsets = [i for i in initial_portfolio if i != "Bank_account"] ##
        self.stock_space_dimension = len(self.stocks_subsets)
        self.time_length = min(len(self.stock_market_history), len(self.close_p)) # 위에 네개가 길이가 다 똑같은지 확인
        self.factor_name = dataset.dataprocess.factor_name 

        # 관찰스페이스와 액션 스페이스 정의
        self.observation_space_dimension = tuple(self.stock_market_history.shape[1:]) # (4317,30,9)
        self.action_space_dimension = self.stock_space_dimension + 1 # include cash
        self.observation_space = gym.spaces.Box(low=-3, high=3, shape=self.observation_space_dimension)
        self.action_space = gym.spaces.Box(low=0, high = 1, shape=(self.action_space_dimension,))
        
        # pretrain model settings
        self.pretrain_net_dict = {}
        if self.pretrain : 
            pretrain_net_dir = f"{BASE_DIR}/bestmodel/{re.sub(r'[0-9]+', '', args.country)}/weights_{args.prewindow}"
            for s in self.stocks_subsets:
                self.pretrain_net_dict[s] = StackedGRU(input_shape=(self.prewindow, self.stock_space_dimension),
                                        hidden_dim=4, 
                                        out_dim=1, 
                                        pretrain=self.pretrain,
                                        )
                self.pretrain_net_dict[s].load_state_dict(torch.load(pretrain_net_dir+ \
                                                        f"/price_{s}/BestModel_price_{s}.pth",
                                                        map_location=self.device), 
                                                        strict=False)    
                self.pretrain_net_dict[s].requires_grad_ = False
        
        else :
            for s in self.stocks_subsets:
                # init weight of Return prediction model
                pretrain_net_dir = self.checkpoint_directory+f"/networks/pretrain/price_{s}"
                os.makedirs(pretrain_net_dir, exist_ok=True)
                self.pretrain_net_dict[s] = StackedGRU(input_shape=(self.prewindow, self.stock_space_dimension),
                                        hidden_dim=4, 
                                        out_dim=1, 
                                        pretrain=self.pretrain,
                                        )                
                self.pretrain_net_dict[s].init_weights()            

                 
        self.pretrain_net_lst = sorted(self.pretrain_net_dict)
        self.reset()




    def reset(self):
        """
        환경 초기화
        은행 잔고, 주식 시장 상태를 초기화하는 함수
        return : 최초의 관찰 정보 (np.array)
        """        
        self.current_step = 0
        self.init_portfolio_value = self.initial_portfolio["Bank_account"]
        self.cash_in_bank = self.initial_portfolio["Bank_account"]
        self.daily_bank_rate = (self.riskfree_df.iloc[self.current_step].item()+1)**(1/365)

        self.stock_prices = torch.tensor(self.close_p[self.current_step], requires_grad=False, device=self.device)
        self.number_of_shares = torch.tensor([self.initial_portfolio[ticker] for ticker in self.stocks_subsets], 
                                             dtype=torch.float32, requires_grad=False, device=self.device)
        
        self.past_value_per_asset = torch.mul(self.number_of_shares, self.stock_prices)
        self.past_portfolio_value= self._get_portfolio_value()
        
        # for sharpe return
        self.return_t_lst = np.array([])
        self.return_cum_lst = np.array([])
        return self._get_observation()




    def step(self, action_ratio:torch.tensor) -> Tuple[np.ndarray, float, bool, dict] :
        '''
        ## Trading 환경에서 한 스텝 이동 st -> at -> st+1 
            - args : action(torch.tensor) = 에이전트 매매 행동, 연속적
            - returns : np.array for the new step

        ## Reward 계산하기 위한 Attribute
            - factor : 
        '''       
        # 현재 시장 및 에이전트의 상황 계산 
        ## 가격 변동
        self.current_step += 1
        self.current_date = self.dates[self.current_step]
        self.old_stock_prices = self.close_p[self.current_step-1] 
        self.stock_prices = self.close_p[self.current_step]
        self.daily_bank_rate = (self.riskfree_df.iloc[self.current_step].item()+1)**(1/365)
        stock_action = action_ratio[:-1].squeeze()
        
        ## action 에 따라 자산 구매
        self._trade(stock_action) 
        
        # reward 형식
        # 내일 정민이한테 여기 최적화 어떻게 할지 살짝 물어보기
        if self.ffactor == 'size':
            factor_score = torch.sum(stock_action*self.stock_market_history[self.current_step, -1, :, 6]).item()
        elif self.ffactor == 'value':
            factor_score = torch.sum(stock_action*self.stock_market_history[self.current_step, -1, :, 5]).item()
        elif self.ffactor == 'vol':
            factor_score = torch.sum(stock_action*self.stock_market_history[self.current_step, -1, :, 10]).item()
        else:
            # 와 여기 필수로... DEFAULT OPTION 지정 필요
            factor_score = 0


        # 학습에 대한 정보 저장(?)
        done = self.current_step >= (self.time_length -1) 
        info = {'portfolio': self.new_portfolio_value, 
                'asset_return':(self.number_of_shares*self.stock_prices-self.past_value_per_asset).tolist(),
                'return_t' : np.log(self.new_portfolio_value/self.past_portfolio_value),
                'return_cum' : np.log(self.new_portfolio_value/self.init_portfolio_value),
                'factor_score' : factor_score}
        

        ##### REWARD #####
        # scale check for factor_score
        # reward = np.log(self.new_portfolio_value/self.past_portfolio_value) * 10 \
        #         + np.log(self.new_portfolio_value/self.init_portfolio_value) * 5 + factor_score*0.3 ##

        # use return as sharpe ratio        
        reward = self._sharpe_return(info)

        ## reward Engineering : Clipping & tuning ##
        if reward > 60 : 
            reward = 60
        elif reward == np.nan:
            print("The Reward is NaN. Check the Reward calculation formula.")
            reward = 0
        else : pass        

        # save the previous portfollo info
        self.past_value_per_asset = self.number_of_shares*self.stock_prices
        self.past_portfolio_value = self.new_portfolio_value

        return self._get_observation(), reward, done, info



    def _trade(self, action_ratio:torch.tensor, ) -> None :
        '''
        에이전트가 내린 구매 행동이 포트폴리오 비중 일때
        에이전트가 내린 구매 행동에 따라 거래를 수행

        Pseudo code
        1. 가격 변화 전에 rebalancing with action_ratio, 새로운 자산 수 구하기
        2. 가격이 변함
        3. 자산 수 * 새로운 가격 으로 new portfolio value 계산

        Buying rule 종류
        - most_first : 가장 큰 구매 비율을 가진 주식을 먼저 구매한다. (추세가 있을 것이라 판단했을 것이므로)
        - random : First sell, then buy 
        
        Args : action_ratio(torch.tensor)
        - returns : None
        '''
        # 1. 과거 가격 기준으로 action_ratio 기준으로 자산분배
        self.number_of_shares = torch.floor((action_ratio*self.past_portfolio_value) / self.old_stock_prices)
        
        # 1-1. 남은 잔고를 계산하기 위해 action ratio 다시 재조정
        self.actions_without_cash = (self.number_of_shares*self.old_stock_prices) / self.past_portfolio_value
        self.cash_in_bank = (1-torch.sum(self.actions_without_cash))*self.past_portfolio_value

        # 2. 가격이 변동하고, 각 자산의 포트폴리오 분배금액이 바뀜
        # 3. 자산 수(1-1) * 새로운 가격 으로 new portfolio value 계산
        trading_cost = (self.number_of_shares*self.stock_prices-self.past_value_per_asset).sum()*self.trade_cost
        self.cash_in_bank *= self.daily_bank_rate 
        self.cash_in_bank -= trading_cost.item()
        self.new_portfolio_value = self._get_portfolio_value()



    def _cal_hidden_state(self, pre_obs:torch.Tensor,):
        '''
        이미 학습시킨 RNN 모델로부터 state 중 하나인 hidden state 값을 받아온다.
        Input :
            - pre_obs(torch.Tensor, (Batch, timeseries, assets, imfs))
        Output : 
            -  prestate(torch.tensor, (Batch, timeseries, assets))
        '''
        windows, n_assets, n_imfs = pre_obs.shape
        assert self.stock_space_dimension == n_assets, f"Dimensions of n_assets are Different: self:{self.stock_space_dimension}, data:{n_assets} "
        pre_state = []

        # with torch.no_grad():
        for i in range(n_assets):
            model = self.pretrain_net_dict[self.stocks_subsets[i]].to(self.device)
            out = model.forward(pre_obs[:,i,:]) # in:(40,3) # out:log return 예측값 # network 
            pre_state.append(out)

        pre_state = torch.stack(pre_state).to(self.device) # shape 확인
        return pre_state
    


    def _get_observation(self) -> np.array :
        '''
        state_space가 제공하고 에이전트가 인식하는 형식의 관찰
        observation에 넣고 싶은 정보(차원) : (timelength, tickers, features)
        observation이 담고 있어야 할 정보는? : (1, tickers, features)
        '''

        # observation을 (timeseries, ticker, features) 형태로 데이터 넣기 위한 과정.
        # 차원 잘 맞춰주기 & type?
        pre_obs = self.log_close[self.current_step] #(40, 30, 3)
        obs = self.stock_market_history[self.current_step] #(40, 30, 7) # gradient 흐르나?
        
        if self.pretrain:
            with torch.no_grad():
                # 학습할 때 pre_obs에 그래디언트 안 흐르게
                # freezing된 모델의 아웃풋에도 그래디언트 흘러도 되나? 상관없나?
                pre_obs = self._cal_hidden_state(pre_obs)
        else :
            # 사전학습 안할 때에는 그래디언트 흘러도 됨
            pre_obs = self._cal_hidden_state(pre_obs)

        pre_obs = pre_obs.unsqueeze(0) 
        obs = obs.unsqueeze(0) # check the shape
        return (obs, pre_obs)
    


    def _get_portfolio_value(self) -> float :
        '''
        기 보유 중인 주식수 * 주가 + 현금 계산
        Returns : 총 포트폴리오 밸류
        '''
        portfolio_value = self.cash_in_bank + (self.number_of_shares*self.stock_prices).sum()
        return portfolio_value.item()



    def _sharpe_return(self, info):
        '''
        should i need to change it tensor based computation?
        
        done = self.current_step >= (self.time_length -1) 
        info = {'portfolio': self.new_portfolio_value, 
                'asset_return':(self.number_of_shares*self.stock_prices-self.past_value_per_asset).tolist(),
                'return_t' : np.log(self.new_portfolio_value/self.past_portfolio_value),
                'return_cum' : np.log(self.new_portfolio_value/self.init_portfolio_value),
                'factor_score' : factor_score}
        
        source code :
        https://github.com/aekram43/rl-trader/blob/master/position_bot/environment.py
        '''
        def clip_sharp(sharp):
            '''sharp ratio가 nan 값이 뜨지 않도록'''
            sharp = np.nan_to_num(sharp)
            sharp = np.clip(sharp, -4, 4)  # -1 1이 원래 클리핑            
            return sharp
        
        # normal return rate
        # self.return_t_lst = np.append(self.return_t_lst, (self.new_portfolio_value/self.past_portfolio_value)-1)
        # self.return_cum_lst = np.append(self.return_cum_lst, (self.new_portfolio_value/self.init_portfolio_value)-1)
        
        # Log Return Rate
        self.return_t_lst = np.append(self.return_t_lst, info["return_t"])
        self.return_cum_lst = np.append(self.return_cum_lst, info["return_cum"])


        sharp_t = np.nanmean(self.return_t_lst) /(np.nanstd(self.return_t_lst) * np.sqrt(len(self.return_t_lst)))
        sharp_cum = np.nanmean(self.return_cum_lst) / (np.nanstd(self.return_cum_lst) * np.sqrt(len(self.return_cum_lst)))
        factor_score = info["factor_score"]        
        
        
        # reward = info['return_t'] + 0.05*clip_sharp(sharp_cum) + 0.01*factor_score
        # reward = clip_sharp(sharp_t) + 0.5*info['return_cum'] + 0.01*factor_score
        reward = clip_sharp(sharp_t) + clip_sharp(sharp_cum) + 0.01*factor_score  ## 이것도 뭔가 하나에만 적용해야 할 거 같은데...
        # print(f"sharp_t: {sharp_t}    sharp_cum: {sharp_cum}    factor_score: {factor_score}")
        return reward
    