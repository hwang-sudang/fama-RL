'''

'''

import gym
import numpy as np
import pandas as pd
from typing import Tuple

from src.utilities import append_corr_matrix, append_corr_matrix_eigenvalues
from src.ff_factor_reward import FamaFrench



class Environment(gym.Env):
    """
        stock_market_history (df): (time_horizon * number_stocks) format ***** 변경
        initial_portfolio (dict): 포트폴리오 초기 구조 (현금, 보유 주식수)
        buy_cost: fees in percentage for buying a stock
        sell_cost: fees in percentage for selling a stock
        bank_rate: 현금 연 이자 수익률 (risk free rate)
        limit_n_stocks (float): 한번에 매매할 수 있는 최대 주식 갯수 
        buy_rule (str): 에이전트가 구매하기로 결정한 주식을 구매하는 순서 지정
        use_corr_matrix: bool = 데이터에 상관관계 붙일지 말지?
        use_corr_eigenvalues: bool = 상관관계 아이겐 벡터 붙일지 말지?
        window: int = 20,
        number_of_eigenvalues: int = 10,
    """
    
    def __init__(self,
                stock_market_history: pd.DataFrame,
                initial_portfolio: dict,
                buy_cost: float = 0.001,
                sell_cost: float = 0.001,
                bank_rate: float = 0.5,
                limit_n_stocks: float =200,
                buy_rule: str = 'most_first',
                use_corr_matrix: bool = False,
                use_corr_eigenvalues: bool = False,
                window: int = 20,
                number_of_eigenvalues: int = 10,
                agent_type: str = 'default',
                )-> None:
    
        super(Environment, self).__init__()

        # multi agent type
        # agent_type : whole(multi-agent 통괄모드), SMB, HML, premium, default(기본 리워드)
        self.agent_type = agent_type

        # 금융 시계열 데이터와 관련된 속성들
        self.stock_market_history = stock_market_history
        self.assets_list = self.stock_market_history.columns ### 컬럼에 따라 자산을 정한다?
        self.stock_space_dimension = stock_market_history.shape[1]

        # 옵션에 따라 추가: sliding correlation matrix of the time-series
        if use_corr_matrix :
            self.stock_market_history = append_corr_matrix(df=self.stock_market_history,
                                                            window = window)
        elif use_corr_eigenvalues :
            self.stock_market_history = append_corr_matrix_eigenvalues(df=self.stock_market_history,
                                                                        window=window,
                                                                        number_of_eigenvalues = number_of_eigenvalues)
        
        # 상관 행렬 또는 그 고유값을 추가한 뒤에 timeseries horizon(시간 지평)을 정의함.
        # 슬라이딩 특성 상 슬라이딩 상관행렬은 몇 개의 시점을 제거하므로..                                                      
        self.time_horizon = self.stock_market_history.shape[0]
        

        # 전처리 끝난 후 관찰 스페이스(환경?)와 액션 스페이스 정의
        self.observation_space_dimension = 1 + self.stock_space_dimension + self.stock_market_history.shape[1]
        self.action_space_dimension = self.stock_space_dimension
        self.observation_space = gym.spaces.Box(low=-np.inf, high = np.inf, shape=(self.observation_space_dimension,))
        self.action_space = gym.spaces.Box(low=-1, high = 1, shape=(self.action_space_dimension,))

        # 한번에 거래할 수 있는 최대 주식 수
        self.limit_n_stocks = limit_n_stocks

        # 매매와 은행 이자율과 관련된 attributes
        self.buy_rule = buy_rule
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.daily_bank_rate = pow(1+bank_rate, 1/365)-1
        
        self.initial_portfolio = initial_portfolio

        #initializing the state of the environment : 환경정보 초기화 
        self.current_step = None
        self.cash_in_bank = None
        self.stock_prices = None
        self.number_of_shares = None
        self.reset()


    def reset(self) -> np.array :
        """
        은행 잔고, 주식 시장 상태를 초기화하는 함수
        return : 최초의 관찰 정보 (np.array)
        """

        self.current_step = 0
        self.cash_in_bank = self.initial_portfolio["Bank_account"] # dict
        self.stock_prices = self.stock_market_history.iloc[self.current_step]
        self.number_of_shares = np.array([self.initial_portfolio[ticker] for ticker in self.stock_market_history.columns[:self.action_space_dimension]])

        return self._get_observation()


    def step(self,
            actions: np.ndarray,
            ) -> Tuple[np.ndarray, float, bool, dict] :
        
        '''
        trading 환경에서 한 스텝 이동 st -> at -> st+1 
        args : action(np.array) = 에이전트 매매 행동, 연속적
        returns : np.array for the new step
        '''

        self.current_step += 1
        initial_value_portfolio = self._get_portfolio_value()
        self.stock_prices = self.stock_market_history.iloc[self.current_step]

        self._trade(actions)

        self.cash_in_bank *= 1 + self.daily_bank_rate
        new_value_portfolio = self._get_portfolio_value()
        done = self.current_step == (self.time_horizon -1)  # 이게 무엇을 의미하나
        info = {'value_portfolio': new_value_portfolio}


        famafrench = FamaFrench() 
        #################################### 여기서 부터 작업하기! #########################################
        # famafrench.calcFactorReturn(self, factor:dict, action:torch.tensor, agent:str ='SMB')

        ## reward 삽입
        if self.agent_type == 'default' or "whole":
            reward = new_value_portfolio - initial_value_portfolio
        elif self.agent_type == 'SMB':
            reward = famafrench.calcFactorReturn(factor, actions, agent="SMB")
        elif self.agent_type == 'HML':
            reward = famafrench.calcFactorReturn(factor, actions, agent="HML")
        elif self.agent_type == 'premium':
            reward = famafrench.calcFactorReturn(factor, actions, agent="premium")     
        else :
            raise Exception("Wrong Agent Factor Type. Choose : SMB, HML, premium. ")
        

        return self._get_observation(), reward, done, info


    def _trade(self, actions: np.ndarray, ) -> None :
        '''
        에이전트가 내린 구매 행동에 따라 거래를 수행

        buying rule 종류
        - most_first : 가장 큰 구매 비율을 가진 주식을 먼저 구매한다. (추세가 있을 것이라 판단했을 것이므로)
        - random : First sell, then buy 
        
        args : action(np.array)
        returns : None
        '''

        actions = (actions * self.limit_n_stocks).astype(int) ### 여기 수정하기 
        sorted_indices = np.argsort(actions) # 어레이 정렬 (ascending) : [-2,-1,1,2,]

        sell_idx = sorted_indices[:np.where(actions<0)[0].size] # 판매에 해당하는 항목이 있는 마지막 idx
        buy_idx = sorted_indices[::-1][:np.where(actions>0)[0].size] # 구매에 해당하는 항목이 있는 첫번째 idx

        # 먼저 판매부터 진행하기
        for idx in sell_idx:
            self._sell(idx, actions[idx])

        # 가장 비율이 높은 것을 먼저 구매하자
        if self.buy_rule  == 'most_first':
            for idx in buy_idx:
                self._buy(idx, actions[idx])
        
        if self.buy_rule == 'random':
            # 팔고나서 랜덤으로 상품 구매 하기
            should_buy = np.copy(actions[buy_idx])
            while self.cash_in_bank > 0 and not np.all((should_buy==0)):
                # 랜덤으로 구매
                i = np.random.choice(np.where(should_buy > 0))
                self._buy(buy_idx[i])
                should_buy[i] -= 1

    
    def _sell(self, idx: int, action: int, ) -> None:
        '''
        판매 디시전 내려진 주식 판매하기
        args: 
        - idx : 판매해야할 주식의 인덱스 정보
        - actions (int) : 주식 판매 갯수 --> 이건 변경해야함.

        returns :None
        '''

        # 만약 주식이 1개도 없으면 못 판다. 아무런 행동 X
        if int(self.number_of_shares[idx]) < 1:
            return
        
        n_stocks_to_sell = min(-action, int(self.number_of_shares[idx])) # 최대 판매 가능 갯수 정함..
        money_outflow = n_stocks_to_sell * self.stock_prices[idx] * (1-self.sell_cost) # 거래비용 감안한 총 구매 금액
        self.cash_in_bank += money_outflow # 판매한 만큼 현금 입금
        self.number_of_shares[idx] -= n_stocks_to_sell 


    def _buy(self, idx: int, action: int = 1, ) -> None:
        '''
        액션에서 구매 비용만큼 구매하기: 잔고에 있는 만큼만
        args : idx(구매해야할 주식의 인덱스 정보)
        actions : 주식 구매 갯수 --> 이건 변경해야함.

        returns :None
        '''
        # 현금 없으면 못 삼
        if self.cash_in_bank < 0:
            return
        
        n_stocks_to_buy = min(action, self.cash_in_bank // self.stock_prices[idx]) # 최대 구매 가능 갯수 정함..
        money_outflow = n_stocks_to_buy * self.stock_prices[idx] * (1+self.buy_cost) # 거래비용 감안한 총 구매 금액
        self.cash_in_bank -= money_outflow # 구매한 만큼 현금 차감
        self.number_of_shares[idx] += n_stocks_to_buy 



    def _get_observation(self) -> np.array :
        '''
        state_space가 제공하고 에이전트가 인식하는 형식의 관찰
        '''
        observation = np.empty(self.observation_space_dimension) # 틀 만들기
        observation[0] = self.cash_in_bank # 첫 항은 현금을 의미함
        observation[1:self.stock_prices.shape[0]+1] = self.stock_prices # 1번째 인덱스부터 끝까지 각 자산의 보유 주식수
        observation[self.stock_prices.shape[0]+1 : ] = self.number_of_shares # 마지막 항은 총 보유 주식수.
        return observation
    


    def _get_portfolio_value(self) -> float :
        '''
        기 보유 중인 주식수 * 주가 + 현금 계산
        
        Returns : np.array for the 총 포트폴리오 밸류
        '''

        portfolio_value = self.cash_in_bank \
                        + self.number_of_shares.dot(self.stock_prices[:self.stock_space_dimension])
        
        return portfolio_value

            


