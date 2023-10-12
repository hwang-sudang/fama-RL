####

import numpy as np
import torch

## inner module import 여기 손한번 보기
from model.agent.base import Agent
from util.configures import *
from model.networks.critic import *
from model.networks.actor import *
torch.autograd.set_detect_anomaly(True)


class EWAgent(Agent):
    '''
    Equally-weighted Portfolio action
    Agent 클래스를 상속받으며, Temperature가 학습과정에서 자동으로 업데이트 되는 SAC.

    Attributes:
        alpha (float): initial value of the temperature parameter,
                       initialized by sampling from a standard normal distribution. 
    '''
    def __init__(self,
                lr_alpha: float,
                pretrain_opt:str ='train',
                country = 'USA',
                *args,
                **kwargs,
                ) -> None:
        super(EWAgent, self).__init__(*args, **kwargs)
        
        number_of_agent = 1
        self.country = country
        self.name = 'EquallyWeight'
        self.mode = 'test'


    def learn(self, 
            step: int = 0,
            eps: float = 0.5
            ) -> None:
        '''
        No use
        To Prevent NotImplementedError        
        '''
        pass
    
    
    def choose_action(self, observation:torch.tensor) -> np.array :
        '''
        Always same action : 1/n_asset
        Args
            observation (Tuple(np.ndarray) : state of the env
            - state : (Batch, 30, factors)
            - prestate : (Batch, 30, 1)
        returns
            action (np.array) taken in the input state
        '''
        
        state, pre_state = observation
        batch_size = pre_state.shape[0]
        n_asset = pre_state.shape[-1]
        # action_ratio = np.repeat(np.array([[1/n_asset]]*n_asset), 
        #                           repeats=batch_size, axis=0)
        action_ratio = torch.repeat_interleave(torch.tensor([[1/(n_asset+1)]]*(n_asset+1)).to(self.device), 
                                  repeats=batch_size, axis=0)
        return action_ratio.squeeze()
    
    
    
    

class SecurityAgent(Agent):
    '''
    Actions for Only Cash Portfolio 
    Agent 클래스를 상속받으며, Temperature가 학습과정에서 자동으로 업데이트 되는 SAC.

    Attributes:
        alpha (float): initial value of the temperature parameter,
                       initialized by sampling from a standard normal distribution. 
    '''
    def __init__(self,
                lr_alpha: float,
                pretrain_opt:str ='train',
                country = 'USA',
                *args,
                **kwargs,
                ) -> None:
        super(SecurityAgent, self).__init__(*args, **kwargs)
        
        number_of_agent = 1
        self.country = country
        self.name = 'Security'
        self.mode = 'test'

    def learn(self, 
            step: int = 0,
            eps: float = 0.5
            ) -> None:
        '''
        No use
        Prevent NotImplementedError        
        '''
        pass
    
    
    def choose_action(self, observation:torch.tensor) -> np.array :
        '''
        Always same action : 0, Only Cash
        Args
            observation (Tuple(np.ndarray) : state of the env
            - state : (Batch, 30, factors)
            - prestate : (Batch, 30, 1)
        returns
            action (np.array) taken in the input state
        '''
        
        state, pre_state = observation
        batch_size = pre_state.shape[0]
        n_asset = pre_state.shape[-1]
        action_ratio = torch.repeat_interleave(torch.tensor([0]*(n_asset+1)).to(self.device), 
                                  repeats=batch_size, axis=0)
        # action_ratio = np.repeat(np.array([[0]]*n_asset), 
        #                           repeats=batch_size, axis=0)
        return action_ratio #.squeeze()
    





class BuyAndHoldAgent(Agent):
    '''
    Actions for Only Cash Portfolio 
    Agent 클래스를 상속받으며, Temperature가 학습과정에서 자동으로 업데이트 되는 SAC.

    Attributes:
        alpha (float): initial value of the temperature parameter,
                       initialized by sampling from a standard normal distribution. 
    '''
    def __init__(self,
                lr_alpha: float,
                country = 'USA',
                *args,
                **kwargs,
                ) -> None:
        super(BuyAndHoldAgent, self).__init__(*args, **kwargs)
        
        # del self.memory
        number_of_agent = 1
        self.name = 'BuyAndHold'
        self.country = country
        self.mode = 'test'
        self.pre_action = torch.repeat_interleave(torch.tensor([[1/self.action_space_dimension,]]*self.action_space_dimension,).to(self.device), 
                                  repeats=self.batch_size, axis=0)
        


    def learn(self, 
            step: int = 0,
            eps: float = 0.5
            ) -> None:
        '''
        No use
        Prevent NotImplementedError        
        '''
        pass
    
    
    
    def choose_action(self, observation:torch.tensor) -> torch.tensor :
        '''
        Always same action : 0, Only Cash
        Args
            observation (Tuple(torch.tensor) : state of the env
            - state : (Batch, 30, OHLCV+factors)
            - prestate : (Batch, 30, 1)
        returns
            action (np.array) taken in the input state
        '''
        return 