# copyright notice
# https://github.com/MatthieuSarkis/Portfolio-Optimization-and-Goal-Based-Investment-with-Reinforcement-Learning/blob/master/src/agents.py
import gym
import numpy as np
import os
import torch
from typing import Tuple

## inner module import
from model.buffer.buffer import *
from model.networks.actor import *
from util.configures import *

torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUCH_BLOCKING'] = "1"


class Agent():
    '''
    다양한 강화학습 에이전트가 상속할 기본 에이전트 클래스
    '''
    def __init__(self,
                lr_Q: float,
                lr_pi: float,
                input_shape: Tuple,
                tau: float,
                env: gym.Env,
                checkpoint_directory_networks: str,
                stocks_subsets:list,
                window:int,
                factor:str='default',
                buffer:str='basic',
                gamma: float = 0.99,
                size: int = 100000,
                layer_size: int = 256,
                batch_size: int = 256,
                delay: int = 1,
                grad_clip: float=1.0,
                device: str = 'cuda:0',
                name: str = None,
                mode: str = 'train'
                ) -> None:
        
        
        # data info
        self.name = name
        self.factor = factor
        self.mode = mode
        self.input_shape = input_shape 
        self.window = window
        self.n_assets = len(stocks_subsets)
        self.env = env
        self.action_space_dimension = self.env.action_space_dimension
        self.stocks_subsets = sorted(stocks_subsets)
        self.memory = None
        
        # hyperparams
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_Q = lr_Q
        self.lr_pi = lr_pi
        self.layer_size = layer_size
        self.size = size
        self.delay = delay
        self.grad_clip = grad_clip
        self.device = device
        self.checkpoint_directory_networks = checkpoint_directory_networks
        self.checkpoint_directory_pretrain = checkpoint_directory_networks.split("/saved_outputs")[0]+"/bestmodel/weights"
        
        # set replay buffer option
        self.buffer_name = buffer
        
        self._network_list = []
        self._targeted_network_list = []



    def remember(self, 
                state: np.ndarray,
                action, 
                reward: float,
                new_state: np.ndarray,
                done: bool,
                ) -> None:
        '''
        리플레이 버퍼에 obs 저장
        근데 좀 비효율적인거 같은데
        '''
        self.memory.push(state, action, reward, new_state, done) 


    @staticmethod
    def _initialize_weights(net: torch.nn.Module) -> None:
        if type(net) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(1e-2)
    

    @staticmethod
    def MSELoss(target, pred, weights):
        loss = torch.abs(target.squeeze()-pred)**2
        loss *= weights
        mse_loss = 0.5 * loss.mean()
        return mse_loss


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    

    def save_networks(self) -> None:
        '''체크포인트 모델 저장'''
        print("\n *** SAVING NETWORK WEIGHTS *** \n")
        for network in self._network_list:
            network.save_network_weights()


    def load_networks(self) -> None :
        '''저장된 체크포인트 모델 불러오기'''
        '''pretrain mode에 따라서 모델 웨이트 한번에 받아 올 수 있게 어떻게 못하나?'''
        print('\n *** LOADING NETWORK WEIGHTS *** \n')
        for network in self._network_list:
            network.load_network_weights()


    def init_networks(self) -> None :
        print('\n *** INITIALIZE NETWORK WEIGHTS *** \n')
        for network in self._network_list:
            network.apply(self._initialize_weights)   



    def learn(self,
            step: int =0,
            ) -> None:
        '''
        One step of the learning process.
        '''
        raise NotImplementedError    
    
    
    def choose_action(self, observation:torch.tensor) -> torch.tensor :
        '''
        환경으로부터 관찰한 state를 기반으로 액션을 선택함.
        Args
            observation (Tuple(torch.tensor) : state of the env
        returns
            action (torch.tensor) taken in the input state

        '''
        raise NotImplementedError   