import torch
import numpy as np


class ReplayBuffer():
    '''
    Plays the role of memory for Agents,
    by storing (state, action, reward, state_, done) tuples.
    여기에서 애초에 인풋을 32개로 넣을 수도 있는데... 그 부분도 생각해보기 : input_shape 확인
    '''

    def __init__(self,
                size: int,
                input_shape: tuple,
                action_space_dimension: int,
                device: str
                )-> None:
        
        '''
        Replay buffer 생성자.
        
        Args:
        - size(int) : 리플레이 버퍼의 최대 사이즈
        - input shape (tuple, (window, asset, features)): dimension of the observation space 
        - action_space_dimension (int) :  dimension of the action space
        
        Output : none
        '''
        self.device = device
        self.size = size
        self.pointer = 0
        self.input_shape = input_shape
        self.pre_shape = input_shape[1] # 인풋사이즈에 혹시 배치사이즈도 들어가는지 확인하기
        
        self.state_buffer = {"obs": torch.zeros((self.size, *self.input_shape), device=self.device, requires_grad=False), 
                             "pre_obs":torch.zeros((self.size, self.pre_shape), device=self.device, requires_grad=False)} 
        self.new_state_buffer = {"obs": torch.zeros((self.size, *self.input_shape), device=self.device, requires_grad=False),
                                 "pre_obs":torch.zeros((self.size, self.pre_shape), device=self.device, requires_grad=False)}  # *input_size : iterable 해야함.
        self.action_ratio_buffer = torch.zeros((self.size, action_space_dimension), device=self.device,requires_grad=False)
        self.reward_buffer = torch.zeros(self.size, device=self.device, requires_grad=False)
        self.done_buffer = torch.zeros(self.size, dtype = bool, device=self.device, requires_grad=False)


    def push(self,
            state: torch.tensor, 
            action: torch.tensor,
            reward: float,
            new_state: torch.tensor, 
            done: bool,
            )-> None:
        
        '''
        Replay bufferd에 메모리 추가.
        
        Args / Input:
        - state (tuple(torch.tensor, torch.tensor)) : (pre_obs, obs) 환경 하에서 관찰한 현 시점의 state
            - obs (0): (B, window, assets, features) == obs_space_dimension  
            - pre_obs (1) : (B, window, assets)
        - action (np.array) : 현 시점 state 하에서 선택한 액션
        - reward (float)
        - new_state (tuple(np.array, np.array)) : (pre_obs, obs) 환경 하에서 관찰한 다음 시점의 state
            - obs : (B, window, assets, features) == obs_space_dimension   
            - pre_obs : (B, window, assets)
        - done (bool) :  whether one has reached the horizion or not

        Returns : none
        '''
        # if not isinstance(state[0], np.ndarray):
        #     state = (state[0].cpu().numpy(), state[1].cpu().numpy())  
        #     new_state = (new_state[0].cpu().numpy(), new_state[1].cpu().numpy())

        assert type(action) in (torch.tensor, torch.Tensor), "action data type should be torch.tensor but {} .".format(type(action))
        index = self.pointer % self.size #나머지 = 새로운 버퍼의 인덱스
        
        self.state_buffer['obs'][index], self.state_buffer['pre_obs'][index] = state
        self.action_ratio_buffer[index] = action
        self.reward_buffer[index] = reward
        self.new_state_buffer['obs'][index], self.new_state_buffer['pre_obs'][index] = new_state
        self.done_buffer[index] =  done

        self.pointer +=1
        assert tuple(self.state_buffer['obs'].shape) == (self.size, *self.input_shape), "Replay Buffer Error"
        


    def sample(self, 
                batch_size: int = 32
                ) :
        '''
        Sample a batch of data from the buffer.

        Args:
        - batch_size (int) : 배치사이즈 
        
        Returns : a tuple of np.array of memories, 
                one memory being of the form (state, action, reward, state_, done)
                -> Tuple[S:tuple(np.ndarray), A:np.ndarray, R:np.ndarray, S_:tuple(np.ndarray), Done:np.ndarray]
        '''
        size = min(self.pointer, self.size)
        batch = np.random.choice(size, batch_size) # 랜덤으로 배치 고르기 # 일단 해보자 인덱스니까

        # sample로서 뽑힌 batch로 구성된 SARS, done
        states = (self.state_buffer['obs'][batch], self.state_buffer['pre_obs'][batch])
        actions = self.action_ratio_buffer[batch]
        rewards = self.reward_buffer[batch]
        new_states = (self.new_state_buffer['obs'][batch], self.new_state_buffer['pre_obs'][batch])
        dones = self.done_buffer[batch]
        return states, actions, rewards, new_states, dones


    def __len__(self):
        return len(self.action_ratio_buffer)



class PERBuffer(ReplayBuffer):
    '''
    Plays the role of memory for Agents,
    by storing (state, action, reward, state_, done) tuples.
    '''

    def __init__(self,
                factor:str,
                alpha:float = 0.8,
                beta:float = 0.4,
                *args,
                **kwargs,
                )-> None:
        super(PERBuffer, self).__init__(*args, **kwargs)
        
        '''
        PER Replay buffer 생성자.
        
        Parents Args:
        - size(int) : 리플레이 버퍼의 최대 사이즈
        - input shape (tuple): dimension of the observation space 
        - action_space_dimension (int) :  dimension of the action space
        
        Args:
        - factor(str): The Factor on which PER is calculated
        - alpha(float): a prioritization adjustment parameter
        - beta(float): importance sampling parameter; importance-sampling correction의 정도를 점진적으로 상승(annealing)시켜 training의 막바지에는 최대로 correction이 되도록 유도한다.
        
        Output : none
        '''
        # 나머지 인풋은 Replay Buffer로 부터 상속받음.
        self.factor = factor
        self.priorities = torch.zeros(self.size, device=self.device, requires_grad=False)
        self.epsilon = 1e-5
        self.alpha = alpha
        self.beta = beta
        


    def _update_priorities(self, noise:bool=True):
        '''
        td error로 기존에는 계산했지만, 여기서는 각 팩터의 정보대로 계산
        '''
        if self.factor == 'size' or self.factor == 'default': 
            factor =  self.state_buffer['obs'][:, -1, :, 6] # 예상 (100000, 30, 8) -> (10000, 30)         
        elif self.factor == 'value':
            factor = self.state_buffer['obs'][:, -1, :, 5] # state 사이즈 정보
        elif self.factor == 'vol':
            factor = self.state_buffer['obs'][:, -1, :, 10] # state 사이즈 정보  
        # elif self.factor == 'default':
            # pass
        else:
            raise NameError("Unknown factor name : {}".format(self.factor))
        
        # add randomness for exploration -> adjust prioritization parameter
        # if noise and self.pointer > self.size:
        #     rand = torch.distributions.Uniform(low=self.epsilon, high=self.epsilon*10).sample(tuple(factor.shape)).to(self.device)
        #     factor_ = factor+rand.detach()
        # else :
        #     factor_ = factor

        # rank and normalize it

        pf_weight = self.action_ratio_buffer[:, :-1].clone().detach() # cash 제외 때문에 ㅠㅠ
        prior_score = (factor*pf_weight).sum(axis=-1) # 예상 (10000,)
        rank_score = (prior_score.argsort().argsort()+1) / len(prior_score) # 모든 샘플에 대해서 랭크스코어를 매기는게 맞나..?...
        self.priorities = rank_score # + self.epsilon   shape : (10000,)
        
        

    def push(self,
            state: torch.tensor, 
            action: torch.tensor,
            reward: float,
            new_state: torch.tensor, 
            done: bool,
            )-> None:
        
        '''
        Replay buffer에 메모리 추가.
        
        Args / Input:
        - state (tuple(np.array, np.array)) : (pre_obs, obs) 환경 하에서 관찰한 현 시점의 state
            - pre_obs : (B, window, assets)
            - obs : (B, window, assets, features) == obs_space_dimension  
        - action (np.array) : 현 시점 state 하에서 선택한 액션
        - reward (float)
        - new_state (tuple(np.array, np.array)) : (pre_obs, obs) 환경 하에서 관찰한 다음 시점의 state
            - pre_obs : (B, window, assets)
            - obs : (B, window, assets, features) == obs_space_dimension              
        - done (bool) :  whether one has reached the horizion or not

        Returns : none
        '''
    
        # if not isinstance(state[0], np.ndarray):
        #     state = (state[0].cpu().numpy(), state[1].cpu().numpy())  
        #     new_state = (new_state[0].cpu().numpy(), new_state[1].cpu().numpy())
        
        assert type(action) in (torch.tensor, torch.Tensor), "action data type should be torch.tensor but {} .".format(type(action))
        index = self.pointer % self.size # 나머지 = 새로운 버퍼의 인덱스

        # Give the priority to the samples that have not learned.
        max_prio = self.priorities.max() if self.action_ratio_buffer.sum()!=0 else 1.0  # gives max priority if buffer is not empty else 1
        
        # push the trajactory data into the buffer
        self.state_buffer['obs'][index], self.state_buffer['pre_obs'][index] = state
        self.action_ratio_buffer[index] = action
        self.reward_buffer[index] = reward
        self.new_state_buffer['obs'][index], self.new_state_buffer['pre_obs'][index] = new_state
        self.done_buffer[index] =  done
        self.priorities[index] = max_prio
        
        self.pointer +=1
        assert tuple(self.state_buffer['obs'].shape) == (self.size, *self.input_shape), "Replay Buffer Error"
        self._update_priorities()

    def sample(self, 
                batch_size: int = 32
                ) :
        '''
        Sample a batch of data from the buffer.

        Args:
        - batch_size (int) : 배치사이즈 
        
        Returns : a tuple of np.array of memories, 
                one memory being of the form (state, action, reward, state_, done)
                -> Tuple[S:tuple(np.ndarray), A:np.ndarray, R:np.ndarray, S_:tuple(np.ndarray), Done:np.ndarray]
        '''
        size = min(self.pointer, self.size)
        priorities = self.priorities[:size]
        probabilities = priorities ** self.alpha / torch.sum(priorities  ** self.alpha)
        batch = np.random.choice(size, batch_size, p=probabilities.detach().cpu().numpy()) # 랜덤으로 배치 고르기 ##
        
        # for PER update : weights size -> rank PER?
        weights = (self.size * probabilities[batch]) ** (-self.beta)
        weights /= torch.max(weights) # shape? (# of batch,)
        
        # sample로서 뽑힌 batch로 구성된 SARS, done
        states = (self.state_buffer['obs'][batch], self.state_buffer['pre_obs'][batch])
        actions = self.action_ratio_buffer[batch]
        rewards = self.reward_buffer[batch]
        new_states = (self.new_state_buffer['obs'][batch], self.new_state_buffer['pre_obs'][batch])
        dones = self.done_buffer[batch]
        
        return states, actions, rewards, new_states, dones, weights
    

    
    
