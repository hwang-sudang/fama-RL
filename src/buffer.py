'''

'''

# from sre_parse import State
from hashlib import new
import numpy as np
from typing import Tuple, List

class ReplayBuffer():
    '''
    Plays the role of memory for Agents,
    by storing (state, action, reward, state_, done) tuples.
    '''

    def __init__(self,
                size: int,
                input_shape: Tuple,
                action_space_dimension: int,
                )-> None:
        
        '''
        Replay bufferdml 생성자.
        
        Args:
        - size(int) : 리플레이 버퍼의 최대 사이즈
        - input shape (tuple): dimension of the observation space 
        - action_space_dimension (int) :  dimension of the action space
        
        Output : none
        '''

        self.size = size
        self.pointer = 0

        self.state_buffer = np.zeros((self.size, *input_shape)) 
        self.new_state_buffer = np.zeros((self.size, *input_shape)) # *input_size : iterable 해야함.
        self.action_buffer = np.zeros((self.size, action_space_dimension))
        self.reward_buffer = np.zeros(self.size)
        self.done_buffer = np.zeros(self.size, dtype = np.bool)


    def push(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            new_state: np.ndarray,
            done: bool,
            )-> None:
        
        '''
        Replay bufferd에 메모리 추가.
        
        Args:
        - state (np.array) : 환경하에서 관찰한 현시점의 state 
        - action (np.array) : 현 시점 state 하에서 선택한 액션
        - reward (float)
        - new_state (np.array)
        - done (bool) :  whether one has reached the horizion or not

        returns : none
        '''

        index = self.pointer % self.size #나머지 = 새로운 버퍼의 인덱스
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.new_state_buffer[index] = new_state
        self.done_buffer[index] =  done

        self.pointer +=1
        pass


    def sample(self, 
                batch_size: int = 32
                ) -> Tuple[np.ndarray, np.array, np.array, np.array, np.array]:
        '''
        Sample a batch of data from the buffer.

        args:
        - batch_size (int) : 배치사이즈 
        
        returns : a tuple of np.array of memories, 
                one memory being of the form (state, action, reward, state_, done)       
        '''

        size = min(self.pointer, self.size)
        batch = np.random.choice(size, batch_size) # 랜덤으로 배치 고르기

        # sample로서 뽑힌 batch로 구성된 sars, done
        states = self.state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        new_states = self.new_state_buffer[batch]
        dones = self.done_buffer[batch]
        
        return states, actions, rewards, new_states, dones

