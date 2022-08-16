import sys
sys.path.insert(0, './src')

import gym
import numpy as np
import os
import torch
from typing import Tuple

from src.buffer import ReplayBuffer
from src.environment import Environment
from src.networks import Actor, Critic, Value, Distributional_Critic


####################################################################################


'''
reward를 정의하는 클래스 만들기...
이때, 리워드는 각 에이전트 종류마다 다르게 줄건데,
1) rm-rf
2) HML
3) SMB
가 각각 극대화 되게 만들 것이다.

이거 수도 코트 부터 짜야할 듯 싶긴한데,
그러려면 어떤 데이터를 어떻게 변환해서 쓸지도 생각해야함...
'''

class Reward():
    def __init__(self) -> None:
        pass

    def calculate(self):
        raise NotImplementedError

    