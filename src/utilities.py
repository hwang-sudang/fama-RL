# conda actibate torchgpu

from datetime import datetime
from tabnanny import check
import gym
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
sns.set_theme()
from sklearn.preprocessing import StandardScaler
from typing import List, Union


def create_directory_tree(mode: str,
                          experimental: bool,
                          checkpoint_directory: str):
    
    date: str = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    if experimental:
        checkpoint_directory = os.path.join("save_outputs", "experimental")
    else:
        checkpoint_directory = os.path.join("saved_outputs", date) if mode =='train' else checkpoint_directory
    
    # create various subdirectories
    checkpoint_directory_networks = os.path.join(checkpoint_directory,"networks")
    checkpoint_directory_logs = os.path.join(checkpoint_directory, "logs")
    checkpoint_directory_plots = os.path.join(checkpoint_directory, "plots")
    Path(checkpoint_directory_networks).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_logs).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_plots).mkdir(parents=True, exist_ok=True)

    # 테스트 시 빠른 액세스를 위해 파일에 체크포인트 디렉토리 이름 쓰기
    if mode == 'train':
        with open(os.path.join(checkpoint_directory, "checkpoint_directory.txt"), "w") as f:
            f.write(checkpoint_directory)
    
    return checkpoint_directory


def plot_reward(x: List[int],
                rewards: np.ndarray,
                figure_file: str,
                mode: str,
                bins : int = 20,
                ) -> None:
    '''
    '''

    running_average = np.zeros(len(rewards))
    for i in range(len(running_average)):
        running_average[i] = np.mean(rewards[max(0, i-50): i+1])
    
    if mode == 'train':
        plt.plot(x, rewards, linestyle='-', color='blue', label='reward')
        plt.plot(x, running_average, linestyle = '--', color='green', label='running average 50' )
        plt.legend()
        plt.title('Reward as a function of the epoch/episode')

    elif mode == 'test':
        plt.hist(rewards, bins=bins)
        plt.title('Reward distribution')
    
    plt.savefig(figure_file)


def plot_portfolio_value(x : List[int],
                        values: np.ndarray,
                        figure_file: str,
                        ) -> None :
    
    plt.plot(x, values.T, linestyle= '-', linewidth = 0.5)
    plt.xlim((0, len(x)))
    plt.title('Portfolio value')
    plt.savefig(figure_file)


def instanciate_scaler(env: gym.Env, mode: str,
                        checkpoint_directory: str) -> StandardScaler:
    
    """   
    인스턴스화하고 모드에 따라 파라미터를 맞추거나 로드.

    학습모드 에이전트 : 에이전트는 환경에서 비롯되는 obs를 저장하기 위해 무작위로 동작
    랜덤에이전트가 더 정확한 스케일러를 위해 10개의 에피소드 시행(Trade)

    Args:
        env (gym.Env): trading environment
        mode (mode) : train or test 

    Returns : 학습된 sklearn standard scaler
    """

    # random agent
    scaler = StandardScaler()

    if mode == 'train':
        observations = []
        for _ in range(10):
            observation = env.reset()
            observations.append(observation)
            done = False
            while not done:
                action = env.action_space.sample()
                observation_, _, done, _ = env.step(action)
                observations.append(observation_)

        scaler.fit(observations)

        with open(os.path.join(checkpoint_directory, 'networks', 'sclaer.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    
    if mode=='test':
        with open(os.path.join(checkpoint_directory, 'networks', 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    
    return scaler


def prepare_initial_portfolio(initial_portfolio: Union[int, float, str],
                              tickers: List[str]) -> dict:
    """
    환경 구축에 필요한 초기 포트폴리오 준비용.
    Args(Input) : 
        initial_portfolio (int, float, string)
        - 숫자 : 초기에 소유한 주식이 없다고 가정, 은행의 초기현금으로 지정
        - str타입 : 은행의 초기 현금과 각 자산의 소유 주식 수를 나타낸 json 파일 경로

        tickers (List[str]): 에셋 이름의 리스트
    
    Returns : 
        dictionary giving the structure of the initial portfolio
        초기 포트폴리오 정보를 갖고 있는 딕셔너리 구조
    """

    print('>>>>> Reading the provided initial portfolio <<<<<')

    if isinstance(initial_portfolio, int) or isinstance(initial_portfolio, float):
        initial_portfolio_returned = {key: 0 for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio

    else:
        with open(initial_portfolio, "r") as file:
            initial_portfolio = json.load(file)
        
        initial_portfolio_returned = {key: 0  for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio["Bank_account"]

        for key in initial_portfolio_returned.keys():
            if key in initial_portfolio.keys():
                initial_portfolio_returned[key] = initial_portfolio[key]
    
    
    return initial_portfolio_returned


def append_corr_matrix(df: pd.DataFrame, window: int,) -> pd.DataFrame :
    """
    다차원 시계열의 슬라이딩 상관 행렬 추가
    timewise : 상관행렬을 평평하게 만들고, 위쪽 삼각형 부분만 추출, 초기 시계열에 추가함

    Args:
        df (pd.DataFrame): 다차원 시계열의 슬라이딩 상관행렬
        window (int): 상관행렬을 계산하기 위해 사용된 슬라이딩 윈도우 사이즈

    Returns:
        슬라이딩 상관행렬이 추가된 입력 시계열, 데이터프레임
    """

    print('>>>>> Appending the correlation matrix <<<<<')

    columns = ['{}/{}'.format(m,n) for (m,n) in itertools.combinations_with_replacement(df.columns, r=2)]
    corr = df.rolling(window).cov()
    corr_flattened = pd.DataFrame(index=columns).transpose()

    for i in range(df.shape[0]):
        ind = np.triu_indices(df.shape[1]) # 상관행렬 인덱싱
        data = corr[df.shape[1]*i : df.shape[1]*(i+1)].to_numpy()[ind]
        index = [corr.index[df.shape[1]*i][0]]

        temp = pd.DataFrame(data=data, columns=index, index=columns).transpose()
        corr_flattened = pd.concat([corr_flattened , temp])

    return pd.concat([df, corr_flattened], axis=1).iloc[window-1 : ]


def append_corr_matrix_eigenvalues(df: pd.DataFrame,
                                    window: int,
                                    number_of_eigenvalues: int = 10) -> pd.DataFrame:
    """
    다차원 시계열의 슬라이딩 상관행렬의 number_of_eigenvalues의 최대 고유값을 추가한다.

    Args:
        df (pd.DataFrame): 다차원 시계열의 슬라이딩 상관행렬
        window (int): 상관행렬을 계산하기 위해 사용된 슬라이딩 윈도우 사이즈

    Returns :
        슬라이딩 상관행렬의 number_of_eigenvalues의 최대 고유값이 추가된 입력 시계열

    """
    print('>>>>> Appending the eigenvalues <<<<<')

    if number_of_eigenvalues > df.shape[1]:
        number_of_eigenvalues = df.shape[1]
    
    columns = ['Eigenvalue_{}'.format(m+1) for m in range(number_of_eigenvalues)]
    corr = df.rolling(window).cov()
    corr_eigenvalues = pd.DataFrame(index=columns).transpose()

    for i in range(window-1, df.shape[0]):
        data = corr[df.shape[1]*i : df.shape[1]*(i+1)].to_numpy()

