import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Union


def create_directory_tree(mode: str,
                          country: str,
                          ffactor: str,
                          agent_type:str, 
                          buffer:str,
                          experimental: bool,
                          checkpoint_directory: str):
    
    date: str = datetime.now().strftime("%Y.%m.%d.%H.%M")

    if experimental == True:
        # for debugging
        checkpoint_directory = os.path.join(checkpoint_directory, f"save_outputs_experimental/{country}/{ffactor}_{buffer}")
    else:
        checkpoint_directory = os.path.join(checkpoint_directory, f"saved_outputs/{country}/{ffactor}_{buffer}/{date}_{agent_type}") if mode =='train' else checkpoint_directory
    
    # create various subdirectories
    checkpoint_directory_networks = os.path.join(checkpoint_directory, "networks")
    checkpoint_directory_logs = os.path.join(checkpoint_directory, "logs")
    checkpoint_directory_plots = os.path.join(checkpoint_directory, "plots")
    checkpoint_directory_json = os.path.join(checkpoint_directory, "json")

    Path(checkpoint_directory_networks).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_logs).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_plots).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_directory_json).mkdir(parents=True, exist_ok=True)

    # 테스트 시 빠른 액세스를 위해 파일에 체크포인트 디렉토리 이름 쓰기
    if mode == 'train':
        with open(os.path.join(checkpoint_directory, "checkpoint_directory.txt"), "w") as f:
            f.write(checkpoint_directory)
    return checkpoint_directory


def prepare_initial_portfolio(initial_portfolio: Union[int, float, str, dict],
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

    # 금액만 주어졌을 때 
    if isinstance(initial_portfolio, int) or isinstance(initial_portfolio, float):
        initial_portfolio_returned = {key: 0 for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio

    # 파일명이 들어가는 경우
    elif isinstance(initial_portfolio, str):
        with open(initial_portfolio, "r") as file:
            initial_portfolio = json.load(file)
        initial_portfolio_returned = {key: 0 for key in tickers}
        initial_portfolio_returned["Bank_account"] = initial_portfolio["Bank_account"]

        for key in initial_portfolio_returned.keys():
            if key in initial_portfolio.keys():
                initial_portfolio_returned[key] = initial_portfolio[key]
    else:
        # dict type이 바로 들어오는 경우
        initial_portfolio_returned = initial_portfolio

    return initial_portfolio_returned


def count_model_params(model, model_name:str):
    # file save need
    logs = f"{model_name} : {sum(p.numel() for p in model.parameters() if p.requires_grad)}  \n"
    print(logs)
    return logs


def set_seed(seed):
    # initializing the random seeds for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            if len(obj.shape)==0:
                return float(obj.item())
            else :
                return obj.tolist()
        return json.JSONEncoder.default(self, obj)


