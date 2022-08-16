

from multiprocessing.spawn import prepare
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.insert(0, '../src')
sys.path.insert(0, '/home/ubuntu2010/바탕화면/DEV/trading7')
sys.path.insert(0, '/home/ubuntu2010/바탕화면/DEV/trading7/src')
# print(sys.path)

from argparse import ArgumentParser
import json
import numpy as np
import os 
import time
import torch

from src.agents import instanciate_agent
from src.environment import Environment
from src.get_data import load_data
from src.run import Run
from src.utilities import create_directory_tree, instanciate_scaler, prepare_initial_portfolio


def main(args):

    # 사용할 hardware 결정해주기
    # print([id for id in args.gpu_devices])
    gpu_devices = ",".join([str(id) for id in args.gpu_devices])
    print("gpu_devices : ", gpu_devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # initializing the random seeds for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # creating all the necessary directory tree structure for efficient logging
    checkpoint_directory = create_directory_tree(mode=args.mode,
                                            experimental = args.experimental,
                                            checkpoint_directory=args.checkpoint_directory)
    
    # saving the (hyper)parameters used for future reference
    # vars : __dict__ 속성을 반환, 객체의 변경 가능한 속성을 포함하는 딕셔너리
    # 매개변수 없이 "vars()" 함수 호출 시, 로컬 객체 테이블이 포함된 딕셔너리 반환
    params_dict = vars(args) 

    with open(os.path.join(checkpoint_directory, args.mode+"_parameters.json"), "w") as f:
        json.dump(params_dict, f, indent=4)
    
    # multidimensional 시계열 데이터프레임을 다운로드, 전처리 & 로드 
    df = load_data(initial_date=args.initial_date,
                    final_date=args.final_date,
                    tickers_subset=args.assets_to_trade,
                    mode=args.mode)

    if args.mode == 'test' and args.initial_date is not None and args.final_date is not None:
        df = df.loc[args.initial_date : args.final_date]
        dates = df.index
    
    print("=================================")
    print(df.head(5))

    # preparing the initial portfolio to pass it to the constructor of the environment
    # 환경의 생성자에게 전달할 초기 포트폴리오 준비
    initial_portfolio = prepare_initial_portfolio(initial_portfolio=args.initial_cash if args.initial_cash is not None \
                                                    else args.initial_portfolio,
                                                    tickers=df.columns.to_list())
    # 트레이딩 환경 인스턴스화
    env = Environment(stock_market_history=df,
                    initial_portfolio=initial_portfolio,
                    buy_cost=args.buy_cost,
                    sell_cost=args.sell_cost,
                    bank_rate=args.bank_rate,
                    limit_n_stocks=args.limit_n_stocks,
                    buy_rule=args.buy_rule,
                    use_corr_matrix=args.use_corr_matrix,
                    use_corr_eigenvalues=args.use_corr_eigenvalues,
                    window=args.window,
                    number_of_eigenvalues=args.number_of_eigenvalues)
    
    print(df)
    exit()

    # instanciating the data standard scaler
    scaler = instanciate_scaler(env=env,
                                mode=args.mode,
                                checkpoint_directory=checkpoint_directory)

    # instanciating the trading agent
    agent = instanciate_agent(env=env,
                                device=device,
                                checkpoint_directory=checkpoint_directory,
                                args=args)
    
    # running the whole training or testing process
    run = Run(env=env,
            agent=agent,
            n_episodes=args.n_episodes,
            agent_type=args.agent_type,
            checkpoint_directory=checkpoint_directory,
            mode = args.mode,
            sac_temperature=args.sac_temperature,
            scaler=scaler)
    

    # running the training or testing, saving plots
    initial_time = time.time()
    run.run()

    if args.plot:
        run.logger.generate_plots()
    final_time = time.time()
    print('\nTotal {}ing duration: {:*^13.3f} \n'.format(args.mode, final_time-initial_time))


if __name__=='__main__':
    parser = ArgumentParser()

    # parameters defining the trading environment
    group1 = parser.add_mutually_exclusive_group() #그룹을 만들고, 해당 그룹에 추가된 매개변수는 단 하나만 선택해야 한다.
    group1.add_argument('--initial_cash',      type=float, default=None,         help='Initial cash in the bank, assuming no shares are owned')
    group1.add_argument('--initial_portfolio', type=str,   default='./portfolios_and_tickers/initial_portfolio.json', help='Path to json file containing the content of an initial portfolio, including the cash in bank')
    
    parser.add_argument('--assets_to_trade',   type=str,   default='./portfolios_and_tickers/tickers_S&P500.txt',     help='List of the tickers of the assets to be traded')
    parser.add_argument('--buy_rule',          type=str,   default='most_first', help="In which order to buy the share: 'most_first' or 'cyclic' or 'random'")
    parser.add_argument('--buy_cost',          type=float, default=0.001,        help='Cost for buying a share, prorata of the quantity being bought')
    parser.add_argument('--sell_cost',         type=float, default=0.001,        help='Cost for selling a share, prorata of the quantity being sold')
    parser.add_argument('--bank_rate',         type=float, default=0.5,          help='Annual bank rate')
    parser.add_argument('--initial_date',      type=str,   default='2014-12-31', help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--final_date',        type=str,   default='2020-12-30', help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")
    parser.add_argument('--limit_n_stocks',    type=int,   default=20,           help='Maximal number of shares that can be bought or sell in one trade')
    
    # type of agent
    parser.add_argument('--agent_type',      type=str,   default='distributional', help="Type of agent: 'manual_temperature' or 'automatic_temperature' or 'distributional'")
    parser.add_argument('--sac_temperature', type=float, default=2.0,              help="Coefficient of the entropy term in the loss function in case 'manual_temperature' agent is used")
    
    # hyperparameters for the RL training process
    parser.add_argument('--gamma',       type=float, default=0.99,    help='Discount factor in the definition of the return')
    parser.add_argument('--lr_Q',        type=float, default=0.0003,  help='Learning rate for the critic networks')
    parser.add_argument('--lr_pi',       type=float, default=0.0003,  help='Learning rate for the actor networks')
    parser.add_argument('--lr_alpha',    type=float, default=0.0003,  help='Learning rate for the automatic temperature optimization')
    parser.add_argument('--tau',         type=float, default=0.005,   help='Hyperparameter for the smooth copy of the various networks to their target version')
    parser.add_argument('--batch_size',  type=int,   default=32,      help='Batch size when sampling from the replay buffer')
    parser.add_argument('--memory_size', type=int,   default=100, help='Size of the replay buffer, memory of the agent')
    parser.add_argument('--grad_clip',   type=float, default=1.0,     help='Bound in case one decides to use gradient clipping in the training process')
    parser.add_argument('--delay',       type=int,   default=1,       help='Delay between training of critic and actor')    
    
    # hyperparameters for the architectures
    parser.add_argument('--layer_size', type=int, default=256, help='Number of neurons in the various hidden layers')
    
    # Number of training or testing episodes and mode
    parser.add_argument('--n_episodes',   type=int, required=True, help='Number of training or testing episodes')
    parser.add_argument('--mode',         type=str, required=True, help="Mode used: 'train' or 'test'")
    parser.add_argument('--experimental', action='store_true',     help='Saves all outputs in an overwritten directory, used simple experiments and tuning')
    
    # random seed, logs information and hardware
    parser.add_argument('--checkpoint_directory', type=str,            default=None,         help='In test mode, specify the directory in which to find the weights of the trained networks')
    parser.add_argument('--plot',                 action='store_true', default=False,        help='Whether to automatically generate plots or not')
    parser.add_argument('--seed',                 type=int,            default='42',         help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices',          type=int, nargs='+', default=[0, 1, 2, 3], help='Specify the GPUs if any')

    # parameters concerning data preprocessing
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--use_corr_matrix',       action='store_true', default=False, help='To append the sliding correlation matrix to the time series')
    group2.add_argument('--use_corr_eigenvalues',  action='store_true', default=False, help='To append the eigenvalues of the correlation matrix to the time series')
    
    parser.add_argument('--window',                type=int,            default=20,    help='Window for correlation matrix computation')
    parser.add_argument('--number_of_eigenvalues', type=int,            default=10,    help='Number of largest eigenvalues to append to the close prices time series')
    
    args = parser.parse_args()
    main(args)