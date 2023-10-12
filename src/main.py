'''
action += action
'''
# 항상 base_dir에서 실행할 것
# base_dir = '/home/ubuntu2010/바탕화면/DEV/trading14'
# base_dir = '/nas3/hwang/trading14'
#/home1/hwang/trading14

from argparse import ArgumentParser
import json
import numpy as np
import os 
import gc
import copy
import time
import torch
from torchinfo import summary

from util.configures import *
from model.agent.SAC import SACagent
from model.agent.DDPG import DDPGagent
from model.agent.Traditional import *
from env.environment import PortfolioEnv
from util.plot import Plot, PortfolioEvaluator
from util.utilities import *
from run import Run, Run_nograd
os.environ['CUDA_LAUCH_BLOCKING'] = "1"
base_dir = os.getcwd() 



def main(args):
    '''
    정민이 말대로 nn.DataParallel
    Device id를 직접 할당해주면 오류
    '''
    # set GPU devices
    gpu_devices = ",".join([str(id) for id in args.gpu_devices])
    device = torch.device(f'cuda:{gpu_devices}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device()) # check

    # initializing the random seeds for reproducibility
    set_seed(args.seed)

    # creating all the necessary directory tree structure for efficient logging
    checkpoint_directory = create_directory_tree(mode=args.mode,
                                                country=args.country,
                                                ffactor=args.ffactor, 
                                                agent_type=args.agent_type,
                                                buffer=args.buffer,
                                                experimental=args.experimental,
                                                checkpoint_directory=args.checkpoint_directory)
    
    checkpoint_directory_networks = os.path.join(checkpoint_directory, 'networks')
    print("\n*----------------------------------------------------------------*\n")
    print("checkpoint_directory: ", checkpoint_directory)
    print("\n*----------------------------------------------------------------*\n")
    
    # saving the (hyper)parameters used for future reference
    params_dict = vars(args) 
    with open(os.path.join(checkpoint_directory, args.mode+"_parameters.json"), "w") as f:
        json.dump(params_dict, f, indent=4)


    ################################# ENV설정 #################################
    ticker_file = f'{BASE_DIR}/portfolio_info/{args.country}/snp_portfolio_smb.json'
    with open(ticker_file, 'r') as f:
        portfolio_info = json.load(f)
        stocks_subset = list(portfolio_info["market_cap"].keys())
        f.close()
    
    stocks_subset.sort()
    assets_subset = stocks_subset + ['cash']

    
    # preparing the initial portfolio to pass it to the constructor of the environment
    initial_portfolio = prepare_initial_portfolio(initial_portfolio=portfolio_info["initial_portfolio"],
                                                tickers=stocks_subset) ###
    print("-"*20, '\n initial_portfolio \n', initial_portfolio, '\n', "-"*20)


    env = PortfolioEnv(
                    initial_portfolio=initial_portfolio, 
                    device=device,
                    args=args,
                    checkpoint_directory = checkpoint_directory) 
    time.sleep(1) # 없으면 가끔 오류일어남


    if args.agent_type == "DDPG":
        agent = DDPGagent(
                        env = env,
                        factor = args.ffactor,
                        buffer = args.buffer,
                        window = args.window,
                        input_shape = env.observation_space_dimension,
                        stocks_subsets = stocks_subset,
                        country = args.country,
                        lr_Q = args.lr_Q,
                        lr_pi = args.lr_pi,
                        lr_alpha = args.lr_alpha,
                        tau = args.tau,
                        size = args.memory_size,
                        batch_size = args.batch_size,
                        layer_size = args.layer_size,
                        delay = args.delay,
                        grad_clip = args.grad_clip,
                        mode = args.mode,
                        checkpoint_directory_networks = checkpoint_directory_networks,
                        device = device)
        
    else :
        agent = SACagent(
                        env = env,
                        factor = args.ffactor,
                        buffer = args.buffer,
                        window = args.window,
                        input_shape = env.observation_space_dimension,
                        stocks_subsets = stocks_subset,
                        country = args.country,
                        lr_Q = args.lr_Q,
                        lr_pi = args.lr_pi,
                        lr_alpha = args.lr_alpha,
                        tau = args.tau,
                        size = args.memory_size,
                        batch_size = args.batch_size,
                        layer_size = args.layer_size,
                        delay = args.delay,
                        grad_clip = args.grad_clip,
                        mode = args.mode,
                        checkpoint_directory_networks = checkpoint_directory_networks,
                        device = device)
        
    
    print("Setting Env and Agent done!")
    print("-------------------------------------------------------------------------")

    # Save model info
    actions = torch.autograd.Variable(torch.rand(env.action_space_dimension)).to(device)
    pre_state = torch.autograd.Variable(torch.rand(1,len(stocks_subset))).to(device)
    state = torch.autograd.Variable(torch.rand(1,*env.observation_space_dimension)).to(device)

    f = open(os.path.join(checkpoint_directory, f"{args.agent_type}_model_params.txt"), 'w') #os.path.join(checkpoint_directory, args.mode+"_parameters.json")
    f.write(count_model_params(agent.actor, "agent_actor_net"))
    f.write(count_model_params(agent.critic, "agent_critic_net"))
    f.write(count_model_params(env.pretrain_net_dict[stocks_subset[0]], "Pretrain_net_per_asset"))
    f.write("\n -----------------------------------------------------------------------------  \n\n")
    f.write(str(agent.actor)+'\n \n')
    f.write(str(summary(agent.actor, [tuple(state.shape), tuple(pre_state.shape)])))
    f.write("\n --------------------------------- c  --------------------------------------------  \n\n")
    f.write(str(agent.critic)+'\n')
    f.write(str(summary(agent.critic, [tuple(state.shape), tuple(pre_state.shape), tuple(actions.shape)])))
    f.write("\n -----------------------------------------------------------------------------  \n\n")
    f.close()

    # running the whole training or testing process
    run = Run(env=env,
            agent=agent,
            n_episodes=args.n_episodes,
            agent_type=args.agent_type,
            checkpoint_directory=checkpoint_directory,
            mode=args.mode,
            asset_label=assets_subset,
            sac_temperature=args.sac_temperature,
            window = args.window,
            )
    
    print("Initiating Run module is done! ")
    print("-------------------------------------------------------------------------")
    
    # # running the training or testing, saving plots
    initial_time = time.time()
    run.run()
    final_time = time.time()
    
    print("-------------------------------------------------------------------------")

    if args.plot:
        run.logger.generate_plots(asset_label = assets_subset)  



    ################################# BACKTEST FOR TEST MODE #################################
    if args.mode == 'test':
        # 망했다 코드 점점 더러워지는데 ㅋㅋ
        model_logger = copy.copy(run.logger)
        
        # delete the not used data - 저장공간때문에 ㅠㅠ
        del agent
        del run
        del actions
        del pre_state
        del state
        gc.collect()

        # Agents for Traditional Portfolio Strategies
        Buyhold_agent = BuyAndHoldAgent(
                        env = env,
                        window = args.window,
                        input_shape = env.observation_space_dimension,
                        stocks_subsets = stocks_subset,
                        country = args.country,
                        lr_Q = args.lr_Q,
                        lr_pi = args.lr_pi,
                        lr_alpha = args.lr_alpha,
                        tau = args.tau,
                        batch_size = 1,
                        checkpoint_directory_networks = checkpoint_directory_networks,
                        device = device
                        )
        
        Equally_agent = EWAgent(
                        env = env,
                        window = args.window,
                        input_shape = env.observation_space_dimension,
                        stocks_subsets = stocks_subset,
                        country = args.country,
                        lr_Q = args.lr_Q,
                        lr_pi = args.lr_pi,
                        lr_alpha = args.lr_alpha,
                        tau = args.tau,
                        batch_size = 1,
                        checkpoint_directory_networks = checkpoint_directory_networks,
                        device = device
                        )
        
        Security_agent = SecurityAgent( 
                            env = env,
                            window = args.window,
                            input_shape = env.observation_space_dimension,
                            stocks_subsets = stocks_subset,
                            country = args.country,
                            lr_Q = args.lr_Q,
                            lr_pi = args.lr_pi,
                            lr_alpha = args.lr_alpha,
                            tau = args.tau,
                            batch_size = 1,
                            checkpoint_directory_networks = checkpoint_directory_networks,
                            device = device
                            )
        
        
        # Define Run Modules for each traditional agent.
        Buyhold_run = Run_nograd(env=env,
                        agent=Buyhold_agent,
                        n_episodes=1,
                        checkpoint_directory=args.checkpoint_directory+"/BuyandHold",
                        mode='test',
                        asset_label=assets_subset,
                        window=args.window)        

        Equally_run = Run_nograd(env=env,
                        agent=Equally_agent,
                        n_episodes=1,
                        checkpoint_directory=args.checkpoint_directory+"/Equally_weighted",
                        mode='test',
                        asset_label=assets_subset,
                        window=args.window)
        
        Security_run = Run_nograd(env=env,
                            agent=Security_agent,
                            n_episodes=1,
                            checkpoint_directory=args.checkpoint_directory+"/Security",
                            mode='test',
                            asset_label=assets_subset,
                            window=args.window)
        
        # Run the agents
        print("\n------------------------- Buy And Hold Portfolio ------------------------")
        Buyhold_run.run()
        Buyhold_logger = Buyhold_run.logger
        print("\n----------------------- Equally Weighted Portfolio ----------------------")
        Equally_run.run()
        equally_logger = Equally_run.logger
        print("\n--------------------- Security (Only Cash) Portfolio --------------------")
        Security_run.run()
        security_logger = Security_run.logger
        print("-------------------------------------------------------------------------\n")



        #----------------------------------- draw the figures -----------------------------------
        
        # 여기에 기존 전략이랑 비교하는 그래프 넣기
        portfolio_dict = {'equally':equally_logger.logs['portfolio_value_timeseries'][-1], 
                            'security':security_logger.logs['portfolio_value_timeseries'][-1],
                            'buyhold':Buyhold_logger.logs['portfolio_value_timeseries'][-1],
                            'Ours': np.array(model_logger.logs['portfolio_value_timeseries'])[-1]}
        
        #각 자산의 action movement
        Plot.action_movement(figure_file = os.path.join(model_logger.checkpoint_directory_plots, args.mode+"_actionmove"),
                        stock_ts = env.close_p.detach().cpu().numpy(),
                        asset_ts = np.array(model_logger.logs['asset_ratio_timeseries']), 
                        asset_label = assets_subset,
                        date = env.dates,
                        seed = args.seed
                        )
        
        # 각 feature의 어텐션 히트맵 
        # feature_name = [f'P_{i}' for i in assets_subset] + \
        #                 [f'R_{i}' for i in assets_subset] + env.factor_name
        feature_name = ['Open', 'High', 'Low', 'Close', 'Volume', 'Expected Return']+env.factor_name
       
        Plot.attention_heatmap(attention_map = model_logger.logs['attention_timeseries'][-1], # np.array
                                feature_name = feature_name,
                                action_dim = len(assets_subset),
                                date = env.dates,
                                seed = args.seed,
                                figure_file = os.path.join(model_logger.checkpoint_directory_plots, args.mode+"_AttentionMap"))

        
        
        # 백테스팅 결과
        Plot.backtest(figure_file = os.path.join(model_logger.checkpoint_directory_plots, "backtest"),
                        portfolio_dict = portfolio_dict,
                        date = env.dates,
                        seed = args.seed)
        
        ###---- 포트폴리오 수치 결과 계산 
        evaluator = PortfolioEvaluator(checkpoint_directory=model_logger.checkpoint_directory_plots,
                                portfolio_dict=portfolio_dict,
                                date=env.dates,
                                seed=args.seed)
        df = evaluator()
        print(df)
        print("\n-------------------------------------------------------------------------\n")
        

    print('\nTotal {}ing duration: {:*^13.3f} \n'.format(args.mode, final_time-initial_time))
    print('\nCheckpoint Directory : {}'.format(checkpoint_directory))





if __name__=='__main__':
    parser = ArgumentParser()

    # parameters defining the trading environment    
    parser.add_argument('--ffactor',           type=str,   default='default',            help='Factors : size, value, vol, default(기본 리워드) ')
    parser.add_argument('--country',           type=str,   default='NASDAQ',         help='USA, NASDAQ, DJIA')
    parser.add_argument('--stage',             type=str,   default='factor',         help='factor, manager')
    parser.add_argument('--buffer',            type=str,   default='BASIC',            help='BASIC, PER')

    parser.add_argument('--buy_rule',          type=str,   default='most_first',     help="In which order to buy the share: 'most_first' or 'cyclic' or 'random'")
    parser.add_argument('--trade_cost',        type=float, default=0.00015,          help='Cost for buying a share, prorata of the quantity being bought')
    parser.add_argument('--initial_date',      type=str,   default='2002-09-20',     help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--final_date',        type=str,   default='2018-12-08',     help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")    
    parser.add_argument('--test_start',        type=str,   default='2018-12-09',     help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--test_end',          type=str,   default='2022-12-31',     help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")
    
    # type of agent
    parser.add_argument('--agent_type',        type=str,   default='DDPG',            help="Type of agent: 'SAC' or 'DDPG'")
    parser.add_argument('--sac_temperature',   type=float, default=1.0,               help="Coefficient of the entropy term in the loss function in case 'manual_temperature' agent is used")
    
    # hyperparameters for the RL training process
    parser.add_argument('--gamma',             type=float, default=0.99,             help='Discount factor in the definition of the return')
    parser.add_argument('--lr_Q',              type=float, default=1e-5,             help='Learning rate for the critic networks')
    parser.add_argument('--lr_pi',             type=float, default=1e-6,             help='Learning rate for the actor networks')
    parser.add_argument('--lr_alpha',          type=float, default=1e-4,             help='Learning rate for the automatic temperature optimization')
    parser.add_argument('--tau',               type=float, default=0.1,              help='Hyperparameter for the smooth copy of the various networks to their target version')
    parser.add_argument('--batch_size',        type=int,   default=64,               help='Batch size when sampling from the replay buffer')
    parser.add_argument('--memory_size',       type=int,   default=1000,           help='Size of the replay buffer, memory of the agent')
    parser.add_argument('--grad_clip',         type=float, default=1.0,              help='Bound in case one decides to use gradient clipping in the training process')
    parser.add_argument('--delay',             type=int,   default=7,                help='Delay between training of critic and actor')    
    parser.add_argument('--pretrain',          type=int,   default=1,                help='if 1 True, else 0 pretrain False')    


    # hyperparameters for the architectures
    parser.add_argument('--layer_size',        type=int,   default=16,               help='Number of neurons in the various hidden layers')
    
    # Number of training or testing episodes and mode
    parser.add_argument('--n_episodes',   type=int,  default=5,         help='Number of training or testing episodes')
    parser.add_argument('--mode',         type=str,  default='train',   help="Mode used: 'train' or 'test'")
    parser.add_argument('--experimental', type=int,  default=1,         help='if 1, Saves all outputs in an overwritten directory, used simple experiments and tuning')
    
    # random seed, logs information and hardware
    parser.add_argument('--checkpoint_directory', type=str,             default=base_dir,    help='In test mode, specify the directory in which to find the weights of the trained networks')
    parser.add_argument('--plot',                 action='store_true',  default=True,        help='Whether to automatically generate plots or not')
    parser.add_argument('--seed',                 type=int,             default='505',        help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices',          type=int,  nargs='+', default=[3],         help='Specify the GPUs if any')
    parser.add_argument('--prewindow',            type=int,             default=60,          help='# of Pretrain Model(rt) Sliding Window')
    parser.add_argument('--window',               type=int,             default=20,          help='# of Pt Sliding Window')
    parser.add_argument('--memo',                 type=str,             default=None,        help='short info about this experiments')
    

    args = parser.parse_args()
    assert args.ffactor in ['default', 'size', 'value', 'vol'], f"Wrong args.ffactor : {args.ffactor}"
    args.country = args.country.upper()
    args.buffer = args.buffer.upper()
    args.experimental = True if args.experimental==1 else False #0,1
    args.pretrain = True if args.pretrain == 1 else False #0,1
    args.memory_size = int(args.memory_size)
    
    
    # for debugging option
    if args.mode == 'test' and args.checkpoint_directory == base_dir:
        args.checkpoint_directory = "/home1/hwang/trading2/saved_outputs/NASDAQ/default_BASIC/2023.08.17.11.03_DDPG"
        args.n_episodes = 1
        args.experimental = False
        args.initial_date = args.test_start
        args.final_date = args.test_end
        
    
    main(args) 