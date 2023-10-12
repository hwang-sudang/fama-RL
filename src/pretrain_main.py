
import os
import wandb
import time
import numpy as np
import pandas as pd
import datetime
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from torchmetrics.classification import BinaryF1Score, Accuracy, BinaryPrecision, BinaryRecall

# inner module import
# from src.networks.actor import *
from actor_pretrain.policy import *
from model.networks.pretrain_net import *
from util.utilities import *
from util.configures import *
from actor_pretrain.utils_pretrain import PretrainDataset, PretrainUtilites
from actor_pretrain.pretrain_config import sweep_configuration



# make folders to save the logs
PRETRAIN_DIR = os.getcwd()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 이거 나중에 실제 RL이랑 맞추기
def load_data(args):
    print("--------------------------------- LOAD DATA -------------------------------\n")
    # # 데이터 로드, 전처리
    # # 오직 Close만
    # 1. data load (이 단계는 밖에서 이루어져야함.)
    pretrain_datadir = f'{DATA_DIR}/stockdata/label_data/{args.country}/pretrain' #
    ticker_list = [i.split("_")[0] for i in os.listdir(pretrain_datadir) if i.endswith(".csv")]
    
    # 자산별로 실험 데이터 셋 만들기
    asset_datasets = {}
    for tic in ticker_list:
        dataset = pd.read_csv(pretrain_datadir+f"/{tic}_pretrain_dataset.csv", parse_dates=['date'], index_col='date')
        dataset = dataset.sort_index().fillna(method='ffill')
        asset_datasets[tic] = dataset
    return asset_datasets



def tvt_split(dataset, window, log_scale:int=1):
    # tvt 나누기
    train_df, val_df, test_df = PretrainUtilites.data_split(dataset,START_DATE,TRAIN_END,VAL_END,TEST_END,window=args.window) ## 여기가 문젠데
    
    # check there's any nan data
    print("train_df number of nan  :", train_df.isnull().sum().sum())
    print("val_df number of nan  :", val_df.isnull().sum().sum())
    print("test_df number of nan  :", test_df.isnull().sum().sum())

    trainset = PretrainDataset(train_df, STOCK_SPACE_DIMENSION, window, log_scale=log_scale, emd_mode=True)
    valset = PretrainDataset(val_df, STOCK_SPACE_DIMENSION, window, log_scale=log_scale, emd_mode=True)
    testset = PretrainDataset(test_df, STOCK_SPACE_DIMENSION, window, log_scale=log_scale, emd_mode=True)
    return trainset, valset, testset



def testPerformance(a_name, model, test_loader, device, args):
    '''
    여기 인자를 어떻게 받아서 함수 활용할 것인지 생각해보기
    '''
    #print(f"--------------------------------- START TO TEST MODEL  -------------------------------\n")

    model.eval()
    mseloss = 0.0
    test_result = []
    
    with torch.no_grad():
        for states, price, label in test_loader:
            states = Variable(states.to(device))
            label = Variable(label.to(device))
            price = Variable(price.to(device))

            # run the model on the test set to predict label
            outputs = model.forward(states)

            # 모델 로스 측정
            loss = nn.functional.mse_loss(outputs, label)
            mseloss += loss.item()

            # mae, rmse, mape 등 측정
            pred_p = torch.exp(outputs).mul(price)
            label_p = torch.exp(label).mul(price)
            mae = mean_absolute_error(pred_p, label_p).item()
            rmse = mean_squared_error(pred_p, label_p).item()**0.5
            mape = mean_absolute_percentage_error(pred_p, label_p).item()
            r2 = r2_score(pred_p, label_p).item()
            test_result.append({"MAE": mae, "RMSE": rmse, "MAPE":mape, "R2":r2})


    result_df = pd.DataFrame(test_result)
    rmse_ = result_df.mean().RMSE
    mae_ = result_df.mean().MAE
    mape_ = result_df.mean().MAPE
    r2_ = result_df.mean().R2

    # print("------------------------------------------------------------------------------------------")
    print("Final Test loss \n  MAE: %.5f, RMSE: %.5f, MAPE: %.5f, R2: %.5f  \n" % (mae_, rmse_, mape_, r2_))
    # result_df.to_csv(f'{PRETRAIN_DIR}/critic_learning_df_%s.csv' % (DATE_TIME), index_label=False)
    
    return mseloss/len(test_loader), [mae_, rmse_, mape_, r2_]


def prediction_plot(a_name, model, dataloader, y_data, device, args):
    '''
    train 부분은 안해도 되나?
    '''
    # 여기서 제일 학습 잘된 모델 웨이트를 갖고오자..
    w = args.window
    y_data = y_data[y_data.index<=TEST_END]
    graph_pred, graph_label = [], []
    train_loader, val_loader, test_loader = dataloader
    train_len, val_len, test_len = len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
    whole_len = train_len+val_len+test_len #161개??????
    
    model.eval()
    with torch.no_grad():
        # 코드가 드럽긴하지만.. 귀찮음
        for loader in (train_loader, val_loader, test_loader):
            for states, price, label in loader:
                states = Variable(states.to(device))
                label = Variable(label.to(device))
                price = Variable(price.to(device))
                outputs = model.forward(states)

                pred_p = torch.exp(outputs).mul(price)
                label_p = torch.exp(label).mul(price)

                graph_label.extend(label_p.squeeze().cpu().tolist())
                graph_pred.extend(pred_p.squeeze().cpu().tolist())

    graph_df = y_data[args.window:].copy().reset_index() #-pred_len # date, close  포함하는 지 확인
    graph_df.columns = ['date', 'label']
    graph_df["pred"] = [np.nan]*len(graph_df)


    # temp_df = pd.DataFrame({"pred":graph_pred, "label":graph_label}) # 5620
    graph_df["pred"][:train_len] = graph_pred[:train_len].copy()
    graph_df["pred"][train_len+w : train_len+w+val_len] = graph_pred[train_len : train_len+val_len].copy() 
    graph_df["pred"][train_len+val_len+w*2:] = graph_pred[train_len+val_len:].copy() # 1137
    graph_df.to_csv(f'{PRETRAIN_SAVEDIR}/fig/prediction_{a_name}_result.csv', index_label=False)

    
    plt.figure(figsize=(10,5))
    ax = plt.gca()
    #y_data = y_data[y_data.index<=TEST_END]
    ax.plot(graph_df['date'],graph_df[['label']], c='black', label="Real") # answer
    ax.plot(graph_df['date'][:train_len], graph_df['pred'][:train_len], c='blue', label="Train") # train
    ax.plot(graph_df['date'][train_len:train_len+val_len+w], graph_df['pred'][train_len:train_len+val_len+w], c='green', label="Val") # val 
    ax.plot(graph_df['date'][train_len+val_len+2*w:], graph_df['pred'][train_len+val_len+2*w:], c='red', label="Test") # test

    ax.legend(loc='best')
    plt.xticks(rotation=20)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1,12,24)))

    plt.savefig(f"{PRETRAIN_SAVEDIR}/fig/prediction_{a_name}_{args.task}.png")
    plt.close()
    time.sleep(0.5)




def train(a_name, model, dataloader, device, args):
    '''
    Args
    - model : actor or pretrain --> input 고민 
    - dataloader(set of DataLoader) : (train_loader, val_loader, test_loader)
    - num epochs : 학습시킬 횟수
    - args..
    '''

    train_loader, val_loader, test_loader = dataloader
    input_size = train_loader.dataset.input_size()

    ## Model define & set the optimizer, loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pi, weight_decay=1e-6) #, weight_decay=0.0001
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr_pi, weight_decay=1e-6) 
    # optimizer = optim.SGD(model.parameters(), lr=args.lr_pi, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    best_accuaracy = float("inf") 

    # set lr_scheduler
    if args.lr_scheduler=='cosw':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=0.00001)
    elif args.lr_scheduler=='reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-6)
    else:
        pass

    #---------------------------------------- TRAINING -------------------------------------
    # START TRAIN
    train_metric_lst = []
    train_loss_lst = []
    val_loss_lst = []
    metric_lst = []
    

    for epoch in range(args.n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        val_loss = 0.0 

        for i, (state, price, label) in enumerate(train_loader, 0):
            autograd.set_detect_anomaly(True)
            model.train()
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                state = Variable(state.to(device), requires_grad=True)
                price = Variable(price.to(device))
                label = Variable(label.to(device))
                outputs = model.forward(state)
                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step() 

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # print(name, param.grad.sum())
                        pass
                    else:
                        print(name, param.grad)
            running_loss += loss.item()

            # calculate metrics while training the model  
            pred_p = torch.exp(outputs).mul(price) #64,64
            label_p = torch.exp(label).mul(price)
            mae = mean_absolute_error(pred_p, label_p).item()
            rmse = mean_squared_error(pred_p, label_p).item()**0.5
            mape = mean_absolute_percentage_error(pred_p, label_p).item()
            r2 = r2_score(pred_p, label_p).item()


        
        if epoch % 30==0 or epoch==args.n_epochs-1:
            print('%d epoch loss: %.3f' % (epoch, running_loss/args.batch_size))
            print('%d epoch Train: MAE %.3f  RMSE %.3f  MAPE %.3f' % (epoch, mae, rmse, mape))

            print("\n---------------------------------------------------------------------------------------\n")
            print("Epoch {} : Start Validation ... ".format(epoch))
            val_loss, _ = testPerformance(a_name, model, val_loader, device, args)
            print('Validation For epoch', epoch+1, ' the test acc loss over the whole test set is %.5f ' % (val_loss) )
            print("\n---------------------------------------------------------------------------------------\n")

            # batch 평균 loss 값
            train_loss_lst.append(running_loss/args.batch_size) #배치평균 learning loss
            val_loss_lst.append(val_loss) 

            # set lr_scheduler
            if args.lr_scheduler == 'reduce':
                scheduler.step(val_loss) #PleuOnReduce
            elif args.lr_scheduler == 'cosw':
                scheduler.step() # warmcosine  
            
            # Compute and print the average accuracy fo this epoch when tested over all trainset
            test_loss, metric = testPerformance(a_name, model, test_loader, device, args)
            metric_lst.append(metric)
            print("\n=======================================================================\n")
            print('Test For epoch', epoch+1,'the test mseloss over the whole test set is %.5f ' % (test_loss))
            print("\n=======================================================================\n")
            
            if test_loss < best_accuaracy:
                print('test_loss < best_accuaracy, Price test_loss(MSE): ', test_loss)
                PretrainUtilites.saveModel(model=model, save_directory=PRETRAIN_SAVEDIR, network_name=f'{args.task}_{a_name}')
                best_accuaracy = test_loss

            # 로스정보 저장
            learning_curve_df = pd.DataFrame({"train":train_loss_lst, "val":val_loss_lst})
            learning_curve_df.to_csv(f"{PRETRAIN_SAVEDIR}/logs/learning_curve_df_{a_name}.csv", index_label=False)
            
            metric_df = pd.DataFrame(metric_lst, columns=[['MAE', 'RMSE', "MAPE", "R2"]])
            metric_df.to_csv(f"{PRETRAIN_SAVEDIR}/logs/test_metric_{a_name}.csv", index_label=False)

            train_metric_lst.append([mae, rmse, mape, r2]) #{"MAE": mae, "RMSE": rmse, "MAPE":mape, "R2":r2}
            train_metric_df = pd.DataFrame(train_metric_lst, columns=[['MAE', 'RMSE', "MAPE", "R2"]])
            train_metric_df.to_csv(f"{PRETRAIN_SAVEDIR}/logs/train_metric_{a_name}.csv", index_label=False)





    #-----------------------After the Learning Phrase is done.. ----------------------------------------
    # Draw learning curve & dataframe
    learning_curve_df = pd.DataFrame({"train":train_loss_lst, "val":val_loss_lst})
    plt.plot(learning_curve_df)
    plt.ylabel("Loss")
    plt.xlabel("epoch (per 30)")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{PRETRAIN_SAVEDIR}/fig/learning_curve_{a_name}_{args.task}_{DATE_TIME}.png")
    plt.close()
    time.sleep(0.5)

    learning_curve_df.to_csv(f"{PRETRAIN_SAVEDIR}/logs/learning_curve_df_{a_name}_{args.task}_{DATE_TIME}.csv", index_label=False)
    print('Finished Training At epoch', epoch+1,': the %s test MSE loss over the whole test set is %.2f ' % (a_name, test_loss)) 
    return val_loss, metric




def main(args):
    # set wandb
    if args.sweep: 
        # note that we define values from `wandb.config`  
        # Set default configurations
        wandb.init(project='pretrain') # default_config=default_config
        args.lr_pi =  wandb.config.lr ####
        args.batch_size = wandb.config.batch_size
        args.n_epochs = wandb.config.epochs
        args.window = wandb.config.window
        args.layer_size = wandb.config.layer_size
        # sweep_params = {'MAE':[], 'RMSE':[], 'MAPE':[], 'Loss':[]} # 각 자산의 최종 메트릭
        sweep_result = []


    # initializing the random seeds for reproducibility
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    OBS_SPACE_DIMENSION = (args.window, 3) #(args.window, STOCK_SPACE_DIMENSION, 3) # IMFs 총 3개

    # # creating all the necessary directory tree structure for efficient logging
    # checkpoint_directory = create_directory_tree(mode=args.mode,
    #                                         buffer='basic',
    #                                         agent_type='pretrain',
    #                                         ffactor='pretrain_lstm',
    #                                         country = args.country,
    #                                         experimental = args.experimental,
    #                                         checkpoint_directory=PRETRAIN_SAVEDIR) ## 이거 어떻게 하면 좋을까나
    print("\n*----------------------------------------------------------------*\n")
    print("checkpoint_directory: ", PRETRAIN_SAVEDIR)
    print("\n*----------------------------------------------------------------*\n")
    
    # saving the (hyper)parameters used for future reference
    params_dict = vars(args) 
    params_dict['checkpoint'] = PRETRAIN_SAVEDIR #checkpoint_directory
    params_dict['START_DATE'] = START_DATE
    params_dict['TRAIN_END'] = TRAIN_END
    params_dict['VAL_END'] = VAL_END
    params_dict['TEST_END'] = TEST_END
    
    with open(os.path.join(PRETRAIN_SAVEDIR, args.mode+"_parameters_pretrain.json"), "w") as f:
        json.dump(params_dict, f, indent=4)
    
    
    # load data and models 
    whole_test_result = {}
    asset_datasets = load_data(args)

    
    # train each asset timeseries
    for a_name in asset_datasets.keys():
        dataset = asset_datasets[a_name]
        y_data = dataset["label"] # 가격정보
        trainset, valset, testset = tvt_split(dataset, args.window, log_scale=args.scale)

        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True) ## 
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False) 
        test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False) 
        dataloader = (train_loader, val_loader, test_loader)
        
        train_loader_graph = DataLoader(trainset, batch_size=args.batch_size, shuffle=False) ## 
        dataloader_graph = (train_loader_graph, val_loader, test_loader)
        
        if args.network == 'LSTM':
            model = SimpleLSTM(                                    
                                lr_pi=args.lr_pi, 
                                input_shape= OBS_SPACE_DIMENSION, ## env.observation_space.shape([1],[2])
                                layer_neurons= args.layer_size,
                                network_name = f'{a_name}_LSTM',
                                checkpoint_directory_networks = f"{PRETRAIN_SAVEDIR}/weights/{a_name}",
                                device=device
                                ).to(device)
        elif args.network == 'GRU':
            model = SimpleGRU(                                    
                                lr_pi=args.lr_pi, 
                                input_shape= OBS_SPACE_DIMENSION, ## env.observation_space.shape([1],[2])
                                layer_neurons= args.layer_size,
                                network_name = f'{a_name}_GRU',
                                checkpoint_directory_networks = f"{PRETRAIN_SAVEDIR}/weights/{a_name}",
                                device=device
                                ).to(device)
            

            print(f'{model.network_name} model params :', count_parameters(model))

        print("\n------------------------ START TRAINING ---------------------\n")
        # running the training or testing, saving plots
        initial_time = time.time()
        # train
        val_loss, asset_result = train(a_name, model, dataloader, device, args) ###
        # asset_result > dataframe
        whole_test_result[f'{a_name}'] = {"MAE":asset_result[0], "RMSE":asset_result[1],
                                          "MAPE":asset_result[2], "R2":asset_result[3], "val_loss":val_loss}
        final_time = time.time()
        prediction_plot(a_name, model, dataloader_graph, y_data, device, args)
        print("-------------------------------------------------------------")
        print('\nTotal {}ing duration: {:*^13.3f} \n'.format(args.mode, final_time-initial_time))
        print('\nAsset {} Learning Ends.. \n'.format(a_name))
        print("-------------------------------------------------------------")


        # whole test result -> df
        whole_df = pd.DataFrame(whole_test_result)
        
        try:
            whole_df = whole_df.transpose()

            # 여기에 sweep으로 컨트롤 할 수 있는 measurement 만들기
            # wandb.log
            if args.sweep:
                # 각 자산별로 메트릭 기록 쌓기
                sweep_result.append({
                                    "MAE": whole_df.MAE.mean(),
                                    'RMSE': whole_df.RMSE.mean(), 
                                    'MAPE': whole_df.MAPE.mean(),
                                    'val_loss': whole_df.val_loss.mean(),
                                    })

                sweep_data = wandb.Table(data = pd.DataFrame(sweep_result))

                wandb.log({
                            "MAE": whole_df.MAE.mean(),
                            'RMSE': whole_df.RMSE.mean(), 
                            'MAPE': whole_df.MAPE.mean(),
                            'val_loss': whole_df.val_loss.mean(),
                            'graph': wandb.plot.line(sweep_data, "n_asset", "metrics", title='Metrics of pretrain network')
                            })
            # csv file
            whole_df.to_csv(f"{PRETRAIN_SAVEDIR}/test_result_per_asset_{args.task}.csv", index_label=False)

        except:
            pass
        





if __name__ == '__main__':
    # argparse 채우기
    parser = ArgumentParser()

    # important options for train 
    parser.add_argument('--task',              type=str,   default='price',      help='direction or price(regression)')
    parser.add_argument('--network',           type=str,   default='GRU',        help='GRU, LSTM, ... ')
    parser.add_argument('--norm',              type=str,   default='logr',     help='minmax(기본), logr ')
    parser.add_argument('--country',           type=str,   default='NASDAQ',        help='USA, UK, JAPAN')
    parser.add_argument('--window',            type=int,   default=60,           help='Window for correlation matrix computation')
    parser.add_argument('--lr_scheduler',      type=str,   default='none',       help='reduce, coswarm, none')
    
    # hyperparameters for the Supervised training process
    parser.add_argument('--lr_pi',       type=float, default=0.0005,  help='Learning rate for the actor networks')
    parser.add_argument('--batch_size',  type=int,   default=64,      help='Batch size when sampling from the replay buffer')
    parser.add_argument('--grad_clip',   type=float, default=0.0,     help='Bound in case one decides to use gradient clipping in the training process')
    parser.add_argument('--layer_size',  type=int,   default=4,      help='Number of neurons in the various hidden layers')
    parser.add_argument('--scale',       type=int,   default=1,      help='Number of neurons in the various hidden layers')

    # Number of training or testing episodes and mode
    parser.add_argument('--n_epochs',     type=int,   default='2000',  help='Number of training or testing episodes')
    parser.add_argument('--mode',         type=str,   default='train', help="Mode used: 'train' or 'test'")
    parser.add_argument('--experimental', action='store_true',         help='Saves all outputs in an overwritten directory, used simple experiments and tuning')

    # parameters defining the trading environment
    parser.add_argument('--initial_date',      type=str,   default='2002-09-03',  help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--final_date',        type=str,   default='2018-05-27',  help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")    
    parser.add_argument('--test_start',        type=str,   default='2020-09-13', help="Initial date of the multidimensional time series of the assets price: str, larger or equal to '2019-07-03'")
    parser.add_argument('--test_end',          type=str,   default='2022-12-31', help="Final date of the multidimensional time series of the assets price: str, smaller or equal to '2020-12-30'")

    # random seed, logs information and hardware
    parser.add_argument('--checkpoint_directory', type=str,             default=BASE_DIR,     help='In test mode, specify the directory in which to find the weights of the trained networks')
    parser.add_argument('--plot',                 action='store_true',  default=True,         help='Whether to automatically generate plots or not')
    parser.add_argument('--seed',                 type=int,             default='42',         help='Random seed for reproducibility')
    parser.add_argument('--gpu_devices',          type=int,  nargs='+', default=[1],         help='Specify the GPUs if any')
    parser.add_argument('--sweep',                type=int,             default=1,            help='if 1 True(Sweep hyperparamenter search) Else False')

    args = parser.parse_args()
    args.country = args.country.upper() # 대문자로 변환
    print(args.country)


    # set trainable gpu_devices
    # GPU 할당 변경하기
    gpu_devices = ",".join([str(id) for id in args.gpu_devices])
    device = torch.device(f'cuda:{gpu_devices}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print('Current cuda device ', torch.cuda.current_device()) # check
    

    # 임의로 하이퍼파라미터 세팅
    START_DATE = args.initial_date
    TEST_END = args.test_end # 20221231
    time_diff = datetime.strptime(TEST_END, "%Y-%m-%d")-datetime.strptime(args.initial_date, "%Y-%m-%d")
    TRAIN_END = datetime.strftime(datetime.strptime(args.initial_date, "%Y-%m-%d")+timedelta(days=int(time_diff.days*0.8)+1), "%Y-%m-%d")
    VAL_END = datetime.strftime(datetime.strptime(args.initial_date, "%Y-%m-%d")+timedelta(days=int(time_diff.days*0.9)+1), "%Y-%m-%d")
    args.final_date = TRAIN_END # 2017-08-08


    # make folders to save the logs
    DATE_TIME = str(datetime.today())
    PRETRAIN_SAVEDIR = args.checkpoint_directory+"/saved_outputs/"+f'{args.country}/pretrain/{DATE_TIME}'
    os.makedirs(PRETRAIN_SAVEDIR+'/fig', exist_ok=True)
    os.makedirs(PRETRAIN_SAVEDIR+'/logs', exist_ok=True)

    if args.country == 'DJIA':
        STOCK_SPACE_DIMENSION = 26
    else:
        STOCK_SPACE_DIMENSION = 30

    # unchangerble parameters
    ACTION_SPACE_DIMENSION = STOCK_SPACE_DIMENSION
    MAX_ACTION = 1 # self.env.action_space.high
    SCALE = 100
    args.sweep = True if args.sweep==1 else False
    print('\n', args, '\n')
    
    if args.sweep : 
        # # Start sweep job.
        wandb.login()
        sweep_id = wandb.sweep(sweep=sweep_configuration, project='pretrain')
        
        # args...
        function = lambda : main(args)  
        wandb.agent(sweep_id, function=function, count=1)
    
    else : 
        main(args=args)


 
