
import time
from datetime import date, datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from torch.autograd import Variable
from torchmetrics.functional import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


from actor_pretrain.policy import *
from util.utilities import *
from util.configures import *
from actor_pretrain.utils_pretrain import PretrainDataset, PretrainUtilites






def prediction_plot(a_name, model, dataloader, y_data, device, pre_savedir, args):
    '''
    train 부분은 안해도 되나?
    - pretrain_savedir
    - TEST_END -> args에 포함

    - 
    '''
    # 여기서 제일 학습 잘된 모델 웨이트를 갖고오자..
    w = args.window
    # y_data = y_data[y_data.index>=VAL_END]
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


