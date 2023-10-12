import os
import time
import copy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Union
sns.set_theme()

# set the pause to prevent plot error
TIME_SLEEP = 3 



class Plot:
    def __init__(self) -> None:
        '''
        어떤 결과를 볼 것인지 고민하기
        1. 누적 포트폴리오 가치(수익률) 변화 확인
        2. 포트폴리오의 리스크를 고려한 계수에 대한 분석 필요
        '''
        pass
    
    
    @staticmethod
    def plot_reward(x: List[int],
                    rewards: np.ndarray,
                    figure_file: str,
                    mode: str,
                    bins : int = 20,
                    ) -> None:
        '''
        에피소드마다의 누적 리워드 결과를 plotting
        각 에이전트마다 주어진 리워드를 가지고 확인
            - 따라서 모델의 성능확인용
            - 50회마다 평균 average plotting

            - max, min에 대한 영역 컬러링 기능 추가
        '''

        running_average = np.zeros(len(rewards))
        running_max = np.zeros(len(rewards))
        running_min = np.zeros(len(rewards))

        for i in range(len(running_average)):
            running_average[i] = np.mean(rewards[max(0, i-50): i+1])
            running_max[i] = np.max(rewards[max(0, i-50): i+1])
            running_min[i] = np.min(rewards[max(0, i-50): i+1])


        if mode == 'train':
            plt.plot(x, rewards, linestyle='-', color='blue', label='reward')
            plt.plot(x, running_average, linestyle = '--', color='green', label='running average 50' )
            plt.fill_between(x, running_max, running_min,
                                facecolor='lightblue', alpha=0.3)
            plt.legend()
            plt.title('Reward as a function of the epoch/episode')

        elif mode == 'test':
            plt.hist(rewards, bins=bins)
            plt.title('Reward as a function of the Test Episodes')
        
        plt.savefig(figure_file+".png", format='png', dpi=200)
        plt.cla()
        plt.close()
        time.sleep(TIME_SLEEP)



    @staticmethod
    def plot_portfolio_value(x:List[int],
                            values: np.ndarray,
                            figure_file: str,
                            mode:str = "train",
                            ) -> None :
        '''
        # Test시 plot_portfolio_value plotting
        - x (list of int) : 
        - 
        '''
        os.makedirs(figure_file, exist_ok=True)
        # plt.plot(x, values.T, linestyle= '-', linewidth = 0.5)
        plt.plot(values.T, linestyle= '-', linewidth = 0.5)
        plt.xlim((0, len(values.T)))
        plt.title('Portfolio value')
        if mode == 'train':
            plt.savefig(figure_file+f"/PortfolioValue__ep{(len(x)-1)}.png")
        else:
            plt.savefig(figure_file+f"/PortfolioValue_test.png")
        plt.cla()
        plt.close()
        time.sleep(TIME_SLEEP)


    @staticmethod
    def plot_portfolio_ratio(x : List[int],
                            # ep :int = 1,
                            asset_ratio: np.ndarray,
                            figure_file: str,
                            asset_label, #list
                            mode: str = 'train', 
                            ) -> None :
        '''
        각 에피소드마다 포트폴리오 자산의 구성 비율을 plotting
        - x(int) : episodes list, [1,2,....,episode]
        - asset_ratio: 자산 배분 정보가 들어있는 np.array
            - shape : (30, epsiodes) 2D
            - pie plot은 1d data만 플랏 가능

        - figure_file(str) : reward를 저장하는 장소
        - mode : train, test

        # pieplot documents : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html
        '''
        # asset_label.append("Cash(Balance)")
        save_dir = os.path.dirname(figure_file)+"/portfolio_ratio"
        os.makedirs(save_dir, exist_ok=True)

        # percentage plot
        if mode == 'train':
            for episode in range(len(asset_ratio)):
                if episode % 50 == 0:
                    plt.figure(figsize=(30,30)) #20,20?
                    plt.pie(asset_ratio[episode, :], autopct='%.2f%%', labels=asset_label, textprops={'fontsize': 20, 'weight':3})
                    plt.title(f'{mode} Ep.{episode} Portfolio value', fontdict={'fontsize': 20, "weight":5})
                    plt.savefig(save_dir+f"/asset_ratio_{mode}_ep{episode}.png")
                    plt.cla()
                    plt.close()    
                    time.sleep(TIME_SLEEP)
        
        elif mode =='test' : 
            # 최종결과만 플랏
            os.makedirs(figure_file, exist_ok=True)
            plt.figure(figsize=(30,30))
            plt.pie(asset_ratio[-1, :], autopct='%.2f%%', labels=asset_label, textprops={'fontsize': 20, 'weight':3})
            plt.title(f'{mode} Final Test Portfolio value', fontdict={'fontsize': 20, "weight":5})
            plt.savefig(figure_file+f"/asset_ratio_{mode}.png")
            plt.cla()
            plt.close()
            time.sleep(TIME_SLEEP)






    @staticmethod
    def learning_curve(figure_file:str,
                        episode:int,
                        train_loss_lst,
                        model_info:str = None):
        '''
        Draw learning curve with train, validation loss
            - model_info : 필요하면 어떤 에이전트인지, Critic/Actor 모델 쓰는지 기재
        '''
        if not os.path.exists(figure_file):
            os.makedirs(figure_file, exist_ok=True)

        # plot the learning curves
        learning_curve_df = pd.DataFrame({"train":train_loss_lst})
        plt.plot(learning_curve_df)
        # plt.xticks(ticks=np.arange(0, len(learning_curve_df)), labels=list(learning_curve_df.index*50))
        # plt.ylim(0, 0.7)
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend(["Train"])

        # save the learning curve logs
        plt.savefig(f"{figure_file}/learning_curve_%s_ep%d_%s.png" % (str(datetime.today()), episode, model_info))
        learning_curve_df.to_csv(f"{figure_file}/learning_curve_df_%s_ep%d_%s.csv" % \
                                                                 (str(datetime.today()), episode, model_info), index_label=False)
        plt.cla()
        plt.close()
        time.sleep(TIME_SLEEP)

    

    @staticmethod
    def action_movement(figure_file:str,
                        stock_ts:np.ndarray,
                        asset_ts:np.ndarray,
                        asset_label:list,
                        date:pd.Series,
                        seed:int
                        ):
        '''
        실제 자산의 weight와 가격 정보 - for only test    
        input 데이터 타입이랑, date만 어떻게 뽑을지 고민해보자
        
        '''
        # make directories
        if not os.path.exists(figure_file):
            os.makedirs(figure_file, exist_ok=True)

        # set the init values
        window = len(date)-len(stock_ts)
        date = [d.strftime('%Y-%m-%d') for d in date[window:]]
        df_container = {'date':date[1:]}
        X_axis = np.arange(len(date))

        # Make StockPrice plot
        for a in range(len(asset_label)-1):
            asset_name = asset_label[a]
            
            # save dataset for df_container
            df_container[f'{asset_name}_p']=stock_ts[1:,a] # date가 어째서인지 asset_ts가 하루 짧음..
            df_container[f'{asset_name}_a']=asset_ts[0,:,a]
            
            # Make Asset_ratio Plot
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.bar(np.arange(len(date[1:])), asset_ts[0,:,a], color='grey')
            ax.set_ylabel('Allocation Ratio')
            ax.set_ylim(bottom=min(asset_ts[0,:,a])) ## 
            ax.tick_params(axis='y', labelcolor='grey')


            # Make StockPricePlot
            stock_plt = ax.twinx()
            stock_plt.plot(np.arange(len(date)), stock_ts[:,a],  linestyle= '-', linewidth = 1, color='blue')
            stock_plt.set_ylabel('Stock Price')
            stock_plt.tick_params(axis='y', labelcolor='blue')
            

            plt.title(f'{asset_name} Stock Price & Allocation Timeseries')
            plt.xticks(X_axis, date, fontsize=15)
            plt.xlabel('Date')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

            # save the figures
            plt.savefig(f"{figure_file}/StockActionComparison_%s.png" % (asset_name))
            plt.cla()
            plt.close()
            
        # save the log csv file
        df = pd.DataFrame(df_container)
        df['cash_a'] = asset_ts[0,:,-1]
        df.to_csv(f"{figure_file}/StockActionComparison_{seed}.csv", index=False)

        df2 = df[['date']+[i for i in df.columns if '_a' in i]].copy()
        df2.to_csv(f"{figure_file}/StockActionComparison_onlyAction_{seed}.csv", index=False)


        time.sleep(TIME_SLEEP)

    


    @staticmethod
    def attention_heatmap(figure_file:str,
                        action_dim:int,
                        attention_map:list, 
                        feature_name:list,
                        date:pd.Series,
                        seed=int):
        ''' 
        attention_map : (batch=1, timelength, asset*2+factor) or (batch=1, timelength, asset, 6+factor) 
        https://nlpstudynote.tistory.com/19

        - x축 통합해서 하나로 나오도록 수정하기
        - 그래프 크기 조정
        '''
        
        # make directories
        if not os.path.exists(figure_file):
            os.makedirs(figure_file, exist_ok=True)

        # make attention_map array to visualize 
        attention_map = np.array(attention_map) 
        # 차원이 3차원이상인 경우, attention이 적용되는 축(dim=-1)으로 합산
        if attention_map.ndim <= 3:
            attention_map = attention_map.sum(axis=1) #(timeseries, features)
        # change the axis
        attention_map = attention_map.T


        # variables
        #.T # 문제가 어텐션이..ㅋㅋ 2차원이 된게 문제구만
        window = len(date)-attention_map.shape[-1]
        date = [d for d in date[window:].strftime('%Y-%m-%d')]

        # set the figure size and location
        plt.rcParams["figure.figsize"] = [20, 60]
        plt.rcParams["figure.autolayout"] = True

        # draw the attention heatmap
        # fig, ax = plt.subplots()
        fig = plt.figure()

        # ax1 : Whole attention
        ax1 = plt.subplot(4, 1, 1)    # nrows=4, ncols=1, index=1
        heatmap= ax1.pcolor(attention_map, cmap=plt.cm.Blues)
        fig.colorbar(heatmap, ax=ax1)
        ## set ticks
        ax1.set_xticks(np.arange(len(date)), minor=False)
        ax1.set_xticklabels(date, minor=False)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_yticks(ticks = np.arange(len(feature_name)), 
                       labels = feature_name,
                       minor=False, rotation=45)       
        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_title('All Feature Attention heatmap', fontsize=14)

        
        ### 흠 어텐션 스코어 weight에 따라 시각화가 달라질 것 같은데.
        # ax2 : Attention of Pt
        feat_p = feature_name[:action_dim]
        att2 = attention_map[:action_dim]
        ax2 = plt.subplot(4, 1, 2, sharex=ax1)    # nrows=4, ncols=1, index=2
        heatmap2 = ax2.pcolor(att2, cmap=plt.cm.Blues)
        fig.colorbar(heatmap2, ax=ax2)
        ## set ticks
        ax2.set_yticks(np.arange(len(feat_p)), labels=feat_p, minor=False, rotation=45)     
        ax2.set_title('Price Attention Heatmap', fontsize=14)        


        # ax3 : Attention of Rt
        feat_r = feature_name[action_dim:action_dim*2]
        att3 = attention_map[action_dim:action_dim*2]
        ax3 = plt.subplot(4, 1, 3, sharex=ax1)    # nrows=4, ncols=1, index=3
        heatmap3 = ax3.pcolor(att3, cmap=plt.cm.Blues)
        fig.colorbar(heatmap3, ax=ax3)
        ## set ticks
        ax3.set_yticks(np.arange(len(feat_r)), labels=feat_r, minor=False, rotation=45)     
        ax3.set_title('Return Attention Heatmap', fontsize=14)   


        # ax4 : Attention of Factor Features
        feat_f = feature_name[action_dim*2:]
        att4 = attention_map[action_dim*2:]
        ax4 = plt.subplot(4, 1, 4, sharex=ax1)    # nrows=2, ncols=1, index=2
        heatmap4 = ax4.pcolor(att4, cmap=plt.cm.Blues)
        fig.colorbar(heatmap4, ax=ax4)
        ## set ticks
        ax4.set_yticks(np.arange(len(feat_f)), labels=feat_f, minor=False, rotation=45) 
        ax4.set_title('Factor Attention Heatmap', fontsize=14)
        
        # Plot title 
        plt.title('Attention Heatmaps', fontsize=16)

        # save the figures
        fig.tight_layout()
        plt.title('Attention Heatmaps', fontsize=16)
        plt.savefig(f"{figure_file}/Attention_heatmap_{seed}.png")
        plt.cla()
        plt.close()




    
    @staticmethod
    def backtest(figure_file:str,
                 portfolio_dict:dict,
                 date,
                 seed:int
                ):
        '''
        Plot.backtest()
            - portfolio_ts(np.ndarray) : 각 자산의 포트폴리오 밸류 timeseries
            - tradional_pf(dict) : 비교 포트폴리오 밸류 timeseries의 모음
                - {'equally':[], 'security':[], ...}
                
        '''
        # make directories
        if not os.path.exists(figure_file):
            os.makedirs(figure_file, exist_ok=True)

        # variables
        window = len(date)-len(portfolio_dict["Ours"])
        date = [d.strftime('%Y-%m-%d') for d in date[window:]]
        df_container = {'date':date}

        # draw portfolio comparison
        fig, ax = plt.subplots(figsize=(20,10)) #20,20?        
        for name, value in portfolio_dict.items():
            if name == "Ours":
                ax.plot(date, np.array(value), linestyle= '-', linewidth = 3, label='Ours')
                df_container["Ours"] = np.array(value)
            else:
                ax.plot(date, np.array(value), linestyle= '-', linewidth = 1, label=name)
                df_container[name] = np.array(value)
        
        plt.title(f'Comparison of the Portfolios', fontdict={'fontsize': 30, 'fontweight':'bold'})
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.legend()

        # save the figures
        plt.savefig(f"{figure_file}/Backtest_Comparison_%s_%d.png" % (str(datetime.today()), seed))
        plt.cla()
        plt.close()
        
        # save the log csv file
        df = pd.DataFrame(df_container)
        df.to_csv(f"{figure_file}/BacktestComparison_{seed}.csv", index=False)        
        time.sleep(TIME_SLEEP)








class PortfolioEvaluator:
    def __init__(self, checkpoint_directory:str, portfolio_dict:dict,  date, seed:int):
        # make directories
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory, exist_ok=True)

        self.checkpoint_directory = checkpoint_directory
        self.window = len(date)-len(portfolio_dict["Ours"])
        self.date = [d.strftime('%Y-%m-%d') for d in date[self.window:]]
        self.portfolio_dict = copy.copy(portfolio_dict)
        self.seed = seed 
        # self.df_container = {'date':self.date}

        # make the portfolio df
        self.portfolio_dict['date'] = self.date 
        self.portfolio_df = pd.DataFrame(self.portfolio_dict)
        self.portfolio_df['date'] = pd.to_datetime(self.portfolio_df['date'], format='%Y-%m-%d')
        self.portfolio_df.set_index('date', inplace=True)

        self.yearlist =  list(self.portfolio_df.index.year.unique())
        self.portfolio_name = self.portfolio_df.columns.to_list()
        # self.risk_free = risk_free[self.portfolio_df.index[0]:]
        self.daily_return_df = (self.portfolio_df/self.portfolio_df.shift(1)).bfill()-1
        self.daily_risk_free = self.daily_return_df['security']

        

    def __call__(self):
        """        
        # 포트폴리오의 최종 실적 계산
        
        * Args
        - portfolio_dict(dict) : 각 포트폴리오의 시간 흐름에 따른 value 계산
            - key() : name of portfolio
            - value() : value of portfolio

        * Objects
        - Annual Sharpe Ratio : 연간 샤프비율 
        - Annual Maximum DropDown :
        - Expected Annual Sharpe Ratio
        - Expected Annual MDD
        
        - Jensen's alpha(고민 중)
 
        """
        sharpe_all, std_all, annual_sharpe, annual_std, rate_of_return, ARR, AR = self.sharpe_ratio()
        MDD_all, annual_mdd = self.MaximumDrawDown()
        dic = {
               'rate_of_return_all':rate_of_return, 'Annual_RR':ARR, 'Annual_Return':AR,
               'std_all':std_all, 'std_annual':annual_std,
               'sharpe_all': sharpe_all, "sharpe_annual":annual_sharpe, 
               'MDD_all':MDD_all, 'MDD_annual':annual_mdd 
                }

        df = pd.DataFrame(dic).T
        df.columns = self.portfolio_name
        df.to_csv(self.checkpoint_directory+f"/test_portfolio_evaluation_{self.seed}.csv")

        # print(dic)
        return df
    

    def sharpe_ratio(self):
        """        
        # Annual Sharpe Ratio : 연간 샤프비율 
        
        * Args
        - portfolio_dict(dict) : 각 포트폴리오의 시간 흐름에 따른 value 계산
            - key() : name of portfolio
            - value() : value of portfolio

        * Objects
        - Annual Sharpe Ratio : 연간 샤프비율 
        - Annual Maximum DropDown :
        - Jensen's alpha(고민 중)
        - Expected Annual Sharpe Ratio
        - Expected Annual MDD
        """

        ########## sharpe ratio ################
        # total sharpe ratio
        sharpe_dict = {}
        std_dict = {}
        ARR_dict = {}
        AR_dict = {}

        # for all period... 
        n_period = len(self.daily_return_df)
        std_all = self.daily_return_df.std(axis=0)*np.sqrt(n_period)
        rate_of_return = self.daily_return_df.sub(self.daily_risk_free, axis=0).mean(axis=0)*n_period 
        return_ = self.daily_return_df.mean(axis=0)*n_period 
        sharpe_all = rate_of_return / std_all #annualize



        # annual sharpe ratio - 연간수익률인건지 
        for year in self.yearlist:
            # 시간별로 데이터를 자르자
            year_returns = self.daily_return_df[self.daily_return_df.index.year==year]
            year_riskfree = self.daily_risk_free[self.daily_risk_free.index.year==year]

            # Calculate the average annual return (daily return to annual return)
            avg_return = year_returns.sub(year_riskfree, axis=0).mean(axis=0)*252 #RoR
            avg_real_return = year_returns.mean(axis=0)*252
            # Calculate the standard deviation of the annual return
            std_dev = year_returns.std(axis=0)*np.sqrt(252)

            # Calculate the annual Sharpe ratio
            sharpe_ratio = avg_return / std_dev # annually
            sharpe_dict[year] = sharpe_ratio.to_list() # dataframe > list maybe...
            std_dict[year] = std_dev.to_list()
            ARR_dict[year] = avg_return.to_list()
            AR_dict[year] = avg_real_return.to_list()
        
        # compute the average of annual metrics
        annual_sharpe = pd.DataFrame(sharpe_dict).mean(axis = 1)
        annual_std = pd.DataFrame(std_dict).mean(axis = 1)
        ARR = pd.DataFrame(ARR_dict).mean(axis = 1) # annual rate of return
        AR = pd.DataFrame(AR_dict).mean(axis = 1) 
        return sharpe_all.to_list(), std_all.to_list(), annual_sharpe.to_list(), \
                annual_std.to_list(), rate_of_return.to_list(), ARR.to_list(), AR.to_list()


        

    def MaximumDrawDown(self):
        """        
        # MaximumDrawDown : https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp
        
        * Args
        - portfolio_dict(dict) : 각 포트폴리오의 시간 흐름에 따른 value 계산
            - key() : name of portfolio
            - value() : value of portfolio

        * Objects
        - Annual Maximum DropDown :
        - Expected Annual MDD
        """
        mdd_dict = {}

        # for all period... 
        cum_max = self.portfolio_df.cummax(axis=0)
        cum_min = self.portfolio_df.cummin(axis=0)    
        MDD_all = (abs((cum_min - cum_max) / cum_max)).mean(axis=0).to_list() # worst possible maximum drawdown would be 100%

        # annual MDD
        for year in self.yearlist:
            # 시간별로 데이터를 자르자
            portfolio_df = self.portfolio_df[self.portfolio_df.index.year==year]

            # Calculate the average annual return
            cum_max = portfolio_df.cummax(axis=0)
            cum_min = portfolio_df.cummin(axis=0)
            year_mdd = (abs((cum_min - cum_max) / cum_max)).mean(axis=0) # cummax-cummin 도 상관없음
            mdd_dict[year] = year_mdd.to_list()
        
        # compute the average of annual metrics
        annual_mdd = pd.DataFrame(mdd_dict).mean(axis = 1).to_list()
        return MDD_all, annual_mdd








