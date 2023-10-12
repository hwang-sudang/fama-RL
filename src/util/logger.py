# 여기에 차라리 경로만드는게 낫나?
import numpy as np
import os
import pandas as pd
import time

from util.plot import Plot
from util.configures import *




class Logger():
    """
    A helper class to better handle the saving of outputs.
    """
    def __init__(self,
                mode: str,
                checkpoint_directory: str,
                ) -> None:
        """
        Args:
            mode (str): 'train' of 'test'
            checkpoint_directory: path to the main checkpoint directory, in which the logs
                                  and plots subdirectories are located
                                  
        Returns:
            no value        
        """
        self.mode = mode
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_directory_logs = os.path.join(self.checkpoint_directory, "logs")
        self.checkpoint_directory_plots = os.path.join(self.checkpoint_directory, "plots")

        print(self.checkpoint_directory)
        print(self.checkpoint_directory_logs)
        print(self.checkpoint_directory_plots)

        # 에피소드 단위로 로그를 저장하는 영역
        self.logs: dict = {}
        self.logs['reward_history'] = []
        self.logs['asset_ratio_history'] = []
        self.logs['portfolio_value'] = []
        self.logs['attention_map'] = []
        # self.logs['sharpe_ratio'] = []
        # self.logs['jensen_a'] = []
        # self.logs['MDD'] = []

        if self.mode == 'test':
            # Test 기간동안의 자산 배분비율, 포트폴리오 가치의 변화율
            self.logs['asset_ratio_timeseries']= [] #시간 흐름에 다른 포트폴리오 비율 변화
            self.logs['portfolio_value_timeseries'] = [] # 시간 흐름에 따른 포트폴리오 가치 변화
            self.logs['attention_timeseries'] = []
            self.logs['traditional_pf_dict'] = {}    

        elif self.mode == 'train':
            self.logs['critic_loss'] = []
            self.logs['actor_loss'] = []

        self.time_stamp = [0, 0]
        self.initial_value_portfolio = None



    def set_time_stamp(self,
                        i: int,
                        ) -> None:
        """
        작업 진행 상황을 모니터링하기 위해 타임스탬프를 추적하는 메소드
        """
        self.time_stamp[i-1] = time.time()



    def print_status(self, 
                    episode: int,
                    ) -> None:
        """
        현재 학습 상황을 출력하는 메소드
        """
        print('   episode: {:<13d} |   reward: {:<13.6f} |    PV: {:<13.2f} |    duration: {:<13.2f}'.format(episode,
                                                                                       (self.logs["reward_history"][-1]) * 100,
                                                                                       self.logs['portfolio_value'][-1],
                                                                                        self.time_stamp[1]-self.time_stamp[0])) ##
        print('------------------------------------------------------\n\n')




    def save_logs(self) -> None:
        """
        Saves all the necessary logs to 'checkpoint_directory_logs' directory.
        """
        reward_history_array = np.array(self.logs['reward_history'])
        os.makedirs(os.path.join(self.checkpoint_directory_logs, self.mode), exist_ok=True)
        np.save(os.path.join(self.checkpoint_directory_logs, self.mode+f"_reward_history.npy"), reward_history_array)


        if self.mode == 'test':
            portfolio_value_history_of_histories_array = np.array(self.logs['portfolio_value'])
            np.save(os.path.join(self.checkpoint_directory_logs, self.mode+'_portfolio_content_history.npy'), portfolio_value_history_of_histories_array)
            df = pd.DataFrame(self.logs['portfolio_value'], columns=['test_reward'])
            df.to_csv(self.checkpoint_directory_logs+"/"+self.mode+f"_portfolio_value.csv")
            



    def generate_plots(self, asset_label:list,) -> None:
        '''
        helper 함수를 호출하여 훈련 모드에서 보상 내역, 테스트 모드에서는 보상의 분포를 plotting 하는 메소드
        여기가 지금 계산이 안되는 상황
        '''
        reward_history_array = np.array(self.logs['reward_history'])
        n_episodes = reward_history_array.shape[0]
        episodes = [i+1 for i in range(n_episodes)]
        
        # plot reward and portfolio_value
        Plot.plot_reward(x=episodes,
                    rewards=(reward_history_array/self.initial_value_portfolio)*100,
                    figure_file=os.path.join(self.checkpoint_directory_plots, self.mode+"_reward"),
                    mode=self.mode,
                    bins=np.sqrt(n_episodes).astype(int))
        
        Plot.plot_portfolio_value(x=episodes,
                                values = np.array(self.logs['portfolio_value']),
                                figure_file = os.path.join(self.checkpoint_directory_plots, self.mode+f"_portfolioValue"))

        # plot assets' allocation
        Plot.plot_portfolio_ratio(x=episodes,
                    asset_ratio=np.array(self.logs['asset_ratio_history']),
                    figure_file=os.path.join(self.checkpoint_directory_plots, self.mode+f"_A_ratio"),
                    asset_label = asset_label,
                    mode=self.mode,
                    )

        # for only train mode
        if self.mode == 'train':
            Plot.learning_curve(os.path.join(self.checkpoint_directory_plots,"learning_curve"),
                            episodes[-1],
                            self.logs["actor_loss"],
                            model_info="actor_loss")

            Plot.learning_curve(os.path.join(self.checkpoint_directory_plots,"learning_curve"),
                            episodes[-1],
                            self.logs["critic_loss"],
                            model_info='critic_loss')
        
        else:
            # for test mode only 
            portfolio_value_history_of_histories_array = np.array(self.logs['portfolio_value'])
            n_days = portfolio_value_history_of_histories_array.shape[-1]
            days = [i+1 for i in range(n_days)]
            
            # 여기에서 보상의 분포를 보여줌
            idx = np.random.choice(n_episodes, min(n_episodes, 5))
            Plot.plot_portfolio_value(x=days,
                                    values = portfolio_value_history_of_histories_array[idx],
                                    figure_file = os.path.join(self.checkpoint_directory_plots, 
                                                            self.mode+f"_portfolioValue"))


    def _store_initial_value_portfolio(self,
                                        initial_value_portfolio: float,
                                        ) -> None:
        """
        initial_portfolio_value 속성에 대한 Setter 메서드
        """
        self.initial_value_portfolio = initial_value_portfolio




    def portfolio_content_to_dataframe(self, tickers:list):
        portfolio_content_history_array = np.array(self.logs['portfolio_value_timeseries'])
        df = pd.DataFrame(data=portfolio_content_history_array, columns=tickers) ##shape
        return df






# 확인
if __name__ == 'main':
    logger = Logger()

