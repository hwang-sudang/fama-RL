import gym
import numpy as np
import json

from model.agent.base import Agent
from util.logger import Logger
from util.plot import *
from util.utilities import NpEncoder


class Run():
    """
    학습 또는 테스트를 실행할 기본 클래스.

    - _reset : 환경과 보상 이력을 초기화함
    - run : 특정 에피소드 만큼 훈련 또는 테스트 실행
    - _run_one_episode : 에피소드 하나 실행

    """

    def __init__(self,
                env: gym.Env,
                agent: Agent,
                n_episodes: int,
                agent_type: str,
                checkpoint_directory: str,
                stocks_subset,
                sac_temperature: float = 1.0,
                mode:str = 'test',
                window:int = 40,
                ) -> None:

        """Constructor method of the class Run.
        
        Args:
            env (gym.Env): trading environment in which the agent evolves
            agent (Agent): Soft Actor Critic like agent
            n_episodes (int): total number of episodes for training, or testing (since the policy is stochastic)
            agent_type (str): name of the type of agent for saving files
            sac_temperature (float): rescaling factor of rewards, in case one uses manual type agent
            mode (str): train or test
            
        Returns:
            no value
        """
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.agent_type = agent_type
        self.sac_temperature = sac_temperature
        self.mode = mode
        self.checkpoint_directory = checkpoint_directory
        self.stocks_subset = stocks_subset

        if self.mode == 'test':
            self.agent.load_networks()

        # step, episode, best_reward 초기값 정의
        self.step = 0
        self.episode = None
        self.best_reward = None
        self.cal_window = window
        # logger ON
        self.logger = Logger(mode=self.mode, checkpoint_directory=self.checkpoint_directory)
        self._reset()


    def _reset(self) -> None:
        """
        initialize the ennvironment and the reward history.
        환경과 보상 이력을 초기화함
        """
        self.step = 0
        self.episode = 0
        self.best_reward = float('-Inf')


    def _init_variables(self):
        # init the variables
        self.logger.set_time_stamp(1)   
        cum_reward: float = 0
        done: bool = False
        observation = self.env.reset()
        step = 0
        self.logger._store_initial_value_portfolio(self.env._get_portfolio_value())
        return step, cum_reward, done, observation


    def _evaluate_pf(self, cal_window=40):
        """        
        # 여기서 포트폴리오 실적 계산
        # 포트폴리오의 마지막 window만큼을 기준으로 포트폴리오 최종 실적을 계산한다.
        - # for Jensen's alpha & Sharpe Ratio
        """
        cal_window = min(cal_window, len(self.logger.logs['portfolio_value']))
        pf_return = np.array(self.logger.logs['portfolio_value'][-cal_window:])/np.array(self.logger.logs['portfolio_value'][-cal_window-1:-1])-1
        if len(pf_return) == 0: 
            pf_return = self.logger.logs['portfolio_value'][-cal_window:]/self.logger.initial_value_portfolio
       
        average_pf_return = np.mean(pf_return)
        std_pf_return = np.std(pf_return) 
        rf_return = self.env.daily_bank_rate
        
        # for MDD
        max_pf_value = max(self.logger.logs['portfolio_value'][-cal_window:])
        min_pf_value = min(self.logger.logs['portfolio_value'][-cal_window:])

        # 로그에 각 포트폴리오 measurements 저장하기
        self.logger.logs['sharpe_ratio'].append(np.nan_to_num((average_pf_return - rf_return)/std_pf_return))
        self.logger.logs['jensen_a'].append(average_pf_return-rf_return)
        self.logger.logs['MDD'].append((max_pf_value-min_pf_value)/max_pf_value)


    def _print_status(self, reward, action, info):
        ##----- log 쌓기
        ##----- for debugging
        print("-----------------------------------------------------------------------------------------")
        print(f"\nStep: {self.step}     Env step: {self.env.current_step}     done: {self.done}")
        # print('action_ratio: ', action.mean(), action.sum(), action.std())
        print("env reward : ", reward)
        print("Cum_reward: ", self.cum_reward)
        print("new_value_portfolio: ", info['portfolio'])
        # print("-----------------------------------------------------------------------------------------")


    def _run_one_episode(self) -> None:
        '''
        *** self.logs에 어떻게 비교 agent를 같이 시뮬레이션 할 수 있을까 고민하기 ***
        '''
        self.step, self.cum_reward, done, observation = self._init_variables()

        if self.mode=='train':
            # 에피소드가 끝나는 조건에 다다를때까지
            while not done:
                action = self.agent.choose_action(observation) ###
                observation_, reward, done, info = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, done)
                self.agent.learn(step = self.step)
                
                # Go to the next step
                observation = observation_
                self.step += 1
                self.cum_reward += reward
                self.done = done
                
                # The break point    
                if self.env.current_step % 500 == 0 or self.env.current_step==self.env.time_length:
                    self._print_status(reward, action, info)
                    if self.env.current_step>=self.env.time_length:
                        break

            # write logs
            self.logger.logs['critic_loss'].append(self.agent.critic_loss.item())
            self.logger.logs['actor_loss'].append(self.agent.actor_loss.item())
            
                        
        else:
            # test mode
            # initializing a list to keep track of the porfolio value during the episode
            # 에피소드 동안 포트폴리오 가치를 추적(track)하기 위해 list 초기화
            portfolio_content_history = [self.env.number_of_shares.tolist()] ### 여기 확인
            action_tslist = []
            pfvalue_tslist = []
            
            # with torch.nograd()
            while not done:
                action = self.agent.choose_action(observation) ###
                observation_, reward, done, info = self.env.step(action)
                self.agent.remember(observation, action, reward, observation_, done)
                
                # Go to the next step
                observation = observation_
                self.step += 1
                self.cum_reward += reward
                self.done = done

                # write timeseries logs
                action_tslist.append(action.tolist())
                pfvalue_tslist.append(self.env._get_portfolio_value())

                if self.env.current_step % 500 == 0 or self.env.current_step>=self.env.time_length:
                    self._print_status(reward, action, info)
                    if self.env.current_step>=self.env.time_length:
                        break

            # write logs
            assert len(action_tslist) == len(pfvalue_tslist), "len(action_tslist) != len(pfvalue_tslist)"
            portfolio_content_history.append(self.env.number_of_shares.tolist())
            self.logger.logs['asset_ratio_timeseries'].append(action_tslist)    
            self.logger.logs['portfolio_value_timeseries'].append(pfvalue_tslist)

            
        ##----- log 쌓기 (공통)
        self.logger.logs['reward_history'].append(self.cum_reward)
        self.logger.logs['asset_ratio_history'].append(action.tolist())
        self.logger.logs['portfolio_value'].append(self.env._get_portfolio_value())
        average_reward = np.mean(self.logger.logs['reward_history'][-self.cal_window:])

        # best reward 기록
        if average_reward  > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train': self.agent.save_networks()
        
        ###---- 포트폴리오 plotting 용 로그 쌓는 함수
        try: 
            self._evaluate_pf(cal_window=self.cal_window)
        except:
            # self._evaluate_pf(cal_window=self.cal_window) -> debugging 오류메세지 확인
            with open(f'{self.checkpoint_directory}/plot_error_{self.step}.json','w') as f:
                json.dump(self.logger.logs, f, cls=NpEncoder, indent=4)
            pass
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)




    def run(self) -> None:
        """
        Run the training or the testing during a certain number of steps.
        특정 에피소드 만큼 훈련 또는 테스트 실행
        """
        print('>>>>> Running <<<<<\n')
        for n in range(1, self.n_episodes+1):
            print("episode : ", n )
            self.episode = n
            self._run_one_episode()
            self.logger.save_logs()
            if n >= 0 and n % 10 == 0 :
                print(f"\n Episode {n}.. save logs to csv.... (Not Implemented)\n")
                try:
                    self.logger.generate_plots(asset_label = self.stocks_subset)
                except: 
                    print(f'Ep.{n} Generating plots failed...')
                    pass

                with open(f'{self.checkpoint_directory}/json/ep{n}_logs_{self.mode}.json','w') as f:
                    json.dump(self.logger.logs, f, indent=4, cls=NpEncoder)
                print("\n---------------------------------------------------------------\n")


