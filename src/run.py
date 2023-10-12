import gym
import numpy as np
import json
import torch
import copy

from model.agent.base import Agent
from util.logger import Logger
from util.plot import *
from util.utilities import NpEncoder, set_seed
from util.configures import *


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
                checkpoint_directory: str,
                asset_label:list,
                agent_type: str = 'default',
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
        self.asset_label = asset_label

        # self.mode : train, retrain, test
        # 이 부분을 에이전트에 둬야하려나? --> DDPG, SAC에 두기
        if self.mode != 'train':
            self.agent.load_networks()
            # 이거 왜 있는거지 ㅋㅋㅋ 이미 있는걸 갖고왔는디
            # os.makedirs(self.checkpoint_directory, exist_ok=True) 
        else :
            self.agent.init_networks()


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



    def _print_status(self, reward, action, info):
        ##----- log 쌓기
        ##----- for debugging
        print("-----------------------------------------------------------------------------------------")
        print(f"\nStep: {self.step}     Env step: {self.env.current_step}     done: {self.done}")
        print("env reward : ", reward)
        print("Cum_reward: ", self.cum_reward)
        print("new_value_portfolio: ", info['portfolio'])
        # print("-----------------------------------------------------------------------------------------")


    def _run_one_episode(self) -> None:
        self.step, self.cum_reward, done, observation = self._init_variables()

        if self.mode=='train':
            # 에피소드가 끝나는 조건에 다다를때까지
            while not done:
                action = self.agent.choose_action(observation) ###
                # with torch.no_grad():
                with torch.no_grad():
                    observation_, reward, done, info = self.env.step(action)
                    self.agent.remember(observation, action, reward, observation_, done)
                self.agent.learn(step = self.step)
                
                # Go to the next step
                observation = observation_
                self.step += 1
                self.cum_reward += reward
                self.done = done
                
                # The break point    
                if self.env.current_step % PRINT == 0 or self.env.current_step==self.env.time_length:
                    self._print_status(reward, action, info)
                    if self.env.current_step>=self.env.time_length:
                        break

            # write logs
            self.logger.logs['critic_loss'].append(self.agent.critic_loss.item())
            self.logger.logs['actor_loss'].append(self.agent.actor_loss.item())
            
                        
        else:
            # test mode
            # initializing a list to keep track of the porfolio value during the episode
            portfolio_content_history = self.env.number_of_shares.tolist() # [self.env.number_of_shares.tolist()] ### 여기 확인
            action_tslist = []
            pfvalue_tslist = []
            att_tslist = []
            
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
                attention = self.agent.actor.attention.squeeze().tolist() ###
                att_tslist.append(attention)

                if self.env.current_step % 500 == 0 or self.env.current_step>=self.env.time_length:
                    self._print_status(reward, action, info)
                    if self.env.current_step>=self.env.time_length:
                        break

            # write logs
            assert len(action_tslist) == len(pfvalue_tslist), "len(action_tslist) != len(pfvalue_tslist)"
            portfolio_content_history.append(self.env.number_of_shares.tolist())
            self.logger.logs['asset_ratio_timeseries'].append(action_tslist)    
            self.logger.logs['portfolio_value_timeseries'].append(pfvalue_tslist)
            self.logger.logs['attention_timeseries'].append(att_tslist)

            
        ##----- log 쌓기 (공통)
        self.logger.logs['reward_history'].append(self.cum_reward)
        self.logger.logs['asset_ratio_history'].append(action.tolist())
        self.logger.logs['portfolio_value'].append(self.env._get_portfolio_value())
        # self.logger.logs['attention_map'].append(self.agent.actor.attention) ## 이걸 매 에피소드에 쌓는게 의미있으려나 
        average_reward = np.mean(self.logger.logs['reward_history'][-self.cal_window:])
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)




    def run(self) -> None:
        """
        Run the training or the testing during a certain number of steps.
        특정 에피소드 만큼 훈련 또는 테스트 실행
        """
        print('>>>>> Running <<<<<\n')
        for n in range(1, self.n_episodes+1):
            # # test시 강건성을 위해서 랜덤 시드 재지정 후 여러 번의 실험 시도
            # if self.mode == 'test' and self.n_episodes>1:
            #     seed = np.random.randint(low=0, high=9999)
            #     set_seed(seed)

            print("episode : ", n )
            self.episode = n
            self._run_one_episode()
            self.logger.save_logs()

            # best reward 기록
            if self.mode=='train' and self.cum_reward > self.best_reward:
                self.best_reward = self.cum_reward
                if self.mode == 'train': 
                    # save the networks
                    self.agent.save_networks()
                    # save the pretrain models
                    for s in self.env.pretrain_net_dict:
                        pretrain_checkpoint = self.checkpoint_directory+f"/networks/pretrain/price_{s}/BestModel_price_{s}.pth"
                        if n == 1 : os.makedirs(self.checkpoint_directory+f"/networks/pretrain/price_{s}", exist_ok=True)
                        torch.save(self.env.pretrain_net_dict[s].state_dict(), pretrain_checkpoint)

            if n >= 0 and n % 10 == 0 :
                print(f"\n Episode {n}.. save logs to csv.... (Not Implemented)\n")
                # draw plots
                try:
                    self.logger.generate_plots(asset_label = self.asset_label)
                except: 
                    print(f'Ep.{n} Generating plots failed...')
                    pass

                # write logs
                with open(f'{self.checkpoint_directory}/json/ep{n}_logs_{self.mode}.json','w') as f:
                    json.dump(self.logger.logs, f, indent=4, cls=NpEncoder)
                print("\n---------------------------------------------------------------\n")






class Run_nograd():
    """
    학습 또는 테스트를 실행할 기본 클래스.
    Neural Network를 사용하지 않는 전통전략을 사용하는 경우, Test모드가 아니면 지원 X

    - _reset : 환경과 보상 이력을 초기화함
    - run : 특정 에피소드 만큼 훈련 또는 테스트 실행
    - _run_one_episode : 에피소드 하나 실행

    """

    def __init__(self,
                env: gym.Env,
                agent: Agent,
                n_episodes: int,
                checkpoint_directory: str,
                asset_label:list,
                agent_type: str = 'default',
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
        assert mode == 'test', "args.mode : Traditional Trading Methods only works on Test mode."
        
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.agent_type = agent_type
        self.sac_temperature = sac_temperature
        self.mode = mode
        self.checkpoint_directory = checkpoint_directory
        self.asset_label = asset_label

        if self.mode == 'test':
            os.makedirs(self.checkpoint_directory, exist_ok=True)

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




    def _print_status(self, reward, action, info):
        ##----- log 쌓기
        ##----- for debugging
        print("-----------------------------------------------------------------------------------------")
        print(f"\nStep: {self.step}     Env step: {self.env.current_step}     done: {self.done}")
        # print('action_ratio: ', action.mean(), action.sum(), action.std())
        print("env reward : ", reward)
        print("Cum_reward: ", self.cum_reward)
        print("new_value_portfolio: ", info['portfolio'])
        print("-----------------------------------------------------------------------------------------")


    def _one_episode_bnh(self) -> None:
        '''
        이게 맞냐...
        '''
        assert self.agent.name == 'BuyAndHold', "wrong mode"

        # make log containers 
        portfolio_content_history = [self.env.number_of_shares.tolist()] 
        action_tslist = []
        pfvalue_tslist = []        

        # init variables
        self.step, self.cum_reward, done, observation = self._init_variables()
        action = self.agent.pre_action.squeeze()
        init_portfolio_value = portfolio_value = self.env._get_portfolio_value()
        n_shares = torch.floor(action[:-1] * portfolio_value / self.env.stock_prices)
        cash = portfolio_value-sum(n_shares*self.env.stock_prices)


        while not done:
            observation_, _, done, _= self.env.step(action)
            cash *= self.env.daily_bank_rate 
            new_portfolio_value = sum(n_shares*self.env.stock_prices).item() + cash

            # compute new action, reward
            action[:-1] = (n_shares*self.env.stock_prices) / new_portfolio_value
            action[-1] = 1 - action[:-1].sum(dim=-1) # cash
            reward = np.log(new_portfolio_value.item()/portfolio_value) * 10 \
                + np.log(new_portfolio_value.item()/init_portfolio_value) * 5 
            
            portfolio_value = new_portfolio_value.item()
            info = {'portfolio': new_portfolio_value}
            
            # Go to the next step
            observation = observation_
            self.step += 1
            self.cum_reward += reward
            self.done = done         

            # write timeseries logs
            action = action.view(self.env.action_space_dimension) # 저장하기 용이하도록
            action_tslist.append(action.tolist())
            pfvalue_tslist.append(portfolio_value)

            if self.env.current_step % 500 == 0 or self.env.current_step >= self.env.time_length:
                self._print_status(reward, action, info)
                if self.env.current_step >= self.env.time_length:
                    break

            # write logs
            assert len(action_tslist) == len(pfvalue_tslist), "len(action_tslist) != len(pfvalue_tslist)"
            portfolio_content_history.append(n_shares.tolist())

            
        ##----- log 쌓기 (공통)
        self.logger.logs['asset_ratio_timeseries'].append(action_tslist) ##
        self.logger.logs['portfolio_value_timeseries'].append(pfvalue_tslist)
        self.logger.logs['reward_history'].append(self.cum_reward)
        self.logger.logs['asset_ratio_history'].append(action.tolist())
        self.logger.logs['portfolio_value'].append(self.env._get_portfolio_value())
        average_reward = np.mean(self.logger.logs['reward_history'][-self.cal_window:])

        # best reward 기록
        if average_reward  > self.best_reward:
            self.best_reward = average_reward
        

        # save the logs
        with open(f'{self.checkpoint_directory}/plot_{self.agent.name}.json','w') as f:
            json.dump(self.logger.logs, f, cls=NpEncoder, indent=4)
        
        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)



    
    def _run_one_episode(self) -> None:
        '''
        *** self.logs에 어떻게 비교 agent를 같이 시뮬레이션 할 수 있을까 고민하기 ***
        '''
        self.step, self.cum_reward, done, observation = self._init_variables()

        # test mode
        # initializing a list to keep track of the porfolio value during the episode
        portfolio_content_history = [self.env.number_of_shares.tolist()]  ### 여기 확인
        action_tslist = []
        pfvalue_tslist = []
        att_tslist = []
        
        while not done:
            action = self.agent.choose_action(observation) # torch.Size([1, 1, 5])
            observation_, reward, done, info = self.env.step(action) 
            
            # Go to the next step
            observation = observation_
            self.step += 1
            self.cum_reward += reward
            self.done = done

            # write timeseries logs
            action = action.view(self.env.action_space_dimension) # 저장하기 용이하도록
            action_tslist.append(action.tolist())
            pfvalue_tslist.append(self.env._get_portfolio_value())


            if self.env.current_step % 500 == 0 or self.env.current_step >= self.env.time_length:
                self._print_status(reward, action, info)
                if self.env.current_step >= self.env.time_length:
                    break

            # write logs
            assert len(action_tslist) == len(pfvalue_tslist), "len(action_tslist) != len(pfvalue_tslist)"
            portfolio_content_history.append(self.env.number_of_shares.tolist())

            
        ##----- log 쌓기 (공통)
        self.logger.logs['asset_ratio_timeseries'].append(action_tslist) ##
        self.logger.logs['portfolio_value_timeseries'].append(pfvalue_tslist)
        self.logger.logs['reward_history'].append(self.cum_reward)
        self.logger.logs['asset_ratio_history'].append(action.tolist())
        self.logger.logs['portfolio_value'].append(self.env._get_portfolio_value())
        average_reward = np.mean(self.logger.logs['reward_history'][-self.cal_window:])

        # best reward 기록
        if average_reward  > self.best_reward:
            self.best_reward = average_reward
            # if self.mode == 'train': self.agent.save_networks() # no grad

        # save the logs
        with open(f'{self.checkpoint_directory}/plot_{self.agent.name}.json','w') as f:
            json.dump(self.logger.logs, f, cls=NpEncoder, indent=4)
        
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

            if self.agent.name == 'BuyAndHold': self._one_episode_bnh()
            else: self._run_one_episode()
            
            self.logger.save_logs()
            if n >= 0 and n % 10 == 0 :
                print(f"\n Episode {n}.. save logs to csv.... (Not Implemented)\n")
                try:
                    self.logger.generate_plots(asset_label = self.asset_label)
                except: 
                    print(f'Ep.{n} Generating plots failed...')
                    pass

                with open(f'{self.checkpoint_directory}/json/ep{n}_logs_{self.mode}.json','w') as f:
                    json.dump(self.logger.logs, f, indent=4, cls=NpEncoder)
                print("\n---------------------------------------------------------------\n")
