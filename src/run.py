

import gym
import numpy as np
from sklearn.decomposition import non_negative_factorization
from sklearn.preprocessing import StandardScaler

from src.agents import Agent
from src.logger import Logger



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
                scaler: StandardScaler,
                checkpoint_directory: str,
                sac_temperature: float = 1.0,
                mode: str = 'test',
                ) -> None:

        """Constructor method of the class Run.
        
        Args:
            env (gym.Env): trading environment in which the agent evolves
            agent (Agent): Soft Actor Critic like agent
            n_episodes (int): total number of episodes for training, or testing (since the policy is stochastic)
            agent_type (str): name of the type of agent for saving files
            scaler (StandardScaler): already fitted sklearn standard scaler, used as a preprocessing step
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
        self.scaler = scaler
        self.checkpoint_directory = checkpoint_directory

        if self.mode == 'test':
            self.agent.load_networks()

        # step, episode, best_reward 초기값 정의
        self.step = None
        self.episode = None
        self.best_reward = None

        # logger ON
        self.logger = Logger(mode=self.mode,
                            checkpoint_directory=self.checkpoint_directory)


        self._reset()




    def _reset(self) -> None:
        """
        initialize the ennvironment and the reward history.
        환경과 보상 이력을 초기화함
        """

        self.step = 0
        self.episode = 0
        self.best_reward = float('-Inf')




    def run(self) -> None:
        """
        Run the training or the testing during a certain number of steps.
        특정 에피소드 만큼 훈련 또는 테스트 실행
        """

        print('>>>>> Running <<<<<\n')

        for _ in range(self.n_episodes):
            self._run_one_episode()
            self.logger.save_logs()




    def _run_one_episode(self) -> None:
        """
        에이전트는 환경에서 한 걸음 나아가 학습 모드인지 학습합니다.
        """

        self.logger.set_time_stamp(1)   

        reward: float = 0
        done: bool = False
        observation = self.env.reset()
        # 전처리 단계에서 이미 fitting 되어 있는 scaler로 표준화 실행
        observation = self.scaler.transform([observation])[0]
        # 초기 포트폴리오 값으로 초기화 시킴
        self.logger._store_initial_value_portfolio(self.env._get_portfolio_value())

        # initializing a list to keep track of the porfolio value during the episode
        # 에피소드 동안 포트폴리오 가치를 추적(track)하기 위해 list 초기화
        if self.mode == 'test':
            portfolio_value_history = [self.env._get_portfolio_value()]
            portfolio_content_history = [self.env.number_of_shares]

        # 에피소드가 끝나는 조건에 다다를때까지
        while not done:
            action = self.agent.choose_action(observation)
            observation_, reward, done, _ = self.env.step(action)
            observation_ = self.scaler.transform([observation_])[0]

            # rescale the reward to account for the relative normalization between the 
            # expected return and the entropy term in the loss function
            if self.agent_type == 'anual_temperature': ####
                reward *= self.sac_temperature
                
            self.step += 1
            reward += reward

            if self.mode == 'test':
                portfolio_value_history.append(self.env._get_portfolio_value())
                portfolio_content_history.append(self.env.number_of_shares)

            self.agent.remember(observation, action, reward, observation_, done)

            if self.mode == 'train':
                self.agent.learn(self.step)

            observation = observation_

        self.logger.logs['reward_history'].append(reward)
        average_reward = np.mean(self.logger.logs['reward_history'][-50:])

        
        
        if self.mode == 'test':
            self.logger.logs['portfolio_value_history_of_histories'].append(portfolio_value_history)
            self.logger.logs['portfolio_content_history_of_histories'].append(portfolio_content_history)

        self.episode += 1

        self.logger.set_time_stamp(2)
        self.logger.print_status(episode=self.episode)

        # best reward 기록
        if average_reward  > self.best_reward:
            self.best_reward = average_reward
            if self.mode == 'train':
                self.agent.save_networks()

