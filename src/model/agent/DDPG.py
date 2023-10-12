
import numpy as np
import torch
import torch.nn.functional as F
import copy

## inner module import 여기 손한번 보기
from model.agent.base import Agent
from model.buffer.buffer import *
from util.configures import *
from model.networks.critic import *
from model.networks.actor import *
from model.networks.pretrain_net import StackedGRU
torch.autograd.set_detect_anomaly(True)



class DDPGagent(Agent):
    '''
    Soft Actor Critic : https://arxiv.org/abs/1812.05905
    Agent 클래스를 상속받으며, Temperature가 학습과정에서 자동으로 업데이트 되는 SAC.

    Attributes:
        alpha (float): initial value of the temperature parameter,
                       initialized by sampling from a standard normal distribution. 
    '''
    def __init__(self,
                lr_alpha: float,
                mode: str = 'train',
                country = 'USA',
                *args,
                **kwargs,
                ) -> None:
        super(DDPGagent, self).__init__(*args, **kwargs)

        number_of_agent = 1
        self.name = 'DDPG'
        self.country = country
        self.noise = OUNoise(size=(self.action_space_dimension,))

        # Buffer 
        if self.buffer_name == 'PER':
            self.memory = PERBuffer(
                                    factor=self.factor,
                                    size=self.size, 
                                    input_shape=self.input_shape, 
                                    action_space_dimension=self.action_space_dimension,
                                    device=self.device
                                    )
        else:
            self.memory = ReplayBuffer(
                                        size=self.size, 
                                        input_shape=self.input_shape, 
                                        action_space_dimension=self.action_space_dimension,
                                        device=self.device
                                        )        
        
        
        self.actor = DDPGActor(lr_pi=self.lr_pi,
                                action_space_dimension= self.action_space_dimension,
                                max_actions=self.env.action_space.high,
                                input_shape=self.input_shape,
                                layer_neurons=self.layer_size,
                                network_name='MainActor',
                                checkpoint_directory_networks=self.checkpoint_directory_networks,
                                device=self.device).to(self.device)
        
        self.target_actor = DDPGActor(lr_pi=self.lr_pi,
                                action_space_dimension= self.action_space_dimension,
                                max_actions=self.env.action_space.high,
                                input_shape=self.input_shape,
                                layer_neurons=self.layer_size,
                                network_name='TargetActor',
                                checkpoint_directory_networks=self.checkpoint_directory_networks,
                                device=self.device).to(self.device)
        
        self.critic = MLPCritic3(lr_Q=self.lr_Q,
                            action_space_dimension=self.action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_size,
                            network_name='critic',
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device).to(self.device)
        
        self.target_critic = MLPCritic3(lr_Q=self.lr_Q,
                            action_space_dimension=self.action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_size,
                            network_name='targetCritic',
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device).to(self.device)

        # network list에 env.pretrain.. 추가할 것인가 말 것인가..
        self._network_list = [self.actor, self.critic]
        self._targeted_network_list = [self.target_actor, self.target_critic]
        
        # load the network params
        if self.mode == 'train': self.init_networks()
        else : self.load_networks() # mode : test or retrain

        # self._update_target_networks(tau=1) ###
        self.soft_update(self.critic, self.target_critic, tau=0)
        self.soft_update(self.actor, self.target_actor, tau=0)
    
        # freeze the target_critic
        for p in self.target_critic.parameters():
            p.requires_grad = False

            
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def learn(self, 
            step: int = 0,
            eps: float = 0.5
            ) -> None:
        """
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Params(수정필요)
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #torch.autograd.set_detect_anomaly(True)
        if self.memory.pointer < self.batch_size:
            return
        
        # Sample the trajectory data in buffer
        #### 상당히 메모리 낭빈데
        if self.buffer_name == 'PER':
            states, actions, rewards, states_, dones, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights).to(self.device)
        else :
            states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
            weights = 1
            
        # 순서 맞추기
        state = torch.tensor(states[0], dtype=torch.float).to(self.device)
        pre_state = torch.tensor(states[1], dtype=torch.float).to(self.device)
        state_ = torch.tensor(states_[0], dtype=torch.float).to(self.device)
        pre_state_ = torch.tensor(states_[1], dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device) ### 리워드가 어디서 저장된건지 확인 해보기
        dones = torch.tensor(dones).to(self.device)
        

        # ---------------------------- update critic ---------------------------- #
        action_ = self.target_actor(state_, pre_state_)
        q_ = self.target_critic.forward(state_, pre_state_, action_.detach()) 
        q_[dones] = 0
        q_target = rewards + self.gamma * q_.squeeze() ##
        q = self.critic.forward(state, pre_state, actions)
        critic_loss = Agent.MSELoss(q, q_target.detach(), weights) + torch.mean(self.critic.loss_term*0.01)
        # 나중에 수정..

        # Minimize the loss
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic.parameters(), max_norm=self.grad_clip) #0.5
        self.critic.optimizer.step()
        

        # ---------------------------- update actor ---------------------------- #
        actions_pred= self.actor.forward(state, pre_state)
        
        # version
        # actor_loss = torch.sum(-self.critic(state, pre_state, actions_pred)*weights) ## Actor 4
        actor_loss = torch.mean(-self.critic(state, pre_state, actions_pred)*weights) + torch.mean(self.actor.feat_layer.loss_term)*0.01

        # Minimize the loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.actor.parameters(), max_norm=self.grad_clip) #0.5
        self.actor.optimizer.step()
        

        # self._update_target_networks(tau=self.tau)
        self.soft_update(self.critic, self.target_critic, tau=self.tau)
        self.soft_update(self.actor, self.target_actor, tau=self.tau)

        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        
        if step % PRINT == 0 or self.env.current_step==self.env.time_length :
            print("------------------------------------------------------")
            print("state : ", state.mean())
            print("actions : ", actions_pred.mean(), actions_pred.sum())
            print("q_ : ", torch.mean(q_).item())
            print("q : ", torch.mean(q).item())
            print("qtarget : ", torch.mean(q_target))
            print("------------------------------------------------------")
            print("critic_loss : ", critic_loss)
            print("actor_loss: ", actor_loss)
            print("------------------------------------------------------")
            self.actor.lrscheduler.step(actor_loss)
            self.critic.lrscheduler.step(critic_loss)


    def choose_action(self, observation:torch.tensor) -> np.array :
        '''
        환경으로부터 관찰한 state를 기반으로 액션을 선택함.
        Args
            observation (Tuple(np.ndarray) : state of the env
        returns
            action (np.array) taken in the input state
        '''
        state, pre_state = observation
        action_ratio = self.actor.forward(state, pre_state)[0] #####

        return action_ratio 
    



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size:tuple, mu=0.0, theta=0.15, sigma=0.15, sigma_min = 0.05, sigma_decay=.975):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
