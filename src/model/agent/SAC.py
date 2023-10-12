'''
# copyright notice
# https://github.com/MatthieuSarkis/Portfolio-Optimization-and-Goal-Based-Investment-with-Reinforcement-Learning/blob/master/src/agents.py
'''

import gym
import numpy as np
import os
import torch
import torch.nn.functional as F

## inner module import 여기 손한번 보기
from model.agent.base import Agent
from util.configures import *
from model.networks.critic import *
from model.networks.actor import *
torch.autograd.set_detect_anomaly(True)


class SACagent(Agent):
    '''
    Soft Actor Critic : https://arxiv.org/abs/1812.05905
    Agent 클래스를 상속받으며, Temperature가 학습과정에서 자동으로 업데이트 되는 SAC.

    Attributes:
        alpha (float): initial value of the temperature parameter,
                       initialized by sampling from a standard normal distribution. 
    '''
    def __init__(self,
                lr_alpha: float,
                pretrain_opt:str ='train',
                country = 'USA',
                *args,
                **kwargs,
                ) -> None:
        super(SACagent, self).__init__(*args, **kwargs)
        self.alpha = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample().to(self.device) # Z분포에서 알파를 일단 추출
        self.target_entropy = -torch.tensor(self.action_space_dimension, dtype=float).to(self.device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha, weight_decay=1e-6)
        # self.criterion = Agent.MSELoss() #torch.nn.HuberLoss()

        self.actor = SACActor(lr_pi=self.lr_pi,
                    action_space_dimension= self.action_space_dimension,
                    max_actions=self.env.action_space.high,
                    input_shape=self.input_shape,
                    layer_neurons=self.layer_size,
                    network_name='SAC',
                    checkpoint_directory_networks=self.checkpoint_directory_networks,
                    device=self.device).to(self.device)
        
        self.critic = TwinnedQNet(lr_Q=self.lr_Q,
                            action_space_dimension=self.action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_size,
                            network_name='critic',
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device).to(self.device)

        self.target_critic = TwinnedQNet(lr_Q=self.lr_Q,
                            action_space_dimension=self.action_space_dimension,
                            input_shape=self.input_shape,
                            layer_neurons=self.layer_size,
                            network_name='targetCritic',
                            checkpoint_directory_networks=self.checkpoint_directory_networks,
                            device=self.device).to(self.device)


        self._network_list = [self.actor, self.critic]
        self._targeted_network_list = [self.target_critic]

        # load the network params
        if self.mode == 'train': self.init_networks()
        else : self.load_networks() # mode : test or retrain

        # freeze the target_critic and update 수동으로
        # # self._update_target_networks(tau=1) ###
        self.soft_update(self.critic, self.target_critic, tau=0)

        for p in self.target_critic.parameters():
            p.requires_grad = False
        



    def learn(self, 
            step: int = 0,
            eps: float = 0.5
            ) -> None:
        """
        - step : 현재 스텝 정보
        - eps : cls loss 반영 비율, epsilon

        ## self.memory.sample -> from Run._run_one_episode
        """
        #torch.autograd.set_detect_anomaly(True)
        if self.memory.pointer < self.batch_size:
            return
        
        
        # Sample the trajectory data in buffer
        if self.buffer_name == 'PER':
            states, actions, rewards, states_, dones, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights).to(self.device)
        else :
            states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
            weights = 1
        

        # 순서 맞추기; 
        state = torch.tensor(states[0], dtype=torch.float).to(self.device)
        pre_state = torch.tensor(states[1], dtype=torch.float).to(self.device)
        state_ = torch.tensor(states_[0], dtype=torch.float).to(self.device)
        pre_state_ = torch.tensor(states_[1], dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device) ### 리워드가 어디서 저장된건지 확인 해보기
        dones = torch.tensor(dones).to(self.device)

        # CRITIC UPDATE
        # Compute target Q
        action_, log_probabilities_ = self.actor.sample(state=state_, prestate=pre_state_, reparameterize=True) ##### 
        q1_, q2_ = self.target_critic.forward(state_, pre_state_, action_) 
        target_soft_value_ = (torch.min(q1_, q2_) - (self.alpha.detach() * log_probabilities_))
        target_soft_value_[dones] = 0
        q_target = rewards + self.gamma * target_soft_value_.squeeze() ##

        # Compute Main Q
        q1, q2 = self.critic.forward(state, pre_state, actions)  # * weights 
        critic_1_loss = 0.5 * Agent.MSELoss(q1, q_target.detach(), weights)  
        critic_2_loss = 0.5 * Agent.MSELoss(q2, q_target.detach(), weights) 
        critic_loss = (critic_1_loss + critic_2_loss) + torch.mean(self.critic.loss_term) # TD-error

        # update Q
        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip) # clipping Q gradients
        self.critic.optimizer.step()
        
        if self.buffer_name == 'PER':
            self.memory._update_priorities()

        if step % self.delay == 0:
            # policy update
            actions, log_probabilities = self.actor.sample(state, pre_state, reparameterize=True)
            q1_, q2_ = self.target_critic.forward(state, pre_state, actions)
            critic_value = torch.min(q1_, q2_)

            actor_loss = (self.alpha.detach()*log_probabilities*0.1) - critic_value 
            actor_loss = torch.mean(actor_loss.view(-1)* weights + self.actor.feat_layer.loss_term*0.01)
            self.actor.optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip)
            actor_loss.backward(retain_graph=True) 
            self.actor.optimizer.step()

            #################### lr Scheduler 설정 ####################
            ## critic lrscheduler 설정
            # self.critic.lrscheduler.step(critic_loss)
            # self.critic_2.lrscheduler.step(critic_2_loss)
            
            ## actor lrscheduler 설정
            self.actor.lrscheduler.step(actor_loss)
            ###########################################################

            # Temperature update
            log_alpha_loss = -(self.log_alpha*(log_probabilities + self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            log_alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.log_alpha, max_norm=self.grad_clip)
            self.log_alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            # Exponentially Smoothed copy to the target critic networks
            # self._update_target_networks()
            self.soft_update(self.critic, self.target_critic, tau=self.tau)
            self.actor_loss = actor_loss
            self.critic_loss = critic_loss

            if step % PRINT == 0 or self.env.current_step==self.env.time_length :
                # print status
                print("------------------------------------------------------")
                print("state : ", state.mean())
                print("actions : ", actions.mean(), actions.sum())
                print("q1 : ", torch.mean(q1).item())
                print("q2 : ", torch.mean(q2).item())
                print("qtarget : ", torch.mean(q_target))
                print("------------------------------------------------------")
                print("critic_loss : ", critic_loss.item())
                print("critic losses : ", critic_loss.item(), critic_2_loss.item())
                print("------------------------------------------------------")
                print("actor_loss = (1-rambda)*(self.alpha * log_probabilities - critic_value) + rambda*(cls_loss) \n")
                print("actor_loss: ", actor_loss)
                print("self.alpha: ", self.alpha)
                print("log_probabilities: ", log_probabilities.mean())
                print("------------------------------------------------------\n")



    def choose_action(self, observation:torch.tensor) -> torch.tensor :
        '''
        환경으로부터 관찰한 state를 기반으로 액션을 선택함.
        Args
            observation (Tuple(torch.tensor)) : state of the env
        returns
            action (torch.tensor) taken in the input state
        '''
        state, pre_state = observation
        action_ratio, _ = self.actor.sample(state, pre_state, reparameterize = True)
        action_ratio = action_ratio[0]
        # action_ratio = action_ratio.cpu().detach().numpy()[0]
        return action_ratio 


