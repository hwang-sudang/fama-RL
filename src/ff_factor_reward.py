# https://github.com/RobertYip/Python-Fama-French-Quant-model/blob/master/Fama%20French%20Model.ipynb

import os
import pandas as pd
import numpy as np
import math
import torch
from datetime import datetime

import matplotlib.pyplot as plt
import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
pd.options.mode.chained_assignment = None  # default='warn'


class FamaFrench():
    '''
    Fama French Factor 계산법으로 Environment 에서 리워드를 계산함. 
    여기에서 학습이 가능한 것은 

    공통 입력 : 안댁스 정보를 어떻게 해결하지? 그걸 고민하기 --> risk_free rate 때문에.
    # 착각금지 : 리플레이 버퍼 = 배치사이즈랑 똑같이 생각해야함. 
                즉, 우리는 과거 20일의 데이터를 보고서 다음날의 디시전을 내리는거니까, 마지막 시점의 factor들만 필요할 것!

    - factor_t (t 시점의 팩터 데이터) : torch.tensor 
                    / t 시점의 close, 각 팩터 관련 데이터 (평균 log return, 각 데이터의 BTM, 각 데이터 시총 )
                    / 마지막 시점의 데이터만 갖고 오기. 즉, 차원은 (n,) 또는 (n, 1)

    - risk_free(t) : torch.tensor / t 시점의 무위험 이자율,
    - action (portfolio weight): torch.tensor(1,30)

    '''
    


    def __init__(self, startdate, enddate, window=20):
        '''
        Args 
        - device (str) : 연산 시 gpu, cpu 중 어떤 것을 쓸지
        - action (torch.tensor) : policy에서 계산한 portfolio weight
        '''

        # args로 받는 것도 생각해보기
        # 
        self.initial_rp = 0
        self.initial_SMB = 0
        self.initial_HML = 0

        self.window = window
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weighted_value = 0 
        self.startdate = startdate
        self.enddate = enddate
        
        # define risk-free dataframe
        self.rf = pd.read_csv("/home/ubuntu2010/바탕화면/famafrench_data/factor_pre/Risk_Free.csv", parse_dates=['Unnamed: 0'], index_col=0)
        self.rf.index = self.rf.index.astype(str)
        self.rf.index = pd.to_datetime(self.rf.index)
        self.rf = self.rf.loc[startdate:enddate]

        self.BTM_df = None
        self.market_cap_df = None



    def calc_factor(self, start, agent:str='SMB'):
        '''
        start : 팩터에 해당하는 데이터를 갖고오는 일자

        '''


        return    

    
    def calcMktPrem(self) -> torch.tensor :
        '''
        Market_Premium 계산하기

        Attributes

        Output
        '''
        self.exp_pf_R = torch.mean(self.log_R) # portfolio's mean log return, scalar
        self.mktprem = self.exp_pf_R - self.rf # scalar tensor
        
        return self.mktprem



    
    def calcSMB(self, SQuantile:int = 0.3, LQuantile:int = 0.7):
        """
        Returns SMB for FF
        """
        self.marketcap = self.factor["market_cap"]
        self.close = self.factor["close"]

        small_q = torch.quantile(self.marketcap, SQuantile, interpolation='lower')
        large_q = torch.quantile(self.marketcap, LQuantile, interpolation='lower')

        # 차원 안맞을 수도 있음..
        scap_tensor = torch.empty(math.floor(30*SQuantile), )
        lcap_tensor = torch.empty(math.floor(30*LQuantile), )

        ## Assigns stock size based on market cap
        # 이렇게 하자! scap 에 넣을때 애초에 그 값이랑 같으면 weight랑 같이 곱해서 넣어버리기
        # 그러려면 필요한 정보 : 각 자산의 weight 정보
        for idx in range(len(self.log_R)):
            if self.marketcap[idx] <= small_q :
                scap_tensor.cat(self.log_R[idx]*self.action[idx])
            elif self.marketcap[idx] >= large_q : 
                lcap_tensor.cat(self.log_R[idx]*self.action[idx])
            else : 
                pass

        
        #Calculates average return of stocks in portfolio subset based on size
        return_smb = torch.mean(scap_tensor) - torch.mean(lcap_tensor)
        return return_smb




    def calcHML(self, SQuantile:int = 0.3, LQuantile:int = 0.7):
        """
        Returns HML for FF
        Uses inverse of P/B as proxy for Book/Mkt
        """

        self.BTM = self.factor["BTM"]
        self.close = self.factor["close"]

        small_q = torch.quantile(self.BTM, SQuantile, interpolation='lower')
        large_q = torch.quantile(self.BTM, LQuantile, interpolation='lower')

        # 차원 안맞을 수도 있음..
        highBTM_tensor = torch.empty( )
        lowBTM_tensor = torch.empty( )

        ## Assigns stock size based on market cap
        # 이렇게 하자! scap 에 넣을때 애초에 그 값이랑 같으면 weight랑 같이 곱해서 넣어버리기
        # 그러려면 필요한 정보 : 각 자산의 weight 정보
        for idx in range(len(self.log_R)):

            if self.marketcap[idx] <= small_q :
                # 고평가주, 성장주
                lowBTM_tensor.cat(self.log_R[idx]*self.action[idx])
            elif self.marketcap[idx] >= large_q : 
                # 저평가주 , 가치주
                highBTM_tensor.cat(self.log_R[idx]*self.action[idx])
            else : 
                pass

        
        #Calculates average return of stocks in portfolio subset based on size
        #Returns SMB based on definition
        return_hml = torch.mean(lowBTM_tensor) - torch.mean(highBTM_tensor)
        return return_hml

    

    
    def calcFactorReturn(self, factor:dict, action:torch.tensor, agent:str ='SMB'):
        '''
        각 에이전트의 취지에 적절한 return 계산함

        Option
        - SMB, HML, premium

        Attributes

        - self.factor (dict) :  fama-french를 계산하기 위한 팩터임
            * 모든 데이터는 30개 자산 만큼 있음. 따라서 각 요소들은 (30,)거나 (30,1) 사이즈
            - log_R** : t 대비 t+1 시점의 로그 리턴
            - BTM : t시점의 Book to market
            - market_cap : t 시점의 시가총액 정보
            - rf (scalar): t 시점의 무위험 이자율 

        - self.action (torch.tensor) : policy에서 계산한 portfolio weight
        
        '''

        self.factor = factor 
        self.action = action
        
        # 나중에 텐서로 변환해주기. 일단은 텐서라는 가정하에 작성
        self.log_R = self.factor["log_R"] # for mktreturn 때문에
        self.rf = self.factor["rf"] #scalar
        self.marketcap = self.factor["market_cap"]
        self.BTM = self.factor["BTM"]
        # self.new_R # 이건 close의 st 대비 st+1 
        
        # self.past_value = self.weighted_value # deepcopy로 기왕이면, 전기 weighted value 정보 -> return 내보내는 데에서 필요할 듯한데..
        

        if agent == 'SMB':
            reward = self.calcSMB()
        elif agent == 'HML':
            reward = self.calcHML()
        elif agent == 'premium':
            reward = self.calcMktPrem()
        else:
            raise Exception("Wrong Agent Factor Type. Choose : SMB, HML, premium. ")
    
        return reward



if __name__ == "__main__":
    ff = FamaFrench()