# https://github.com/MatthieuSarkis/Portfolio-Optimization-and-Goal-Based-Investment-with-Reinforcement-Learning/blob/master/src/get_data.py

import os
from matplotlib.cbook import strip_math
import pandas as pd
from pathlib import Path
from typing import List
import yfinance as yf




class DataFetcher():
    '''
    데이터를 yfinance에서 다운로드 받아 갖고 오는 클래스
    또는 기존 데이터를 갖고 온다.
    '''

    def __init__(self,
                stock_symbols: List[str],
                start_date: str = '2000-01-01',
                end_date: str = '2021-12-31',
                directory_path: str = 'data',
                ) -> None:

        Path(directory_path).mkdir(parents=True, exist_ok=True)

        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.directory_path = directory_path

    
    def fetch_and_merge_data(self) -> None:
        """
        다운로드 파일 경로가 없다 -> 데이터가 없는 것으로 판단하여 stock list에서 관련 데이터 다운로드
        경로 있다 -> 이미 파일이 다운로드 되어 있다고 봄 -> dataframe으로 각 열에 각 자신의 가격 정보가 들어가도록 df 만듦.
        이후 csv로 path 위치에 stocks.csv로 저장함

        추후에 개별 자산을 병합할 때 이 함수를 바꿔서 쓰면 좋을 듯.
        """
        final_df = None

        for stock in self.stock_symbols:
            
            file_path = os.path.join(self.directory_path, "{}.csv".format(stock))
            if not os.path.exists(file_path):
                data = yf.download(stock, start=self.start_date, end=self.end_date)
                if data.size > 0:
                    data.to_csv(file_path)
                    file = open(file_path).readlines()
                    if len(file) < 10:
                        os.system("rm "+file_path)

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                stock_name = file_path.split('/')[1].split('.')[0] ###
                df["Name"] = stock_name

                if final_df is None:
                    final_df = df
                else: 
                    final_df = final_df.append(df, ignore_index=True)
                    
                    os.system('rm '+file_path)
        
        path = os.path.join(self.directory_path, 'stocks.csv')
        final_df.to_csv(path, index=False)







class Preprocessor():
    """
    데이터 전처리
    1. 종가만 모으는 함수
    2. missing value 처리 함수
    """

    def __init__(self,
                df_directory: str = 'data',
                file_name: str = 'stock.csv',
                ) -> None :

        self.df_directory = df_directory
        path = os.path.join(df_directory, file_name)
        self.df = pd.read_csv(path)


    def collect_close_prices(self) -> pd.DataFrame:
        '''
        데이터들 중에 close 종가만 갖고 오는 함수
        '''
        self.df["Date"] = pd.to_datetime(self.df['Date'])
        dates = pd.date_range(self.df['Date'].min(), self.df['Date'].max())
        stocks = self.df['Name'].unique()
        close_prices = pd.DataFrame(index=dates)

        for stock in stocks:
            df_temp = self.df[self.df['Name']==stock]
            df_temp2 = pd.DataFrame(data=df_temp['Close'].to_numpy(), index=df_temp['Date'], columns=[stock])
            close_prices = pd.concat([close_prices, df_temp2], axis=1)
        self.df = close_prices
        return close_prices

    
    def handle_missing_values(self) -> pd.DataFrame:
        '''
        1. NA 모두 drop 
        -> 2. NA를 앞에 데이터로부터 채우기 
        -> 3. 채워지지 않은 값은 뒤의 데이터를 갖고와서 채우기
        
        '''

        self.df.dropna(axis=0, how='all', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df.to_csv(os.path.join(self.df_directory, 'close.csv'))
        return self.df

    


def load_data(initial_date: str,
            final_date: str,
            tickers_subset: str,
            mode: str = 'test') -> pd.DataFrame:
    
    """
    데이터를 전체적으로 로드하는 함수.
    - fetcher ->  DataFetcher() : 데이터가 없는 경우 다운로드하여 df 로 만드는 과정
    - preprocessor -> Preprocessor() : 종가만 모여있는 데이터 추출하는 코드
    - 
    """


    text_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    with open(f'{text_dir}/portfolios_and_tickers/tickers_S&P500.txt') as f:
        stocks_symbols = f.read().splitlines()

    # 데이터가 없는 경우 다운로드하여 df 로 만드는 과정
    if not os.path.exists('data/'):

        print('\n>>>>> Fetching the data <<<<<')
        fetcher = DataFetcher(stock_symbols=stocks_symbols,
                              start_date=initial_date,
                              end_date=final_date,
                              directory_path="data")        
        
        fetcher.fetch_and_merge_data()

    
    
    # 종가만 모여있는 데이터 추출하는 코드
    if not os.path.exists('data/close.csv'):
        
        print('>>>>> Extracting close prices <<<<<')
        
        preprocessor = Preprocessor(df_directory='data',
                                    file_name='stocks.csv')
    
        df = preprocessor.collect_close_prices()
        df = preprocessor.handle_missing_values()        

    
    
    else:
        # 종가 데이터가 잘 다운로드 된 경우, 바로 파일 로드해서 갖고 옴.
        print('\n>>>>> Reading the data <<<<<')
        
        df = pd.read_csv('data/close.csv', index_col=0)

        # 초점을 맞추고 싶은 티커를 선택하고 티커 텍스트 파일 목록에서 읽습니다.
        with open(tickers_subset) as f:
            stocks_subset = f.read().splitlines()
            stocks_subset = [ticker for ticker in stocks_subset if ticker in df.columns]

        df = df[stocks_subset]

        time_horizon = df.shape[0] #1511
        print("time_horizon : ", time_horizon)

        #
        if mode == 'train':
            df = df.iloc[:3*time_horizon//4, :]
        else:
            df = df.iloc[3*time_horizon//4:, :]
        

        return df


def load_diy_data(data_dir:str,
            initial_date: str,
            final_date: str,
            tickers_subset: str,
            mode: str = 'test') -> pd.DataFrame:
    '''
    Args
    ticker_subset : ticker 가 적혀있는 텍스트 데이터 경로
                  : args asset_to_trade 정보임
    '''
    
    data_dir = '/home/ubuntu2010/바탕화면/famafrench_data/stockdata/30stocks_2000'
    data_container = {}

    
    with open(tickers_subset) as f:
        stocks_subset = f.read().splitlines()
        stocks_subset = [ticker for ticker in stocks_subset]

    for ticker in stocks_subset:
        data = pd.read_csv(data_dir+f'/{ticker}.csv', parse_dates=['date'], index_col=0)
        if 'market cap' in data.columns:
            data.drop('market cap', axis=1, inplace=True)
        
        data.columns = ['open', 'high', 'low', 'close', 'volume', 'market_cap','BTM', 'number of shares']
        data['ticker'] = ticker
        data_container[ticker] = data

    final_data = pd.DataFrame(columns=data.columns)
    
    for tic in data_container:
        final_data = pd.concat([final_data, data_container[tic]])

    final_data = final_data.loc[initial_date : final_date]

    # final_data : final_data 

    ################ 나중에 작업하기
    time_horizon = len(final_data)
    if mode == 'train':
        final_data = final_data.iloc[:3*time_horizon//4, :]
    else:
        final_data = final_data.iloc[3*time_horizon//4:, :]

    return final_data