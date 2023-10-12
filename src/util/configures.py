import os

# 파일 경로 설정 : 모듈 폴더가 설정되어있는 상위 폴더 주소
BASE_DIR = os.path.abspath(__file__).split('/src')[0]
DATA_DIR = os.path.abspath(__file__).split('/trading')[0]+'/famafrench_data'
FACTOR = 'whole' #'rep'
EMD_FILE = f'emd_{FACTOR}' #대표 데이터:rep, 이런식으로 데이터마다 이름이 다름
PRINT = 2000 #step 몇번마다 Agent learning status 프린팅?

# # ORIGINAL DATA
# START_DATE = '2000-02-17'
# TRAIN_END = '2015-05-28'
# VAL_END = '2016-10-28'
# TEST_END = '2021-12-31'


# # PRE-TRAIN
# PRE_TRAIN = '1998-12-23' # train start
# PRE_VAL = '2015-06-29'
# PRE_TEST = '2017-06-29'
# PRE_END = '2021-12-31'