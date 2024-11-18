import os
import re
import torch
import glob
#모델 학습 및 저장 
import pickle 
import multiprocessing
import random 
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import Tensor 

from tqdm.notebook import tqdm
from types import SimpleNamespace
from typing import Callable, Optional

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
device = torch.device("cpu")
import warnings 
warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))



### 기본 설정 및 함수 
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42) 

def normalize_data(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def inverse_normalize(data, scaler):
    return scaler.inverse_transform(data)

def interpolate_zeros(df, column):
    col = df[column]
    # 값이 0인 위치를 찾음
    zeros = col == 0
    # 0인 값들을 NaN으로 대체
    col[zeros] = np.nan
    # 보간 수행
    col.interpolate(method='linear', inplace=True, limit_direction='both')
    # 결과를 데이터프레임에 반영
    df[column] = col
    
품목_리스트 = ["감자 수미", "무", "양파", "배추", "대파(일반)", "건고추", "깐마늘(국산)", "사과", "상추", "배"]
group1 = ['배추', '무', '양파', '감자 수미', '대파(일반)']
group2 = ['건고추', '깐마늘(국산)']
group3 = ['상추', '사과', '배']

item_columns = {
    "감자 수미": ["YYYYMMSOON", "평균가격(원)"],
    "무": [ "YYYYMMSOON","평균가격(원)"],
    "양파": ["YYYYMMSOON","평균가격(원)"],

    "배추": ["YYYYMMSOON", "평균가격(원)"],
    "대파(일반)": ["YYYYMMSOON", "평균가격(원)"],

    "건고추": ["YYYYMMSOON", "평균가격(원)"],
    "깐마늘(국산)": [ "YYYYMMSOON","평균가격(원)"],
    "사과": ["YYYYMMSOON" , "평균가격(원)"],
    
    "상추": ["YYYYMMSOON","평균가격(원)"],

    "배": [ "YYYYMMSOON","평균가격(원)"]
}


# 사용 예시
selected_dome = ['감자_수미_100000', '대파_대파(일반)_100000', '마늘_깐마늘_100000', 
                 '무_기타무_100000', '배_신고_100000', '배추_기타배추_100000', 
                 '상추_포기찹_100000']
dome_items = ['감자 수미', '대파(일반)', '깐마늘(국산)', '무', '배', '배추', '상추']
dome_cols = [
    '감자_수미_100000_경매 건수', '마늘_깐마늘_100000_총반입량(kg)', '대파_대파(일반)_100000_총반입량(kg)', 
    '배추_기타배추_100000_총반입량(kg)', '상추_포기찹_100000_경매 건수', '상추_포기찹_100000_고가(20%) 평균가', 
    '배_신고_100000_고가(20%) 평균가', '무_기타무_100000_평균가(원/kg)'
]


       

def get_deal_info():
    # 스크립트 파일의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 데이터 파일의 경로 설정
    apple_file_path = os.path.join(script_dir, '../data/extradata/사과_확정거래물량.csv')
    pear_file_path = os.path.join(script_dir, '../data/extradata/배_확정거래물량.csv')

    # 사과 데이터 로드 및 전처리
    사과_deal_info = pd.read_csv(apple_file_path, encoding='cp949')
    사과_deal_info = 사과_deal_info[사과_deal_info['품목명'] == '후지']
    사과_deal_info = 사과_deal_info.rename(columns={col: f'사과_{col}' for col in 사과_deal_info.columns if col not in ['거래일자', '품목명']})
    사과_deal_info = 사과_deal_info[['거래일자', '사과_금액', '사과_평년 반입량 증감률(%)']]

    # 배 데이터 로드 및 전처리
    배_deal_info = pd.read_csv(pear_file_path, encoding='cp949')
    배_deal_info = 배_deal_info.rename(columns={col: f'배_{col}' for col in 배_deal_info.columns if col not in ['거래일자', '품목명']})
    배_deal_info = 배_deal_info[['거래일자', '배_반입량']]

    # 날짜 형식 변환 함수 정의
    def format_date(row):
        year = row[:4]  # 연도 (예: '2023')
        month = row[5:7]  # 월 (예: '01')
        period = row[8]  # 주기 ('상', '중', '하')
        return f"{year}{month}{period}순"

    # 날짜 형식 변환 적용
    사과_deal_info['거래일자'] = 사과_deal_info['거래일자'].apply(format_date)
    배_deal_info['거래일자'] = 배_deal_info['거래일자'].apply(format_date)

    # 거래일자 기준으로 병합하여 하나의 데이터프레임으로 결합
    combined_deal_info = pd.merge(사과_deal_info, 배_deal_info, on='거래일자', how='outer')

    return combined_deal_info
       
deal_info = get_deal_info()



def jointmarket_filter(df):
    # 필요한 컬럼만 읽어서 메모리 사용 최적화
    df = df[['공판장코드', '품목명', '품종명', '등급코드', '공판장명', 'YYYYMMSOON', '경매 건수', '총반입량(kg)']]

    mask = (
        ((df['품목명'] == '대파') & (df['품종명'] == '대파(일반)') & (df['등급코드'] == 11) & (df['공판장명'] == '*전국농협공판장')) |
        ((df['품목명'] == '무') & (df['품종명'] == '기타무') & (df['등급코드'] == 11) & (df['공판장명'] == '*전국농협공판장'))
    )
    df = df[mask]
    df['item'] = df['공판장코드'].astype(str) + '_' + df['품목명'] + '_' + df['품종명'] + '_' + df['등급코드'].astype(str)
    df = df[['item', 'YYYYMMSOON', '경매 건수', '총반입량(kg)']]
    
    # 피벗 테이블 생성
    df_pivot = df.pivot_table(index='YYYYMMSOON', columns='item', values=['경매 건수', '총반입량(kg)'], aggfunc='sum')
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()
    df_filtered = df_pivot.loc[:, df_pivot.notna().all() & (df_pivot != 0).all()]
    
    return df_filtered

def add_jointmarket_info(df, item):
    # item 단어가 포함된 열을 필터링하여 새로운 DataFrame 생성
    df = df[[col for col in df.columns if item in col]]
    return df

def load_test_jointmarket():
    # 스크립트 파일의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_data = []
    for i in range(52):
        # 테스트 파일 경로 설정
        file_path = os.path.join(script_dir, f'../data/test/meta/TEST_경락정보_산지공판장_{i:02d}.csv')
        one_test_jointmarket = pd.read_csv(file_path)
        filtered_data = jointmarket_filter(one_test_jointmarket)
        all_data.append(filtered_data)
    
    # 모든 테스트 데이터를 하나의 DataFrame으로 병합
    test_jointmarket = pd.concat(all_data, axis=0, ignore_index=True)
    test_jointmarket = test_jointmarket.drop_duplicates().reset_index(drop=True)
    
    return test_jointmarket

def load_train_jointmarket():
    # 스크립트 파일의 디렉토리 경로 가져오기
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # train_jointmarket 파일 경로 설정
    file_path = os.path.join(script_dir, '../data/train/meta/TRAIN_경락정보_산지공판장_2018-2022.csv')
    
    # 파일 읽기
    train_jointmarket = pd.read_csv(file_path)
    return train_jointmarket

# 함수 호출 예시
train_jointmarket = load_train_jointmarket()
test_jointmarket =load_test_jointmarket() 
train_jointmarket =jointmarket_filter(train_jointmarket)


# 전국 도매 데이터 

selected_dome = ['감자_수미_100000', '대파_대파(일반)_100000', '마늘_깐마늘_100000', 
                 '무_기타무_100000', '배_신고_100000', '배추_기타배추_100000', 
                 '상추_포기찹_100000']
dome_items = ['감자 수미', '대파(일반)', '깐마늘(국산)', '무', '배', '배추', '상추']
dome_cols = [
    '감자_수미_100000_경매 건수', '마늘_깐마늘_100000_총반입량(kg)', '대파_대파(일반)_100000_총반입량(kg)', 
    '배추_기타배추_100000_총반입량(kg)', '상추_포기찹_100000_경매 건수', '상추_포기찹_100000_고가(20%) 평균가', 
    '배_신고_100000_고가(20%) 평균가', '무_기타무_100000_평균가(원/kg)'
]


# 전국 도매 정보 불러오기 및 처리 함수
def get_dome_data(df, selected_dome, final_cols):
    # '품목_품종_시장코드' 컬럼 생성 및 필터링
    df['품목_품종_시장코드'] = df['품목명'].replace({'깐마늘(국산)': '마늘', '대파(일반)': '대파', '감자 수미': '감자'}) + '_' + df['품종명'] + '_' + df['시장코드'].astype(str)
    df_filtered = df[df['품목_품종_시장코드'].isin(selected_dome)][['YYYYMMSOON', '품목_품종_시장코드', '총반입량(kg)', '총거래금액(원)', '평균가(원/kg)', '고가(20%) 평균가', '경매 건수']]
    
    # 피벗 테이블 생성
    df_pivot = df_filtered.pivot_table(index='YYYYMMSOON', columns='품목_품종_시장코드', values=['총반입량(kg)', '총거래금액(원)', '평균가(원/kg)', '고가(20%) 평균가', '경매 건수'], aggfunc='sum')
    df_pivot.columns = [f'{col[1]}_{col[0]}' for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # 필요한 컬럼만 유지하여 반환
    return df_pivot[['YYYYMMSOON'] + [col for col in final_cols if col in df_pivot.columns]]

# 스크립트 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))

# 전국 도매 데이터 파일 경로
nation_dome_file_path = os.path.join(script_dir, '../data/train/meta/TRAIN_경락정보_전국도매_2018-2022.csv')
nation_dome_info = pd.read_csv(nation_dome_file_path)[['YYYYMMSOON', '시장코드', '품목명', '품종명', '총반입량(kg)', '총거래금액(원)', '평균가(원/kg)', '고가(20%) 평균가', '경매 건수']]
train_dome = get_dome_data(nation_dome_info, selected_dome, dome_cols)

# 모든 테스트 데이터 로드 함수
def load_all_test_dome(selected_dome, final_cols):
    test_files = glob.glob(os.path.join(script_dir, '../data/test/meta/TEST_경락정보_전국도매_*.csv'))
    all_test_data = pd.concat([get_dome_data(pd.read_csv(file), selected_dome, final_cols) for file in test_files], ignore_index=True)
    return all_test_data.drop_duplicates()

# 모든 테스트 데이터 결합
test_dome = load_all_test_dome(selected_dome, dome_cols)
import pickle

def store_data(test_dome, test_jointmarket, deal_info, folder_name='for_infer'):
    # 스크립트 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 저장 경로 설정
    test_dome_path = os.path.join(save_dir, 'test_dome.pkl')
    test_jointmarket_path = os.path.join(save_dir, 'test_jointmarket.pkl')
    deal_info_path = os.path.join(save_dir, 'deal_info.pkl')

    # 데이터를 각각의 pickle 파일로 저장
    with open(test_dome_path, 'wb') as f:
        pickle.dump(test_dome, f)
    with open(test_jointmarket_path, 'wb') as f:
        pickle.dump(test_jointmarket, f)
    with open(deal_info_path, 'wb') as f:
        pickle.dump(deal_info, f)

    print(f"Data saved to '{save_dir}' with filenames: 'test_dome.pkl', 'test_jointmarket.pkl', 'deal_info.pkl'")


store_data(test_dome, test_jointmarket, deal_info)


##### Functions 

def process_data(raw_file, meta_file, 품목명, scaler=None):
    # 데이터 파일 로드
    raw_data = pd.read_csv(raw_file)
    meta_data = pd.read_csv(meta_file)

    # '품목명' 컬럼이 없을 경우, '품목(품종)명'을 사용하여 생성
    if '품목명' not in raw_data.columns:
        if '품목(품종)명' in raw_data.columns:
            raw_data['품목명'] = raw_data['품목(품종)명']
    
    # 선택한 품목에 대한 raw_data 필터링
    raw_품목 = raw_data[raw_data['품목명'] == 품목명]

    # 품목명에 따른 필터링 조건
    meta_conditions = {
        '감자 수미': lambda x: (x['품목(품종)명'] == '감자 수미') & (x['등급(특 5% 상 35% 중 40% 하 20%)'] == '특'), 
        '무': lambda x: (x['품목(품종)명'] == '무') & (x['거래단위'] == '20키로상자아아'),
        '양파': lambda x: (x['품목(품종)명'] == '양파') & (x['등급(특 5% 상 35% 중 40% 하 20%)'] == '상') & (x['거래단위'] == '12키로'),
        '배추': lambda x: (x['품목(품종)명'] == '알배기배추') & (x['등급(특 5% 상 35% 중 40% 하 20%)'] == '상'),
        '대파(일반)': lambda x: (x['품목(품종)명'] == '대파(일반이이)') | ((x['품목(품종)명'] == '쪽파') & (x['등급(특 5% 상 35% 중 40% 하 20%)'] == '상')),
        '건고추': lambda x: (x['품목명'] == '건고추') & (x['품종명'] == '화건') & (x['등급명'] == '중품'),
        '깐마늘(국산)': lambda x: (x['품목명'] == '깐마늘(국산산)'),
        '상추': lambda x: (x['품목명'] == '상추') & (x['품종명'] == '청') & (x['등급명'] == '중품'),
        '사과': lambda x: (x['품목명'] == '사과아'),
        '배': lambda x: (x['품목명'] == '배') & (x['품종명'] == '신고오')
    }

    # 메타데이터 필터링
    filtered_meta = meta_data[meta_conditions[품목명](meta_data)].copy()

    # '품목명_거래단위_등급' 열 생성
    if 품목명 in ['감자 수미', '무', '양파', '배추', '대파(일반)']:
        filtered_meta['품목명_거래단위_등급'] = filtered_meta['품목(품종)명'] + '_' + filtered_meta['거래단위'] + '_' + filtered_meta['등급(특 5% 상 35% 중 40% 하 20%)']
    else:
        filtered_meta['품목명_거래단위_등급'] = (
            filtered_meta['품목명'] + '_' + filtered_meta['품종명'] + '_' + filtered_meta['등급명'] + '_' + filtered_meta['유통단계별 단위 '].astype(str)
        )

    # 필요한 열만 선택
    columns_to_keep = ['YYYYMMSOON', '품목명_거래단위_등급', '평균가격(원)', '평년 평균가격(원) Common Year SOON']
    filtered_meta = filtered_meta[columns_to_keep]

    # 피벗 테이블 생성
    filtered_meta_pivot = filtered_meta.pivot_table(
        index='YYYYMMSOON',
        columns='품목명_거래단위_등급',
        values=['평균가격(원)', '평년 평균가격(원) Common Year SOON']
    )
    filtered_meta_pivot.columns = ['_'.join(col).strip() for col in filtered_meta_pivot.columns.values]
    filtered_meta_pivot.reset_index(inplace=True)

    # 원본 데이터와 피벗 테이블 병합
    train_data = pd.merge(raw_품목, filtered_meta_pivot, on='YYYYMMSOON', how='left')

    return train_data



fin_cols1 = {
    '감자 수미': ['평균가격(원)_감자 수미_20키로상자_특', '감자_수미_100000_경매 건수'],
    '건고추': ['평균가격(원)_건고추_화건_중품_30'],
    '깐마늘(국산)': ['마늘_깐마늘_100000_총반입량(kg)'],
    '대파(일반)': ['평균가격(원)_쪽파_10키로상자_상', '1000000000_대파_대파(일반)_11_총반입량(kg)'],
    '무': ['무_기타무_100000_평균가(원/kg)', '1000000000_무_기타무_11_총반입량(kg)'],
    '배추': ['평년 평균가격(원) Common Year SOON', '평균가격(원)_알배기배추_8키로상자_상'],
    '사과': ['평년 평균가격(원) Common Year SOON', '사과_금액', '사과_평년 반입량 증감률(%)'],
    '상추': ['상추_포기찹_100000_경매 건수', '상추_포기찹_100000_고가(20%) 평균가'],
    '양파': ['평균가격(원)_양파_12키로_상'],
    '배': ['배_신고_100000_고가(20%) 평균가', '배_반입량']
}


fin_cols2 = {
    '감자 수미': ['평균가격-평년가격', '평균가격(원)_감자 수미_20키로상자_특', '감자_수미_100000_경매 건수'],
    '건고추': ['평균가격-평년가격', '평균가격(원)_건고추_화건_중품_30'],
    '깐마늘(국산)': ['마늘_깐마늘_100000_총반입량(kg)'],
    '대파(일반)': ['평균가격-평년가격', '평균가격(원)_쪽파_10키로상자_상', '대파_대파(일반)_100000_총반입량(kg)'],
    '배': ['평균가격-평년가격', '배_신고_100000_고가(20%) 평균가'],
    '배추': ['평균가격-평년가격', '평년 평균가격(원) Common Year SOON', '평균가격(원)_알배기배추_8키로상자_상', '배추_기타배추_100000_총반입량(kg)'],
    '사과': ['평균가격-평년가격', '평년 평균가격(원) Common Year SOON', '사과_금액', '사과_평년 반입량 증감률(%)'],
    '상추': ['평균가격-평년가격', '상추_포기찹_100000_경매 건수'],
    '양파': ['평균가격-평년가격', '평균가격(원)_양파_12키로_상'],
    '무': ['무_기타무_100000_평균가(원/kg)', '1000000000_무_기타무_11_경매 건수', '1000000000_무_기타무_11_총반입량(kg)']
}










