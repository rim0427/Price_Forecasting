import os
import torch
import glob
import random 
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
 
 
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
class momentum(nn.Module):
    def __init__(self, window_size):
        super(momentum, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        # x: [Batch, Seq_len, Channels]
        momentum = x[:, self.window_size:, :] - x[:, :-self.window_size, :]
        padding = torch.zeros(x.size(0), self.window_size, x.size(2)).to(x.device)
        momentum = torch.cat([padding, momentum], dim=1)
        return momentum

class series_decomp2(nn.Module):
    def __init__(self, kernel_size, momentum_window):
        super(series_decomp2, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.momentum = momentum(momentum_window)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        momentums =self.momentum(x) 
        return res, moving_mean , momentums
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

#DlinwithAttn 
class ModelWithMultiheadAttention(nn.Module):
    def __init__(self, configs):
        super(ModelWithMultiheadAttention, self).__init__()
        self.seq_len = configs.window_size
        self.pred_len = configs.forecast_size
        self.n_heads = configs.n_heads
        self.channels = configs.feature_size
        self.kernel_size =configs.kernel_size
        self.momentum_window =configs.momentum_window
        self.individual = configs.individual
        
        
        self.decomposition = series_decomp2(self.kernel_size , self.momentum_window)
        # Multihead Attention 레이어
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=self.n_heads, batch_first=True)

        if configs.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        seasonal_init, trend_init, momentum_init = self.decomposition(x)
        # 계절성, 트렌드성 추출하기 
        seasonal_init, trend_init,momentum_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1) ,momentum_init.permute(0,2,1) 
        combined_features = trend_init + seasonal_init + momentum_init
        attn_output, _ = self.multihead_attn(query=trend_init, key=momentum_init, value=seasonal_init)
        

        if self.individual:
            seasonal_output = torch.zeros([attn_output.size(0), self.channels, self.pred_len], dtype=attn_output.dtype).to(attn_output.device)
            trend_output = torch.zeros([attn_output.size(0), self.channels, self.pred_len], dtype=attn_output.dtype).to(attn_output.device)
            
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](attn_output[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(attn_output)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]로 변환

# Linear
class LinModel(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(LinModel, self).__init__()
        self.seq_len = configs.window_size
        self.pred_len = configs.forecast_size
        self.channels = configs.feature_size
        self.individual = configs.individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]

# Dlinear
class LTSF_DLinear(torch.nn.Module):
    def __init__(self,config):
        super(LTSF_DLinear, self).__init__()
        self.window_size = config.window_size
        self.forecast_size = config.forecast_size
        self.decomposition = series_decomp(config.kernel_size)
        self.individual = config.individual
        self.channels = config.feature_size
        if self.individual:
            self.Linear_Seasonal = torch.nn.ModuleList()
            self.Linear_Trend = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Trend.weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))

    def forward(self, x):
        trend_init, seasonal_init = self.decomposition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forecast_size], dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forecast_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

       
 

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


## config 
class potato_config1:
    def __init__(self):
        self.seed = 258
        self.learning_rate = 0.001
        self.epoch = 71
        self.batch_size = 16
        self.optimizer = 'adam'
        self.weight_decay = 1e-8
        self.scheduler = 'reduce_on_plateau'
        self.patience = 6
        self.step_size = 20      
        self.gamma = 0.5   
        
        self.model = 'D'
        self.window_size = 3
        self.fin_cols = ['평균가격(원)_감자 수미_20키로상자_특', '감자_수미_100000_경매 건수']
        self.forecast_size = 2
        self.kernel_size = 17
        self.individual = False
        self.feature_size = 3
        self.year = 2018
        
class garlic_config1:
    def __init__(self):
        self.seed = 97
        self.learning_rate = 0.0015
        self.epoch = 109
        self.batch_size = 8
        self.optimizer = 'rmsprop'
        self.weight_decay = 0
        self.scheduler = 'reduce_on_plateau'
        self.patience = 4
        self.step_size = 20      
        self.gamma = 0.5   
        
        self.model = 'D'
        self.window_size = 3
        self.fin_cols = ['마늘_깐마늘_100000_총반입량(kg)']
        self.forecast_size = 2
        self.kernel_size = 19
        self.individual = True
        self.feature_size = 2 
        self.year = 2019
        
class apple_config1:
    def __init__(self):
        self.seed = 319
        self.learning_rate = 0.0025
        self.epoch = 125
        self.batch_size = 16
        self.optimizer = 'adam'
        self.weight_decay = 0
        self.scheduler = 'none'
        self.patience = 3
        self.step_size = 20      
        self.gamma = 0.5   
        self.model = 'D'
        self.window_size = 3
        self.fin_cols = ['평년 평균가격(원) Common Year SOON', '사과_금액', '사과_평년 반입량 증감률(%)']
        self.forecast_size = 2
        self.kernel_size = 13
        self.individual = True
        self.feature_size = 4
        self.year = 2018

class lettuce_config1:
    def __init__(self):
        self.seed = 435
        self.learning_rate = 0.002
        self.epoch = 99
        self.batch_size = 16
        self.optimizer = 'rmsprop'
        self.weight_decay = 0
        self.scheduler = 'none'
        self.patience = 6
        self.step_size = 20      
        self.gamma = 0.5   
        self.model = 'D'
        self.window_size = 3
        self.fin_cols = ['상추_포기찹_100000_경매 건수', '상추_포기찹_100000_고가(20%) 평균가']
        self.forecast_size = 2
        self.kernel_size = 19
        self.individual = True
        self.feature_size = 3
        self.year = 2021

class pepper_config1:
    def __init__(self):
        self.seed = 81
        self.learning_rate = 0.002
        self.epoch = 115
        self.batch_size = 32
        self.optimizer = 'rmsprop'
        self.weight_decay = 1e-09
        self.scheduler = 'reduce_on_plateau'
        self.patience = 5
        self.step_size = 20      
        self.gamma = 0.5   
        
        self.fin_cols = ['평균가격(원)_건고추_화건_중품_30']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 17
        self.individual = True
        self.feature_size = 2
        self.momentum_window = 2
        self.n_heads = 3
        self.year = 2020

class daepa_config1:
    def __init__(self):
        self.seed = 800
        self.learning_rate = 0.003
        self.epoch = 140
        self.batch_size = 8
        self.optimizer = 'adam'
        self.weight_decay = 1e-10
        self.scheduler = 'reduce_on_plateau'
        self.patience = 3
        self.step_size = 20      
        self.gamma = 0.5   
        self.fin_cols = ['평균가격(원)_쪽파_10키로상자_상', '1000000000_대파_대파(일반)_11_총반입량(kg)']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 15
        self.individual = True
        self.feature_size = 3
        self.momentum_window = 1
        self.n_heads = 3
        self.year = 2020
        
class moo_config1:
    def __init__(self):
        self.seed = 2551
        self.learning_rate = 0.003
        self.epoch = 81
        self.batch_size = 8
        self.optimizer = 'adamw'
        self.weight_decay = 0
        self.scheduler = 'none'
        self.patience = 5
        self.step_size = 20      
        self.gamma = 0.5   
        self.fin_cols = ['무_기타무_100000_평균가(원/kg)', '1000000000_무_기타무_11_총반입량(kg)']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 15
        self.individual = True
        self.feature_size = 3
        self.momentum_window = 2
        self.n_heads = 1
        self.year = 2018
        

class cabbage_config1:
    def __init__(self):
        self.seed = 318
        self.learning_rate = 0.0009
        self.epoch = 72
        self.batch_size = 16
        self.optimizer = 'rmsprop'
        self.weight_decay = 1e-8
        self.scheduler = 'none'
        self.patience = 3
        self.step_size = 20      
        self.gamma = 0.5   
        self.fin_cols = ['평년 평균가격(원) Common Year SOON', '평균가격(원)_알배기배추_8키로상자_상']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 21
        self.individual = True
        self.feature_size = 3
        self.momentum_window = 3
        self.n_heads = 1
        self.year = 2019
        
class onion_config1:
    def __init__(self):
        self.seed = 321
        self.learning_rate = 0.0025
        self.epoch = 145
        self.batch_size = 32
        self.optimizer = 'adam'
        self.weight_decay = 0
        self.scheduler = 'none'
        self.patience = 3
        self.step_size = 20      
        self.gamma = 0.5   
        self.fin_cols = ['평균가격(원)_양파_12키로_상']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 21
        self.individual = False
        self.feature_size = 2
        self.momentum_window = 1
        self.n_heads = 3
        self.year = 2020
        
class pear_config1:
    def __init__(self):
        self.seed = 2713
        self.learning_rate = 0.003
        self.epoch = 156
        self.batch_size = 16
        self.optimizer = 'adamw'
        self.weight_decay = 0
        self.scheduler = 'none'
        self.patience = 4
        self.step_size = 20      
        self.gamma = 0.5   
        self.fin_cols = ['배_신고_100000_고가(20%) 평균가', '배_반입량']
        self.window_size = 3
        self.forecast_size = 2
        self.kernel_size = 19
        self.individual = False
        self.feature_size = 3
        self.momentum_window = 1
        self.n_heads = 3
        self.year = 2019
        
class potato_config2:
    def __init__(self):
        self.seed = 199
        self.learning_rate = 0.003
        self.epoch = 90
        self.patience = 4
        self.batch_size = 16
        self.optimizer = 'rmsprop'
        self.weight_decay = 0
        self.scheduler = 'none'
        
        self.model = 'L'
        self.window_size = 9
        self.fin_cols = ['평균가격-평년가격', '평균가격(원)_감자 수미_20키로상자_특', '감자_수미_100000_경매 건수']
        self.forecast_size = 3
        self.kernel_size = 15
        self.individual = False
        self.feature_size = len(self.fin_cols) + 1  # 주요 특징의 개수에 따라 조정
        self.year = 2018
        
class garlic_config2:
    def __init__(self):
        self.seed = 25
        self.learning_rate = 0.002
        self.epoch = 113
        self.patience = 5
        self.batch_size = 8
        self.optimizer = 'adam'
        self.weight_decay = 1e-09
        self.scheduler = 'none'
        
        self.model = 'D'
        self.window_size = 9
        self.fin_cols = ['마늘_깐마늘_100000_총반입량(kg)']
        self.forecast_size = 3
        self.kernel_size = 21
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.year = 2019

class apple_config2:
    def __init__(self):
        self.seed = 332
        self.learning_rate = 0.002
        self.epoch = 120
        self.patience = 6
        self.batch_size = 8
        self.optimizer = 'adam'
        self.weight_decay = 1e-08
        self.scheduler = 'none'
        
        self.model = 'D'
        self.window_size = 9
        self.fin_cols = ['평균가격-평년가격', '평년 평균가격(원) Common Year SOON', '사과_금액', '사과_평년 반입량 증감률(%)']
        self.forecast_size =3
        self.kernel_size = 13
        self.individual = False
        self.feature_size = len(self.fin_cols) + 1
        self.year = 2018

class lettuce_config2:
    def __init__(self):
        self.seed = 888
        self.learning_rate = 0.002
        self.epoch = 100
        self.patience = 6
        self.batch_size = 8
        self.optimizer = 'rmsprop'
        self.weight_decay = 1e-08
        self.scheduler = 'none'
        
        self.model = 'D'
        self.window_size = 9
        self.fin_cols = ['평균가격-평년가격', '상추_포기찹_100000_경매 건수']
        self.forecast_size = 3
        self.kernel_size = 21
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.year = 2021

class pepper_config2:
    def __init__(self):
        self.seed = 221
        self.learning_rate = 0.0015
        self.epoch = 79
        self.patience = 5
        self.batch_size = 32
        self.optimizer = 'adam'
        self.weight_decay = 1e-08
        self.scheduler = 'none'
        
        self.fin_cols = ['평균가격-평년가격', '평균가격(원)_건고추_화건_중품_30']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 15
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 3
        self.n_heads = 1
        self.year = 2020
        
class daepa_config2:
    def __init__(self):
        self.seed = 6
        self.learning_rate = 0.001
        self.epoch = 112
        self.patience = 5
        self.batch_size = 8
        self.optimizer = 'rmsprop'
        self.weight_decay = 0
        self.scheduler = 'none'
        
        self.fin_cols = ['평균가격-평년가격', '평균가격(원)_쪽파_10키로상자_상', '대파_대파(일반)_100000_총반입량(kg)']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 15
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 1
        self.n_heads = 1
        self.year = 2020

class moo_config2:
    def __init__(self):
        self.seed = 101
        self.learning_rate = 0.0025
        self.epoch = 97
        self.batch_size = 16
        self.optimizer = 'adamw'
        self.weight_decay = 1e-10
        self.scheduler = 'none'
        self.patience = 4 
        
        self.fin_cols = ['무_기타무_100000_평균가(원/kg)', '1000000000_무_기타무_11_경매 건수', '1000000000_무_기타무_11_총반입량(kg)']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 19
        self.individual = False
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 3
        self.n_heads = 3
        self.year = 2018


class cabbage_config2:
    def __init__(self):
        self.seed = 268
        self.learning_rate = 0.0015
        self.epoch = 73
        self.batch_size = 16
        self.optimizer = 'adam'
        self.weight_decay = 0
        self.scheduler = 'none'
        
        self.fin_cols = ['평균가격-평년가격', '평년 평균가격(원) Common Year SOON', '평균가격(원)_알배기배추_8키로상자_상', '배추_기타배추_100000_총반입량(kg)']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 21
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 3
        self.n_heads = 3
        self.year = 2019
        
        
class onion_config2:
    def __init__(self):
        self.seed = 3
        self.learning_rate = 0.0025
        self.epoch = 157
        self.batch_size = 32
        self.optimizer = 'adamw'
        self.weight_decay = 1e-09
        self.scheduler = 'none'
        
        self.fin_cols = ['평균가격-평년가격', '평균가격(원)_양파_12키로_상']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 17
        self.individual = False
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 2
        self.n_heads = 3
        self.year = 2020


class pear_config2:
    def __init__(self):
        self.seed = 337
        self.learning_rate = 0.0015
        self.epoch = 134
        self.batch_size = 8
        self.patience = 7 
        self.optimizer = 'adam'
        self.weight_decay = 0
        self.step_size = 20      
        self.gamma = 0.5 
        self.scheduler = 'reduce_on_plateau'
        
        self.fin_cols = ['평균가격-평년가격', '배_신고_100000_고가(20%) 평균가']
        self.window_size = 9
        self.forecast_size = 3
        self.kernel_size = 21
        self.individual = True
        self.feature_size = len(self.fin_cols) + 1
        self.momentum_window = 3
        self.n_heads = 3
        self.year = 2019
        
        

############################################################################################3
#scaler, 추론에 필요한 거 불러오기 
print('start')



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
       
       
       

# 모든 테스트 데이터 로드 함수
def load_all_test_dome(selected_dome, final_cols):
    test_files = glob.glob(os.path.join(script_dir, '../data/test/meta/TEST_경락정보_전국도매_*.csv'))
    all_test_data = pd.concat([get_dome_data(pd.read_csv(file), selected_dome, final_cols) for file in test_files], ignore_index=True)
    return all_test_data.drop_duplicates()


# 모든 테스트 데이터 결합
test_dome = load_all_test_dome(selected_dome, dome_cols)
test_jointmarket =load_test_jointmarket() 
       
deal_info = get_deal_info()




def load_infer_data(folder_name='for_infer'):
    # 스크립트 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.join(script_dir, folder_name)

    # 스케일러 파일 경로
    scalers_path = os.path.join(load_dir, 'scalers.pkl')
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    return scalers

scalers= load_infer_data()






def infer_dlin2(품목리스트, config, scaler):
    # 감자는 Linear로 하기
    seed_everything(config.seed)
    predicts = {}

    for item in 품목리스트:
        # 모델 불러오기
        model = LinModel(config) if item == '감자 수미' else LTSF_DLinear(config)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 상위 폴더로 이동 후 'big/dl_weights2' 경로 지정
        model_path = os.path.join(script_dir, '..', 'dl_weights2', f"{item}_model2_win9.pth")

        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        item_test_tensors = []

        for i in range(52):
            if item in group1:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_1.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_경락정보_가락도매_{i:02d}.csv")
            elif item in group2:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_중도매_{i:02d}.csv")
            elif item in group3:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_소매_{i:02d}.csv")
            
            test_data = process_data(test_file, meta_file, item)
            if item != '깐마늘(국산)':
                interpolate_zeros(test_data, '평년 평균가격(원) Common Year SOON')
                test_data['평균가격-평년가격'] = test_data['평균가격(원)'] - test_data['평년 평균가격(원) Common Year SOON']
            
            fincols = ['YYYYMMSOON', '평균가격(원)'] + [col for col in config.fin_cols if col in test_data.columns]
            test_data = test_data[fincols]
            
            dome_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_dome.columns]
            test_dome_filtered = test_dome[dome_cols]
            test_data = pd.merge(test_data, test_dome_filtered, how='left', on='YYYYMMSOON')

            if item == '사과':
                deal_cols = ['거래일자'] + [col for col in config.fin_cols if col in deal_info.columns]
                deal_info_filtered = deal_info[deal_cols]
                test_data = pd.merge(test_data, deal_info_filtered, how='left', left_on='YYYYMMSOON', right_on='거래일자')
                interpolate_zeros(test_data, '사과_평년 반입량 증감률(%)')

            final_columns = ['평균가격(원)'] + config.fin_cols
            test_data = test_data[final_columns]
            test_price_df = test_data.reset_index(drop=True)
            test_price_df = test_price_df.iloc[-1 * config.window_size:, :]
            normalized_testdata = scaler.transform(test_price_df)
            test_tensor = torch.tensor(normalized_testdata, dtype=torch.float32)
            item_test_tensors.append(test_tensor)

        item_test_batch = torch.stack(item_test_tensors).to(device)
        with torch.no_grad():
            prediction = model(item_test_batch)
        
        prediction = prediction.cpu().numpy()
        product_predict = []
        for pred in prediction:
            inverse_pred = inverse_normalize(pred, scaler)
            product_predict.append(inverse_pred[:, 0])
        
        flatlist = np.concatenate(product_predict).tolist()
        predicts[item] = flatlist
    return predicts


def infer_dlinAttn2(품목리스트, config, scaler):
    seed_everything(config.seed)
    predicts = {}

    for item in 품목리스트:
        model = ModelWithMultiheadAttention(config)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', 'dl_weights2', f"{item}_model2_win9.pth")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        item_test_tensors = []

        for i in range(52):
            if item in group1:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_1.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_경락정보_가락도매_{i:02d}.csv")
            elif item in group2:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_중도매_{i:02d}.csv")
            elif item in group3:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_소매_{i:02d}.csv")
            
            test_data = process_data(test_file, meta_file, item)
            if item != '무':
                interpolate_zeros(test_data, '평년 평균가격(원) Common Year SOON')
                test_data['평균가격-평년가격'] = test_data['평균가격(원)'] - test_data['평년 평균가격(원) Common Year SOON']
            
            fincols = ['YYYYMMSOON', '평균가격(원)'] + [col for col in config.fin_cols if col in test_data.columns]
            test_data = test_data[fincols]
            
            dome_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_dome.columns]
            test_dome_filtered = test_dome[dome_cols]
            test_data = pd.merge(test_data, test_dome_filtered, how='left', on='YYYYMMSOON')
            
            if item in ['대파(일반)', '무']:
                joint_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_jointmarket.columns]
                test_jointmarket_filtered = test_jointmarket[joint_cols]
                test_data = pd.merge(test_data, test_jointmarket_filtered, how='left', on='YYYYMMSOON')
            elif item == '배':
                deal_cols = ['거래일자'] + [col for col in config.fin_cols if col in deal_info.columns]
                deal_info_filtered = deal_info[deal_cols]
                test_data = pd.merge(test_data, deal_info_filtered, how='left', left_on='YYYYMMSOON', right_on='거래일자')

            final_columns = ['평균가격(원)'] + config.fin_cols
            test_data = test_data[final_columns]
            test_price_df = test_data.reset_index(drop=True)
            test_price_df = test_price_df.iloc[-1 * config.window_size:, :]
            normalized_testdata = scaler.transform(test_price_df)
            test_tensor = torch.tensor(normalized_testdata, dtype=torch.float32)
            item_test_tensors.append(test_tensor)

        item_test_batch = torch.stack(item_test_tensors).to(device)
        with torch.no_grad():
            prediction = model(item_test_batch)
        
        prediction = prediction.cpu().numpy()
        product_predict = []
        for pred in prediction:
            inverse_pred = inverse_normalize(pred, scaler)
            product_predict.append(inverse_pred[:, 0])
        
        flatlist = np.concatenate(product_predict).tolist()
        predicts[item] = flatlist
    return predicts


def infer_dlinAttn1(품목리스트, config, scaler):
    # windowsize = 3, forecasting size = 2 
    seed_everything(config.seed)
    predicts = {}
    
    for item in 품목리스트:
        print(f"Processing {item}")
        
        # 모델 불러오기
        model = ModelWithMultiheadAttention(config)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..', 'dl_weights2', f"{item}_model1_win3.pth")
        
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        item_test_tensors = []
        
        for i in range(52):
            if item in group1:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_1.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_경락정보_가락도매_{i:02d}.csv")
            elif item in group2:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_중도매_{i:02d}.csv")
            elif item in group3:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_소매_{i:02d}.csv")
            
            test_data = process_data(test_file, meta_file, item)
            fincols = ['YYYYMMSOON', '평균가격(원)'] + [col for col in config.fin_cols if col in test_data.columns]
            test_data = test_data[fincols]
            dome_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_dome.columns]
            test_dome_filtered = test_dome[dome_cols]
            test_data = pd.merge(test_data, test_dome_filtered, how='left', on='YYYYMMSOON')
            
            if item in ['대파(일반)', '무']:
                joint_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_jointmarket.columns]
                test_jointmarket_filtered = test_jointmarket[joint_cols]
                test_data = pd.merge(test_data, test_jointmarket_filtered, how='left', on='YYYYMMSOON')
            elif item == '배':
                deal_cols = ['거래일자'] + [col for col in config.fin_cols if col in deal_info.columns]
                deal_info_filtered = deal_info[deal_cols]
                test_data = pd.merge(test_data, deal_info_filtered, how='left', left_on='YYYYMMSOON', right_on='거래일자')
            
            final_columns = ['평균가격(원)'] + config.fin_cols
            test_data = test_data[final_columns]
            
            test_price_df = test_data.reset_index(drop=True)
            test_price_df = test_price_df.iloc[-1 * config.window_size:, :]
            normalized_testdata = scaler.transform(test_price_df)
            test_tensor = torch.tensor(normalized_testdata, dtype=torch.float32)
            item_test_tensors.append(test_tensor)
        
        item_test_batch = torch.stack(item_test_tensors).to(device)
        with torch.no_grad():
            prediction = model(item_test_batch)
            
        prediction = prediction.cpu().numpy()
        product_predict = []
        for pred in prediction:
            inverse_pred = inverse_normalize(pred, scaler)
            extended_pred = np.append(inverse_pred[:, 0], 0)
            product_predict.append(extended_pred)
        flatlist = np.concatenate(product_predict).tolist()
        predicts[item] = flatlist
    return predicts 


def infer_dlin1(품목리스트, config, scaler):
    seed_everything(config.seed)
    predicts = {}
    
    for item in 품목리스트:
        print(f"Processing {item}")
        model = LTSF_DLinear(config)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '..',  'dl_weights2', f"{item}_model1_win3.pth")
        
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        
        item_test_tensors = []
        
        for i in range(52):
            if item in group1:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_1.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_경락정보_가락도매_{i:02d}.csv")
            elif item in group2:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_중도매_{i:02d}.csv")
            elif item in group3:
                test_file = os.path.join(script_dir, f"../data/test/TEST_{i:02d}_2.csv")
                meta_file = os.path.join(script_dir, f"../data/test/meta/TEST_소매_{i:02d}.csv")
            else:
                continue

            test_data = process_data(test_file, meta_file, item)
            fincols = ['YYYYMMSOON', '평균가격(원)'] + [col for col in config.fin_cols if col in test_data.columns]
            test_data = test_data[fincols]
            dome_cols = ['YYYYMMSOON'] + [col for col in config.fin_cols if col in test_dome.columns]
            test_dome_filtered = test_dome[dome_cols]
            test_data = pd.merge(test_data, test_dome_filtered, how='left', on='YYYYMMSOON')
            if item == '사과':
                deal_cols = ['거래일자'] + [col for col in config.fin_cols if col in deal_info.columns]
                deal_info_filtered = deal_info[deal_cols]
                test_data = pd.merge(test_data, deal_info_filtered, how='left', left_on='YYYYMMSOON', right_on='거래일자')
                interpolate_zeros(test_data, '사과_평년 반입량 증감률(%)')
                
            final_columns = ['평균가격(원)'] + config.fin_cols
            test_data = test_data[final_columns]
            
            test_price_df = test_data.reset_index(drop=True)
            test_price_df = test_price_df.iloc[-1 * config.window_size:, :]
            normalized_testdata = scaler.transform(test_price_df)
            test_tensor = torch.tensor(normalized_testdata, dtype=torch.float32)
            item_test_tensors.append(test_tensor)
        
        item_test_batch = torch.stack(item_test_tensors).to(device)
        
        with torch.no_grad():
            prediction = model(item_test_batch)
        
        prediction = prediction.cpu().numpy()
        
        product_predict = []
        for pred in prediction:
            inverse_pred = inverse_normalize(pred, scaler)
            extended_pred = np.append(inverse_pred[:, 0], 0)
            product_predict.append(extended_pred)
        
        flatlist = np.concatenate(product_predict).tolist()
        predicts[item] = flatlist
    
    return predicts



# 각 품목별 예측
potato_preds1 = infer_dlin1(['감자 수미'], potato_config1(), scalers['potato_scaler1'])
garlic_preds1 = infer_dlin1(['깐마늘(국산)'], garlic_config1(), scalers['garlic_scaler1'])
apple_preds1 = infer_dlin1(['사과'], apple_config1(), scalers['apple_scaler1'])
lettuce_preds1 = infer_dlin1(['상추'], lettuce_config1(), scalers['lettuce_scaler1'])
pepper_preds1 = infer_dlinAttn1(['건고추'], pepper_config1(), scalers['pepper_scaler1'])
daepa_preds1 = infer_dlinAttn1(['대파(일반)'], daepa_config1(), scalers['daepa_scaler1'])
moo_preds1 = infer_dlinAttn1(['무'], moo_config1(), scalers['moo_scaler1'])
cabbage_preds1 = infer_dlinAttn1(['배추'], cabbage_config1(), scalers['cabbage_scaler1'])
onion_preds1 = infer_dlinAttn1(['양파'], onion_config1(), scalers['onion_scaler1'])
pear_preds1 = infer_dlinAttn1(['배'], pear_config1(), scalers['pear_scaler1'])

# 장기 모델 추론 (감자, 배 제외)
garlic_preds2 = infer_dlin2(['깐마늘(국산)'], garlic_config2(), scalers['garlic_scaler2'])
apple_preds2 = infer_dlin2(['사과'], apple_config2(), scalers['apple_scaler2'])
lettuce_preds2 = infer_dlin2(['상추'], lettuce_config2(), scalers['lettuce_scaler2'])
pepper_preds2 = infer_dlinAttn2(['건고추'], pepper_config2(), scalers['pepper_scaler2'])
daepa_preds2 = infer_dlinAttn2(['대파(일반)'], daepa_config2(), scalers['daepa_scaler2'])
moo_preds2 = infer_dlinAttn2(['무'], moo_config2(), scalers['moo_scaler2'])
cabbage_preds2 = infer_dlinAttn2(['배추'], cabbage_config2(), scalers['cabbage_scaler2'])
onion_preds2 = infer_dlinAttn2(['양파'], onion_config2(), scalers['onion_scaler2'])
pear_preds2 = infer_dlinAttn2(['배'], pear_config2(), scalers['pear_scaler2'])

# 예측값 딕셔너리 생성
preds_dict = {
    'potato': (potato_preds1, None),  # 장기 예측 없음
    'garlic': (garlic_preds1, garlic_preds2),
    'apple': (apple_preds1, apple_preds2),
    'lettuce': (lettuce_preds1, lettuce_preds2),
    'pepper': (pepper_preds1, pepper_preds2),
    'daepa': (daepa_preds1, daepa_preds2),
    'moo': (moo_preds1, moo_preds2),
    'cabbage': (cabbage_preds1, cabbage_preds2),
    'onion': (onion_preds1, onion_preds2),
    'pear': (pear_preds1, pear_preds2),  # 장기 예측 없음
}

# 제출 파일 생성 함수
def generate_submission_files(preds_dict, output_dir='../data', template_file='../data/sample_submission.csv'):
    # 스크립트 파일의 디렉토리 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 샘플 제출 파일 불러오기
    dl_model1 = pd.read_csv(os.path.join(script_dir, template_file))
    dl_model2 = pd.read_csv(os.path.join(script_dir, template_file))
    
    # 예측 결과를 모델1과 모델2에 업데이트
    for item, (preds1, preds2) in preds_dict.items():
        # preds1의 예측값을 dl_model1에 추가
        if preds1 is not None:
            for sub_item, prices in preds1.items():
                if sub_item in dl_model1.columns:
                    dl_model1[sub_item] = prices
        
        # preds2의 예측값이 있을 때만 dl_model2에 추가
        if preds2 is not None:
            for sub_item, prices in preds2.items():
                if sub_item in dl_model2.columns:
                    dl_model2[sub_item] = prices

    # 저장 경로 설정
    os.makedirs(os.path.join(script_dir, output_dir), exist_ok=True)
    dl_model1_save_path = os.path.join(script_dir, output_dir, 'dl_model1_submission.csv')
    dl_model2_save_path = os.path.join(script_dir, output_dir, 'dl_model2_submission.csv')

    # 결과 파일 저장
    dl_model1.to_csv(dl_model1_save_path, index=False)
    dl_model2.to_csv(dl_model2_save_path, index=False)

    print(f"dl_model1 saved at {dl_model1_save_path}")
    print(f"dl_model2 saved at {dl_model2_save_path}")

# 제출 파일 생성 함수 호출
generate_submission_files(preds_dict)
