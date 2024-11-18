import random
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import itertools
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import butter, filtfilt, lfilter


def common_FE(train, test, c):



    # 선형보간
    def fill_nan_with_avg_all_columns(df):
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        return df
    train = fill_nan_with_avg_all_columns(train)




    # 3개월 전과 현재 시점의 차이 피처
    train['E_M_T'] = train['평균가격(원)'] - train['평균가격(원)_T-8']
    test['E_M_T'] = test['평균가격(원)'] - test['평균가격(원)_T-8']


    train['평균_평년_diff'] = train['평균가격(원)'] - train['평년 평균가격(원) Common Year SOON']
    test['평균_평년_diff'] = test['평균가격(원)'] - test['평년 평균가격(원) Common Year SOON']


    # 3개월간 가격 차이 누적 변동량
    def cumulative_increase(row):
        cumulative_change = 0
        cumulative_change += row["평균가격(원)"] - row["평균가격(원)_T-1"]
        for i in range(1, 8):
            cumulative_change += row[f"평균가격(원)_T-{i}"] - row[f"평균가격(원)_T-{i+1}"]
        return cumulative_change
    train["누적 증감"] = train.apply(cumulative_increase, axis=1)
    test["누적 증감"] = test.apply(cumulative_increase, axis=1)


    # 각각의 시점에서 과거와의 가격 차이 계산 / 변동
    def create_difference_features(df):
        for i in range(8, 0, -1):
            col_current = f'평균가격(원)_T-{i-1}' if i-1 > 0 else '평균가격(원)'
            col_previous = f'평균가격(원)_T-{i}'
            df[f'{col_previous} - {col_current}'] = df[col_previous] - df[col_current]
    create_difference_features(train)
    create_difference_features(test)
    



    # 그룹별 피처 드랍
    group1 = ['배추', '무', '양파', '감자 수미', '대파(일반)']
    group2 = ['건고추', '깐마늘(국산)','상추', '사과', '배']
    if c in group1:
        train.drop(['가락시장 품목코드(5자리)', '품목(품종)명', '거래단위', '등급(특 5% 상 35% 중 40% 하 20%)'],axis=1,inplace=True)
        test.drop(['가락시장 품목코드(5자리)', '품목(품종)명', '거래단위', '등급(특 5% 상 35% 중 40% 하 20%)'],axis=1,inplace=True)
    elif c in group2: 
        train.drop(['품목코드 ', '품종코드 ', '품종명', '품목명','등급명','유통단계별 무게 ','유통단계별 단위 '],axis=1,inplace=True)
        test.drop(['품목코드 ', '품종코드 ', '품종명', '품목명','등급명','유통단계별 무게 ','유통단계별 단위 '],axis=1,inplace=True)





    # YYYYMMSOON sin, cos 변환
    순_mapping = {'상순': 1, '중순': 2, '하순': 3}
    def process_YYYYMMSOON(value):
        year = int(value[:4]) 
        month = int(value[4:6])
        순 = 순_mapping.get(value[6:], None) 
        return year, month, 순

    for col in train.columns:
        if 'YYYYMMSOON' in col:
            train[f'{col}_년도'], train[f'{col}_달'], train[f'{col}_순'] = zip(*train[col].map(process_YYYYMMSOON))
            test[f'{col}_년도'], test[f'{col}_달'], test[f'{col}_순'] = zip(*test[col].map(process_YYYYMMSOON))
            
    train['연도'] = train['YYYYMMSOON_년도']
    test['연도'] = test['YYYYMMSOON_년도']
    '''
    train['달_순'] = train['YYYYMMSOON_달'].astype(str) + "_" + train['YYYYMMSOON_순'].astype(str)
    test['달_순'] = test['YYYYMMSOON_달'].astype(str) + "_" + test['YYYYMMSOON_순'].astype(str)


    mean_encoding_1 = train.groupby('달_순')['1순'].mean()
    mean_encoding_2 = train.groupby('달_순')['2순'].mean()
    mean_encoding_3 = train.groupby('달_순')['3순'].mean()
    train['1_encoding'] = train['달_순'].map(mean_encoding_1)
    test['1_encoding'] = test['달_순'].map(mean_encoding_1)

    train['2_encoding'] = train['달_순'].map(mean_encoding_2)
    test['2_encoding'] = test['달_순'].map(mean_encoding_2)

    train['3_encoding'] = train['달_순'].map(mean_encoding_3)
    test['3_encoding'] = test['달_순'].map(mean_encoding_3)

    std_encoding_1 = train.groupby('달_순')['1순'].std()
    std_encoding_2 = train.groupby('달_순')['2순'].std()
    std_encoding_3 = train.groupby('달_순')['3순'].std()


    train['1std_encoding'] = train['달_순'].map(std_encoding_1)
    test['1std_encoding'] = test['달_순'].map(std_encoding_1)

    train['2std_encoding'] = train['달_순'].map(std_encoding_2)
    test['2std_encoding'] = test['달_순'].map(std_encoding_2)

    train['3std_encoding'] = train['달_순'].map(std_encoding_3)
    test['3std_encoding'] = test['달_순'].map(std_encoding_3)


    train.drop(['달_순'],axis=1,inplace=True)
    test.drop(['달_순'],axis=1,inplace=True)
    '''
    def determine_harvest_weight(item, month):
        # 품목별 수확 가능한 월 정의
        harvest_months = {
            '감자 수미': [5, 6, 8, 9, 10],
            '양파': [4, 5, 6],
            '배추': [1, 2, 3, 4, 5, 11, 12],
            '건고추': [4, 5, 6],
            '사과': [9, 10, 11],
            '배': [8, 9]
        }

        # 품목별 수확 가능한 월에 따라 1 또는 0 반환
        if month in harvest_months.get(item, []):
            return 1
        return 0

    # 예시 사용법
    train['harvest_weight'] = train.apply(lambda x: determine_harvest_weight(c, x['YYYYMMSOON_달']), axis=1)
    test['harvest_weight'] = test.apply(lambda x: determine_harvest_weight(c, x['YYYYMMSOON_달']), axis=1)

    train['mean_123'] = (train['1순'] + train['2순'] + train['3순']) / 3

    mean_encoding_1 = train.groupby('harvest_weight')['mean_123'].mean()
    train['1_encoding'] = train['harvest_weight'].map(mean_encoding_1)
    test['1_encoding'] = test['harvest_weight'].map(mean_encoding_1)

    
    std_encoding_1 = train.groupby('harvest_weight')['mean_123'].std()

    train['1std_encoding'] = train['harvest_weight'].map(std_encoding_1)
    test['1std_encoding'] = test['harvest_weight'].map(std_encoding_1)

    train.drop(['mean_123'],axis=1,inplace=True)

    def determine_season(month):
        if month in [3, 4, 5]:
            return 1  # 봄
        elif month in [6, 7, 8]:
            return 2  # 여름
        elif month in [9, 10, 11]:
            return 3  # 가을
        elif month in [12, 1, 2]:
            return 4  # 겨울
        return 0  # 알 수 없음
    train['season'] = train['YYYYMMSOON_달'].apply(determine_season)
    test['season'] = test['YYYYMMSOON_달'].apply(determine_season)
    train['season_sin'] = np.sin(2 * np.pi * train['season'] / 4)
    train['season_cos'] = np.cos(2 * np.pi * train['season'] / 4)
    test['season_sin'] = np.sin(2 * np.pi * test['season'] / 4)
    test['season_cos'] = np.cos(2 * np.pi * test['season'] / 4)
    train.drop(['season'],axis=1,inplace=True)
    test.drop(['season'],axis=1,inplace=True)

    
    ### 기존 "YYYYMMSOON" 단어가 포함된 피처 삭제
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON' in col and not col.endswith(('_년도', '_달', '_순'))], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON' in col and not col.endswith(('_년도', '_달', '_순'))], inplace=True)
    ### sin, cos 변환 함수 정의
    def apply_sin_cos_transform(value, period):
        sin_val = np.sin(2 * np.pi * value / period)
        cos_val = np.cos(2 * np.pi * value / period)
        return sin_val, cos_val

    def apply_sin_cos_transform_year(value):
        sin_val = np.sin(2 * np.pi * value)
        cos_val = np.cos(2 * np.pi * value)
        return sin_val, cos_val
    ### '년도', '달', '순'에 대해 각각 sin/cos 변환 적용
    for col in train.columns:
        if '_년도' in col:
            train[f'{col}_sin'], train[f'{col}_cos'] = zip(*train[col].map(lambda x: apply_sin_cos_transform_year(x - 2017)))  #  변환
            test[f'{col}_sin'], test[f'{col}_cos'] = zip(*test[col].map(lambda x: apply_sin_cos_transform_year(x - 2017)))  #  변환
        if '_달' in col:
            train[f'{col}_sin'], train[f'{col}_cos'] = zip(*train[col].map(lambda x: apply_sin_cos_transform(x, 12)))  # 달은 12개월 주기
            test[f'{col}_sin'], test[f'{col}_cos'] = zip(*test[col].map(lambda x: apply_sin_cos_transform(x, 12)))  # 달은 12개월 주기
        if '_순' in col:
            train[f'{col}_sin'], train[f'{col}_cos'] = zip(*train[col].map(lambda x: apply_sin_cos_transform(x, 3)))   # 순은 3단계 (상순, 중순, 하순)
            test[f'{col}_sin'], test[f'{col}_cos'] = zip(*test[col].map(lambda x: apply_sin_cos_transform(x, 3)))   # 순은 3단계 (상순, 중순, 하순)
    ### 기존 '년도', '달', '순' 피처 삭제
    train.drop(columns=[col for col in train.columns if col.endswith(('_년도', '_달', '_순'))], inplace=True)
    test.drop(columns=[col for col in test.columns if col.endswith(('_년도', '_달', '_순'))], inplace=True)


    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-1' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-1' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-2' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-2' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-3' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-3' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-4' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-4' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-5' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-5' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-6' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-6' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-7' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-7' in col], inplace=True)
    train.drop(columns=[col for col in train.columns if 'YYYYMMSOON_T-8' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if 'YYYYMMSOON_T-8' in col], inplace=True)



    # 누적합 기상 요소 목록
    feature_types = ['평균일강수량(mm)', '평균기온(℃)', '일조합(hr)', '일사합(MJ/m2)', '평균습도(%rh)']
    

    if(c == '깐마늘(국산)'): ## 전남, 경북 
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 

    elif(c == '사과'): ## 전남, 경북  
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 

    elif(c == '배'): ## 전남, 울산(경남)
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 

    elif(c == '상추'): ##경남, 서울경기
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 

    elif(c == '양파'): ## 경북, 경남
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 

    elif(c == '건고추'): ## 전남, 경북  
            train.drop(columns=[col for col in train.columns if '강원영동' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '강원영동' in col], inplace=True) 



 

    train.drop(columns=[col for col in train.columns if '금액' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '금액' in col], inplace=True)
    
    train.drop(columns=[col for col in train.columns if '전년 반입량' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '전년 반입량' in col], inplace=True)

    train.drop(columns=[col for col in train.columns if '증감률' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '증감률' in col], inplace=True)
    
    train.drop(columns=[col for col in train.columns if '최저습도' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '최저습도' in col], inplace=True)
        
  
    def calculate_cumulative_features_fixed(df):
        # Null 값이 있는 경우 0으로 대체
        df.fillna(0, inplace=True)
        
        all_cumulative_features = pd.DataFrame(index=df.index)
        existing_columns = set()  # 중복 컬럼명 방지를 위한 집합
        
        for feature_type in feature_types:
            feature_columns = [col for col in df.columns if feature_type in col]
            region_names = {col.split('_')[0] for col in feature_columns}
            
            for region in region_names:
                # 컬럼 정렬 후 출력해 확인
                region_feature_cols = sorted(
                    [col for col in feature_columns if col.startswith(region)],
                    key=lambda x: int(x.split('_T-')[-1]) if '_T-' in str(x) else 0
                )
                
                # 누적합 계산 및 확인
                cumulative_df = df[region_feature_cols].iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
                
                # 고유한 컬럼명 생성
                cumulative_df.columns = [
                    f'{region}_누적_{col.split("_", 1)[-1]}' + (f'_{i}' if f'{region}_누적_{col.split("_", 1)[-1]}' in existing_columns else '')
                    for i, col in enumerate(region_feature_cols, 1)
                ]
                
                # 기존 컬럼명 집합에 추가
                existing_columns.update(cumulative_df.columns)
                
                # 누적합 데이터 병합
                all_cumulative_features = pd.concat([all_cumulative_features, cumulative_df], axis=1)
        
        return all_cumulative_features

    # 누적합 계산
    train_cumulative = calculate_cumulative_features_fixed(train)
    test_cumulative = calculate_cumulative_features_fixed(test)
    print(train_cumulative.columns.to_list())
    # 기존 기상 요소 컬럼 제거
    drop_columns = [col for feature_type in feature_types for col in train.columns if feature_type in col]
    train = train.drop(columns=drop_columns)
    test = test.drop(columns=drop_columns)

    # 원본 데이터프레임에 누적 컬럼 추가
    train = pd.concat([train, train_cumulative], axis=1)
    test = pd.concat([test, test_cumulative], axis=1)

    train.drop(columns=[col for col in train.columns if '(mm)_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '(mm)_T' in col], inplace=True)

    train.drop(columns=[col for col in train.columns if '(℃)_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '(℃)_T' in col], inplace=True)

    train.drop(columns=[col for col in train.columns if '(hr)_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '(hr)_T' in col], inplace=True)
    
    train.drop(columns=[col for col in train.columns if '(MJ/m2)_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '(MJ/m2)_T' in col], inplace=True)
    
    train.drop(columns=[col for col in train.columns if '(%rh)_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '(%rh)_T' in col], inplace=True)

    train.drop(columns=[col for col in train.columns if '수입_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '수입_T' in col], inplace=True)

    train.drop(columns=[col for col in train.columns if '수출_T' in col], inplace=True)
    test.drop(columns=[col for col in test.columns if '수출_T' in col], inplace=True)


    if(c == '건고추'):
            train.drop(columns=[col for col in train.columns if '수출' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '수출' in col], inplace=True)        
            train.drop(columns=[col for col in train.columns if '수입' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '수입' in col], inplace=True)

    elif(c == '무'): ## 전남, 경북 
            train.drop(columns=[col for col in train.columns if '수입' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '수입' in col], inplace=True) 

    elif(c == '상추'): ## 전남, 경북  
            train.drop(columns=[col for col in train.columns if '수출' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '수출' in col], inplace=True) 

    elif(c == '양파'): ## 전남, 울산(경남)
            train.drop(columns=[col for col in train.columns if '수출' in col], inplace=True)
            test.drop(columns=[col for col in test.columns if '수출' in col], inplace=True) 


            
    # 개입 확률 계산 함수 (변화율 기반)
    def calculate_intervention_probability(prices, threshold=30, scaling_factor=0.005):
        # 변화율 계산: (현재 가격 - 과거 가격) / 과거 가격 * 100
        change_rate = ((prices[-1] - prices[0]) / prices[0]) * 100
        # 개입 확률 계산 공식
        intervention_probability = min(1, max(0, (abs(change_rate) - threshold) * scaling_factor))
        return intervention_probability

    # 비축 물량 방출 가능성 계산 함수 (이동 평균 기반)
    def release_probability(current_price, moving_average, threshold=15, scaling_factor=0.03):
        change_rate = ((current_price - moving_average) / moving_average) * 100
        return min(1, max(0, (change_rate - threshold) * scaling_factor))

    # 변동성 기반 개입 가능성 계산 함수
    def volatility_intervention(volatility, threshold=5, scaling_factor=0.02):
        return min(1, max(0, (volatility - threshold) * scaling_factor))

    # 개별 행마다 개입 피처 추가 함수
    def add_intervention_features(df):
        # 새 피처 컬럼 초기화
        df["가격 변화율에 따른 개입 확률"] = 0.0
        df["비축 물량 방출 가능성"] = 0.0
        df["변동성 기반의 개입 가능성"] = 0.0
        
        for idx, row in df.iterrows():
            # 각 행에서 과거 3시점과 현재 가격을 가져와 리스트로 저장
            prices = [
                row["평균가격(원)_T-3"],
                row["평균가격(원)_T-2"],
                row["평균가격(원)_T-1"],
                row["평균가격(원)"]
            ]
            
            # 개입 확률 계산
            intervention_probability = calculate_intervention_probability(prices)
            df.at[idx, "가격 변화율에 따른 개입 확률"] = intervention_probability

            # 비축 물량 방출 가능성 계산 (3일 이동 평균 기반)
            moving_average_3d = sum(prices[-3:]) / 3
            release_prob = release_probability(prices[-1], moving_average_3d)
            df.at[idx, "비축 물량 방출 가능성"] = release_prob

            # 변동성 기반 개입 가능성 계산
            volatility = pd.Series(prices).std()
            volatility_prob = volatility_intervention(volatility)
            df.at[idx, "변동성 기반의 개입 가능성"] = volatility_prob

        return df
    
    if c == "배추" or c == "무" or c == "양파":
        train = add_intervention_features(train)
        test = add_intervention_features(test)









    def climate_intervention_probability(current_value, average_value, threshold=30, scaling_factor=0.01):
        # 평균으로부터의 편차 계산
        deviation = abs(current_value - average_value)
        if deviation > threshold:
            intervention_probability = min(1, (deviation - threshold) * scaling_factor)
        else:
            intervention_probability = 0
        return intervention_probability

    # 기후 피처 개입 확률 계산 함수 (여러 기후 피처에 대해 적용)
    def add_climate_intervention_features(df, climate_features):
        for feature in climate_features:
            # train 데이터에서 해당 피처의 평균값을 구함
            average_value = train[feature].mean()
            
            # 각 기후 피처별 개입 확률 계산
            intervention_feature_name = f"{feature}_개입확률"
            df[intervention_feature_name] = df[feature].apply(
                lambda x: climate_intervention_probability(x, average_value)
            )
        return df

    # 적용할 기후 피처 목록
    climate_features = [
        '강원영동_누적_평균일강수량(mm)', '강원영동_누적_평균기온(℃)', 
        '강원영동_누적_일조합(hr)', '강원영동_누적_일사합(MJ/m2)', 
        '강원영동_누적_평균습도(%rh)'
    ]

    if c == "배추" or c == "무":
        train = add_climate_intervention_features(train, climate_features)
        test = add_climate_intervention_features(test, climate_features)








    def add_mean_features(df, quantity_col=f'{c}_총반입량'):
        df['가격_총반입량_평균'] = (df['평균가격(원)'] + df[quantity_col]) / 2
        return df
    if c != "건고추":
        train = add_mean_features(train)
        test = add_mean_features(test)
  
    def LPF2(df, low, order=1):
        # Columns to which the low-pass filter should be applied
        price_columns = [
            '평균가격(원)_T-8', '평균가격(원)_T-7', '평균가격(원)_T-6', 
            '평균가격(원)_T-5', '평균가격(원)_T-4', '평균가격(원)_T-3', 
            '평균가격(원)_T-2', '평균가격(원)_T-1', '평균가격(원)'
        ]
        new_df = pd.DataFrame()
        b, a = butter(N=order, Wn=low, btype='low')
        for col in price_columns:
            if col in df.columns: 
                filtered_col = lfilter(b, a, df[col])
                new_df[f"{col}_LPF"] = filtered_col  

        df = pd.concat([df, new_df], axis=1)
        
        return df
    
    #train = LPF(train, low=0.1, order=1)
    #test = LPF(test, low=0.1, order=1)

  
    def LPF(df, low, order=1):
        # Columns to which the low-pass filter should be applied
        price_columns = [
            '평균가격(원)_T-2', '평균가격(원)_T-1', '평균가격(원)'
        ]
        new_df = pd.DataFrame()
        b, a = butter(N=order, Wn=low, btype='low')
        for col in price_columns:
            if col in df.columns: 
                filtered_col = lfilter(b, a, df[col])
                new_df[f"{col}_LPF"] = filtered_col  

        df = pd.concat([df, new_df], axis=1)
        
        return df
    
    train = LPF(train, low=0.1, order=1)
    test = LPF(test, low=0.1, order=1)



    
    # 0 값 일정 수치 이상 피처 드랍
    #total_samples = len(train)
    #features_to_drop = [col for col in train.columns if (train[col] == 0).sum() / total_samples >= 0.3]

    #train = train.drop(columns=features_to_drop)
    #test = test.drop(columns=features_to_drop)



    # 선형보간
    def fill_nan_with_avg_all_columns(df):
        df = df.interpolate(method='linear', limit_direction='both', axis=0)
        return df
    train = fill_nan_with_avg_all_columns(train)

    
    return train, test
