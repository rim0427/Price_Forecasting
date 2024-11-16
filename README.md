# Price_Forecasting [(Link)](https://dacon.io/competitions/official/236381/leaderboard)
# 인기상 [(Link)](https://dacon.io/competitions/official/236417/leaderboard)

## 🏆 Result
## **Public score 1st** 0.08605 | **Private score 1st** 0.07665 | 최종 1등

<img width="50%" src="https://github.com/user-attachments/assets/51ed525a-5ab5-439a-8086-cc0ac25b2eca"/>  

주최 : ```디지털플랫폼정부위원회, 농림축산식품부```

규모 : 총 1500여명 참가
  

  
## 🧐 About
국민생활과 밀접한 10개 농산물 품목의 가격 예측
(배추, 무, 양파, 사과, 배, 건고추, 깐마늘, 감자, 대파, 상추)
  
## [대회 방식]
### 본 경진대회는 1차 예선, 2차 예선 그리고 본선으로 진행

### ```1차 예선``` : Private 리더보드 상위 20팀이 2차 예선에 진출(평가 산식 : NMAE)  
[문제 상세 설명]  
1) 학습 데이터는 2018년 ~ 2021년의 순 단위(10일)의 데이터가 주어지며,  

2) 평가 데이터는 추론 시점 T가 비식별화된 2022년의 순 단위의 데이터가 주어집니다.  

3) 평가 데이터 추론은 추론 시점 T 기준으로 최대 3개월의 순 단위의 입력 데이터를 바탕으로 T+1순, T+2순, T+3순의 평균가격을 예측해야합니다.  


### ```2차 예선``` : 별도의 대회 페이지에서 진행하고 추가된 데이터가 제공되며, 2차 예선의 Private 리더보드 점수와 추가 평가 점수를 합산한 점수 상위 10팀이 본선에 진출(평가 산식 : Weighted NMAE + 추론 시간)  
[문제 상세 설명]  
1) 학습 데이터는 2018년 ~ 2022년의 순 단위(10일)의 데이터가 주어지며,  

2) 평가 데이터는 식별화된 추론 시점 T가 2023년 ~ 2024년의 순 단위의 데이터가 주어집니다.  

3) 평가 데이터 추론은 추론 시점 T 기준으로 최대 3개월의 순 단위의 입력 데이터를 바탕으로 T+1순, T+2순, T+3순의 평균가격을 예측해야합니다.  


### ```본선``` : 오프라인 발표 평가를 통해 최종 수상자가 선정(평가 : 발표평가 50%(EDA, 모델링, 모델 활용)+ 정량평가 50%(2차 예선 최종 환산 점수))  
[본선 산출물]  

1) 결과보고서(모델링) 자료 : 2차 예선 대회에서 개발한 AI 모델의 결과보고서 작성  

2) 발표 자료 : 작성한 결과보고서를 바탕으로 평가 항목에 따른 발표 자료 작성  



---

## 🔥**대회 접근법(개조식)**

---

# **1차 예선**

### **주요 분석 및 결정 사항**
1. **통합 모델링 vs. 품목별 모델링**
2. **동일 품목 내 여러 품종 데이터 처리**
3. **머신러닝 모델 vs. 시계열 딥러닝 모델 선택**

---

## **1. 통합 모델링 vs. 품목별 모델링**

- **통합 모델링**
  - 모든 데이터를 하나의 모델로 학습하여, **데이터 양 증가**와 함께 **일반화 성능** 향상 기대 가능.
  - 여러 품목의 데이터를 동시에 학습해, **공통된 패턴**을 반영할 수 있음.

- **품목별 모델링**
  - 각 품목의 **고유한 계절성과 변동성**을 효과적으로 학습 가능.
  - 품목별로 맞춤형 모델을 설계해, **특화된 예측 성능** 확보 가능.

- **결정**: **품목별 모델링** 채택.
  - **이유**: EDA 결과, 농산물마다 고유한 계절성 패턴과 가격 변동 특성이 다르게 나타났기 때문.

---

## **2. 동일 품목 내 여러 품종 데이터 처리**

- **가정**:
  - 동일 품목 내 품종들은 **유사한 계절성과 분포**를 보일 가능성이 높음.
    - 이유: 같은 품목에 속하는 품종들은 비슷한 **재배 환경**과 **수확 시기**를 가짐.

- **검증**:
  - **UMAP 분석**: 동일 품목 내 품종들이 **비슷한 군집 형성** 확인.
  - **시간별 EDA**: 동일 품목 내 품종들 간의 **계절성 패턴**과 **가격 변동**이 유사하게 나타남.

- **결정**: 동일 품목 내 품종 데이터를 **다변량 입력 형식**으로 통합해 학습.
  - **이유**: 품종별 미세한 차이를 반영하면서도, **공통적인 패턴**을 학습할 수 있음.
  - **효과**: 데이터 양 증가로 **모델의 일반화 성능** 강화.

---

## **3. 머신러닝 모델 vs. 시계열 딥러닝 모델 선택**

- **머신러닝 모델**
  - **비선형 관계 학습**과 다양한 **파생 변수 활용**에 강점이 있음.
  - 기후, 경제적 요인, 외부 개입 등 다양한 **복잡한 변수**들을 효과적으로 반영 가능.

- **시계열 딥러닝 모델**
  - 시간적 흐름과 **계절성 패턴** 학습에 유리.
  - **LSTM, Transformer** 모델은 최근의 **시간적 변화**를 잘 반영 가능.

- **결정**: **앙상블 전략** 채택.
  - **이유**: 단기 예측에서는 시계열 모델의 강점을, 장기 예측에서는 머신러닝 모델의 강점을 활용하기 위함.
  - **구성**:
    - **단기 예측**: 시계열 모델의 가중치 부여.
    - **장기 예측**: 머신러닝 모델의 가중치 부여.

---

## **머신러닝 모델의 시계열 데이터 처리 방식**

- **고려한 방법**:
  1. **시점별 독립 입력**:
     - 각 시점의 데이터를 그대로 사용하여 현재 시점의 예측에 활용.
     - 간단한 구조로 빠른 학습 가능.
  2. **Transpose 방식**:
     - 과거 데이터를 **Transpose(전치)**하여 현재 시점의 피처로 사용.
     - 시간적 흐름과 과거 패턴을 반영 가능.
  3. **다변량 입력 방식**:
     - 동일 품목 내의 다른 품종명 데이터를 **다변량 형식**으로 함께 입력.
     - 품종 간 상호작용과 공통 패턴 학습 가능.

- **결정**: 머신러닝 모델은 **다변량 입력 방식(3번)** 사용.
  - **이유**: 품종 간 관계를 반영하여 예측 성능을 높이기 위함.

---

## **최종 전략 및 성과**
- **품목별 모델링 채택**: 각 품목의 고유 패턴을 반영해 예측 성능 강화.
- **다변량 입력 형식 활용**: 동일 품목 내 다양한 품종 데이터를 효과적으로 학습.
- **앙상블 전략 적용**: 단기 예측에는 시계열 모델, 장기 예측에는 머신러닝 모델을 결합하여 최종 예측 성능 극대화.

---

---

# **2차 예선: 외부 데이터 활용 및 현실 세계 적용**

### **목표 및 전략**
- 1차 예선의 모델링을 기반으로, **외부 데이터 활용** 및 **현실 세계에서의 모델 적용**에 중점.
- 농산물 가격은 **자연적**, **경제적**, **시계열적**, **외부적 요인**에 의해 결정됨을 인식하고, 이 **4가지 요인**을 중점으로 분석 및 모델링 진행.

---

## **1. 자연적 요인 분석**
- **환경 변수 분석**:
  - 농산물 가격에 영향을 미치는 주요 자연적 요인인 **강수량, 기온, 일조량, 습도** 등을 분석.
  - 작물의 생육과 생산량에 직접적인 영향을 미치는 변수로 고려.

- **누적값 피처 생성**:
  - 계절성과 날씨의 변동성으로 인한 노이즈를 줄이기 위해, **누적값 피처** 생성.
  - 장기적인 환경 영향을 반영해 예측 성능 개선.

- **이상 기후 분석**:
  - 태풍, 폭염 등의 **극단적 기상 이벤트**가 농산물 가격에 미치는 영향 분석.
  - 이러한 이벤트를 반영할 수 있는 추가 변수를 생성해 모델에 활용.

---

## **2. 경제적 요인 분석**
- **경제 지표 분석**:
  - 농산물 가격에 영향을 줄 수 있는 **유가, 환율, 총반입량** 등의 경제 지표 분석.
  - 재배 및 유통 비용과 밀접한 관계가 있어 중요한 변수로 활용.

- **Granger 인과 검정**:
  - 경제적 요인들이 농산물 가격의 **선행 지표**인지 검증하기 위해, Granger 인과 검정 실시.
  - **결과**: 유가와 총반입량이 농산물 가격의 중요한 선행 지표임을 확인.

- **경제적 외부 변수 추가**:
  - 주요 경제 지표들을 모델 학습에 반영해, 가격 예측의 정확도 향상.

---

## **3. 시계열적 요인 분석**
- **추세 및 계절성 분석**:
  - 농산물 가격 데이터의 **추세(Trend)**와 **계절성(Seasonality)**을 분석.
  - 계절에 따른 반복적인 가격 변동 패턴을 확인.

- **Low-Pass Filter 사용**:
  - 이동 평균 대신 **Low-Pass Filter**를 사용해, 더 정확하게 **추세 성분**을 추출.
  - 노이즈를 줄이고, 추세 분석의 정확도 향상.

- **sin, cos 변환**:
  - 계절성과 주기성을 효과적으로 반영하기 위해, **sin, cos 변환**을 사용해 시계열 데이터를 주기적 패턴으로 변환.

---

## **4. 외부적 요인 분석**
- **정부 개입 분석**:
  - **수매, 방출, 보조금 지급** 등의 정부 개입이 농산물 가격에 미치는 영향을 분석.
  - 인위적인 가격 조정으로, AI 모델이 학습하기 어려운 요인임을 고려.

- **개입 확률 피처 생성**:
  - 가격 급락이나 급등 시기에 **시장 개입 확률 변수** 추가.
  - 가격 변화율, 변동성 등을 기준으로 개입 확률을 계산해 모델이 외부 개입 상황을 잘 반영할 수 있도록 함.

- **실제 개입 데이터 검증**:
  - EDA 과정에서 가격 급변 시기에 실제로 정부의 **물량 방출**이 있었음을 확인.
  - 외부 개입 변수가 효과적으로 작동함을 검증.

---

## **모델 경량화 및 선택**
- **본선 진출을 위해** 정확도뿐만 아니라 **추론 시간**도 고려.
- **LGBM**과 **DLinear** 모델 선택:
  - 1차 예선에서 사용했던 **Extra Tree, Random Forest, SegRNN, iTransformer** 대신, 더 가볍고 빠른 모델 채택.

### **DLinear 모델 사용 이유**
- 데이터를 **추세(Trend)**와 **계절성(Seasonality)**으로 분해하여 학습.
- 학습과 추론이 빠르며, **계절성 패턴 강화를 위해 Attention 메커니즘** 추가.
- **이유**: 김장철과 같은 중요한 계절적 이벤트에 더 집중해 학습할 수 있고, **추론 시간 단축** 가능.

### **LGBM 모델 사용 이유**
- **leaf-wise 성장 방식**과 **Histogram 기반 학습**으로 비선형적 변화를 잘 포착.
- 급격한 변동과 예외적인 패턴을 세밀하게 분할해, 빠르게 적응 가능.
- **이유**: 계산량 감소로 추론 시간을 단축하고, 다양한 파생 변수를 효과적으로 반영해 복잡한 패턴 학습 가능.

---

## **새로운 모델링 전략: TIME ML**
- **개발 배경**:
  - 기존의 시계열 모델과 머신러닝 모델의 한계를 극복하기 위해 설계된 혼합 모델.

- **구조**:
  - 시계열 모델의 예측값을 머신러닝 모델의 입력 피처로 사용.
  - 시계열 데이터의 **시간적 흐름**과 **장기 추세**를 더 잘 반영할 수 있도록 설계.

- **사용 이유**:
  - **자기회귀적 구조** 기반으로, 시계열 모델의 예측값을 피처로 사용해 **시간 종속성**을 효과적으로 학습.
  - 단기 예측에서는 시계열 모델의 **패턴 학습 능력**, 장기 예측에서는 머신러닝 모델의 **관계 학습 능력**을 결합해 성능 극대화.

---

## **최종 전략 및 성과**
- **외부 데이터 활용 강화**: 자연적, 경제적, 시계열적, 외부적 요인을 모두 반영해 예측 정확도 향상.
- **경량화 모델 선택**: DLinear와 LGBM 모델로 빠른 추론 시간 확보.
- **TIME ML 전략 적용**: 단기 예측에서는 시계열 모델, 장기 예측에서는 머신러닝 모델의 장점을 결합.
- **성과**: 단기와 장기 예측 모두에서 우수한 성능을 보이며, 본선 진출 가능성 극대화.

---

---

# **본선: AI 기반 수매-방출 의사결정 시스템 개발**

### **목표 및 개요**
- 본선에서는 **AI 기반 수매-방출 의사결정 시스템**을 개발.
- **목표**: 농산물 시장의 가격 안정화.
- 농민, 소비자, 정부 모두에게 이익이 되는 솔루션 제공.

---

## **1. 문제 정의**
- **농산물 가격 문제**:
  - 가격이 **하락할 경우**: 농민의 수익 감소 → 생산 의욕 저하.
  - 가격이 **상승할 경우**: 소비자의 부담 증가 → 구매력 감소.
- **해결 목표**:
  - AI 시스템이 가격 예측을 통해 **시장 가격을 조정**하여, 농민과 소비자의 문제를 동시에 해결.

---

## **2. 의사결정 전략과 최적화 과정**
### **1) AI 기반 수매-방출 의사결정 시스템 1**
- N 시점에서 예측된 가격이 **‘심각’ 구간에 도달할 경우**:
  - '심각' 구간은 농림축산식품부 기준으로, 가격 변동이 큰 상황을 의미.
  - **N-1 시점에서 미리 조정**하여, 시장 충격을 최소화하고 안정화를 유도.

### **2) AI 기반 수매-방출 의사결정 시스템 2 : 조정 강도 설정**
- **조정 강도**:
  - 예측된 가격이 **상한선을 초과**: 방출량 증가 → 가격 하락 유도.
  - 예측된 가격이 **하한선 이하**: 수매량 증가 → 가격 상승 유도.

### **3) AI 기반 수매-방출 의사결정 시스템 3 : TIME ML 모델을 통한 최적화 과정**
- **TIME ML 모델 사용 이유**:
  - 자기회귀 구조로 이전 예측값을 반영해 **시간 종속성을 학습** 가능.
- **최적화 과정**:
  1. N-1 시점에서 예측된 가격을 바탕으로 수매/방출 조정.
  2. 조정된 가격을 TIME ML 모델에 입력해 N 시점의 새로운 예측 수행.
  3. 이 과정을 반복해 **최적의 수매/방출량 결정**.

### **4) AI 기반 수매-방출 의사결정 시스템 4 : 비용 고려 및 목적 함수 설계**
- **목적 함수 목표**:
  - 가격이 **‘심각’ 구간에서 벗어나도록** 조정.
- **비용 요소**:
  - **수매 및 방출 비용**: 물량 조정에 따른 직접 비용.
  - **보관 및 폐기 비용**: 조정된 물량의 관리 비용.
- **비용 최적화**:
  - 모든 비용을 최소화하며, 시장 안정화 목표 달성.

---

## **3. AI 기반 수매-방출 의사결정 시스템의 강점과 기대효과**
- **AI 기반 수매-방출 의사결정 시스템 강점**:
  - N 시점의 예측 결과가 '심각' 구간에 도달하기 전에, **N-1 시점에서 조정**하여 시장 변동성 완화.
  - 실시간 예측과 조정을 통해 **가격 안정화**에 효과적.
- **기대 효과**:
  - **정부**: 과잉 생산과 가격 변동으로 인한 **보관 및 폐기 비용 절감**.
  - **농민**: 최소 가격 보장을 통해 **안정적인 소득 확보** 및 지속 가능한 생산 가능.
  - **소비자**: 가격 변동성이 줄어, **안정된 가격으로 농산물 구매 가능**.

---

## **4. AI 기반 수매-방출 의사결정 시스템의 차별화 요소**
- **기존 시스템 대비 차별점**:
  - 기존 시스템은 **정해진 규칙**에 따라 작동하며, 시장 변화에 민첩하게 대응하지 못함.
  - AI 기반 시스템은 **실시간 가격 예측**을 통해, 시장 데이터를 반영한 **빠르고 유연한 대응** 가능.
- **효과**:
  - 더 정확한 가격 조정과 시장 안정화 실현 가능.

---

## **5. 추가 모델 활용 제안: 선물 거래 시스템과 맞춤형 농산물 보험**

### **1) 선물 거래 시스템 (Futures Trading System)**
- **사용 이유**:
  - 소비자는 미리 **정해진 가격**에 농산물을 구매 → **가격 급등 리스크 감소**.
  - 농가는 미리 판매 계약을 통해 **안정적인 수익 확보** 가능.
- **효과**:
  - 시장의 **가격 변동성 감소** 및 **수급 예측 가능성** 향상.

### **2) 맞춤형 농산물 보험 (Customized Agricultural Insurance)**
- **사용 이유**:
  - AI 예측 모델을 통해 가격 변동을 분석하고, **농가나 유통업체의 특성에 맞춘 보장 범위 설계**.
- **효과**:
  - 가격 하락 시 **손실 보상**으로 농가의 재정적 안정성 강화.
  - 농업 생태계의 안정화 및 지속 가능한 농업 지원.

---

## **6. 최종 요약**
- **문제 정의**: 농민과 소비자의 문제를 해결하기 위해, AI 기반 수매-방출 시스템 설계.
- **의사결정 전략**: N 시점에서 예측된 가격이 '심각' 구간에 도달할 경우, N-1 시점에서 선제적 조정 수행.
- **TIME ML 모델 최적화**: 자기회귀 구조와 반복적인 최적화 과정 적용.
- **비용 고려 및 목표 달성**: 가격 안정화와 비용 최소화를 동시에 달성.
- **기대 효과**: 정부, 농민, 소비자 모두에게 긍정적인 효과 제공.
- **차별화 요소**: 실시간 예측과 민첩한 대응으로 기존 시스템 대비 우수한 성능 제공.

### 추론 시간 최적화

- 평가 기준에 포함되는 추론 시간 최적화
- 총 52개의 test 파일을 처리해야 하기 때문에 파일을 불러올 때 상당한 시간이 소요됨
- 이를 해결하기 위해 ThreadPoolExecutor을 이용해 파일 로드를 병렬로 수행
- 전체 데이터 처리 시간 단축 및 CPU와 I/O 자원을 효율적으로 사용  
---




===================================================================================
## 🔥**대회 접근법(줄글형식)**

# **1차 예선**

1차 예선에서는 농산물 데이터를 어떻게 모델링할지에 대해 결정이 필요했습니다. 우선, 모든 농산물을 하나의 통합 모델로 학습할지, 아니면 각 품목별로 개별 모델링할지를 고민했습니다. 통합 모델링의 경우, 데이터의 양이 많아지며 일반화 성능이 높아질 가능성이 있습니다. 하지만 EDA 분석 결과, 각 농산물마다 고유한 계절성과 가격 변동 패턴이 다르게 나타났습니다. 예를 들어, 배추는 김장철에 가격이 급등하고, 감자는 여름철 기후에 민감하게 반응했습니다. 이처럼 품목마다 특성이 다르기 때문에 통합 모델링 시 오히려 예측의 정확도가 떨어질 수 있고, 노이즈가 증가할 위험이 있었습니다. 이를 고려하여, 품목별 모델링을 선택하여 각 농산물의 고유 패턴을 반영하고 예측 성능을 높였습니다.

다음으로, 동일 품목명 내 다양한 품종(예: 감자의 수미, 대지 품종)을 어떻게 처리할지에 대한 고민이 있었습니다. 동일 품목 내의 품종들은 비슷한 재배 환경과 수확 시기를 가지기 때문에, 유사한 가격 변동 패턴을 보일 것이라는 가정을 세웠습니다. 이를 검증하기 위해 UMAP 분석과 시간에 따른 가격 변동 EDA를 진행했습니다. 분석 결과, 동일 품목 내의 품종들은 비슷한 군집을 형성하며 유사한 계절성 패턴과 가격 변동을 보였습니다. 따라서, 동일 품목 내 다양한 품종 데이터를 하나의 다변량 모델에서 통합 학습하는 접근을 택했습니다. 이를 통해 품종별 미세한 차이도 반영하면서, 공통적인 패턴을 학습할 수 있었으며 데이터 양의 증가로 인해 모델의 일반화 성능도 강화되었습니다.

마지막으로, 시계열 딥러닝 모델과 머신러닝 모델 중 어떤 모델을 사용할지 고민했습니다. 머신러닝 모델은 비선형 관계 학습과 다양한 파생 변수 활용이 가능해, 기후나 경제적 요인 등 복잡한 변수를 반영하기 유리합니다. 반면, 시계열 딥러닝 모델은 최근의 시간적 흐름과 계절성 패턴을 학습하기에 적합해 단기 예측에 강점을 보입니다. 이를 고려해, 단기 예측에서는 시계열 모델의 가중치를 더 높이고, 장기 예측에서는 머신러닝 모델의 가중치를 높이는 앙상블 전략을 채택했습니다.

또한, 머신러닝 모델에서는 시계열 데이터를 다루는 세 가지 방법을 고려했습니다. 첫 번째는 시점별 데이터를 독립적으로 입력하는 방식으로, 간단한 구조로 빠르게 학습할 수 있습니다. 두 번째는 과거 데이터를 전치(Transpose)하여 현재 시점의 피처로 사용하는 방식으로, 시간적 흐름과 과거 패턴을 반영할 수 있습니다. 세 번째는 동일 품목 내의 다양한 품종 데이터를 다변량 형식으로 입력하는 방식으로, 품종 간 상호작용과 공통 패턴을 학습할 수 있습니다. 이 중에서, 품종 간 관계를 잘 반영하기 위해 다변량 입력 방식을 선택하였습니다.

결과적으로, 1차 예선에서는 품목별 모델링과 다변량 데이터를 활용한 머신러닝 모델링을 통해 각 품목의 고유 패턴을 반영하고, 앙상블 전략으로 단기 및 장기 예측의 성능을 극대화했습니다.


# **2차 예선**

2차 예선에서는 1차 예선과 달리 외부 데이터를 활용할 수 있었습니다. 따라서, 기존 1차 예선의 모델링 전략을 기반으로 외부 데이터를 추가 분석하고 현실적인 모델 적용 가능성에 중점을 두었습니다. 농산물 가격은 다양한 요인이 복합적으로 작용해 결정되므로, 이를 ‘자연적 요인’, ‘경제적 요인’, ‘시계열적 요인’, ‘외부적 요인’의 네 가지 카테고리로 나누어 분석을 진행했습니다.

우선, 자연적 요인으로 농산물 가격에 영향을 미치는 환경 변수를 분석했습니다. 주요 변수로는 강수량, 기온, 일조량, 습도 등이 포함되었으며, 이는 작물의 생육과 생산량에 직접적인 영향을 미치기 때문에 중요한 요소로 고려되었습니다. 그러나 이러한 자연적 요인들은 계절성과 날씨의 변동성으로 인해 노이즈가 많았습니다. 이를 해결하기 위해 누적값 피처를 생성하여 장기적인 환경 영향을 반영했습니다. 또한, 태풍이나 폭염과 같은 극단적인 기상 이벤트가 농산물 가격에 미치는 영향을 분석하고, 이를 반영할 수 있는 추가 변수를 생성했습니다.

다음으로, 경제적 요인을 분석했습니다. 농산물 가격은 유가, 환율, 총반입량 등과 같은 경제 지표와 밀접하게 연관되어 있습니다. 이러한 경제 지표들은 농산물의 재배 및 유통 비용에 영향을 미치기 때문에 중요한 분석 대상으로 선정되었습니다. 특히, Granger 인과 검정을 통해 경제적 요인이 농산물 가격의 선행 지표로 작용하는지 검토했습니다. 그 결과, 유가와 총반입량이 농산물 가격의 중요한 선행 지표임이 확인되었고, 이를 모델 학습에 반영함으로써 예측의 정확도를 높였습니다.

세 번째로, 시계열적 요인을 분석했습니다. 농산물 가격 데이터는 일반적으로 계절성(Seasonality)과 추세(Trend)를 포함하고 있습니다. 이를 파악하기 위해 Low-Pass Filter를 사용해 추세 성분을 추출했으며, sin, cos 변환을 통해 주기성을 효과적으로 반영했습니다. 이 방식은 이동 평균보다 노이즈를 줄이면서도 계절적 패턴을 더 잘 반영할 수 있었습니다.

마지막으로, 외부적 요인으로 정부 개입의 영향을 분석했습니다. 수매, 방출, 보조금 지급 등의 정부 정책은 농산물 가격에 직접적인 영향을 미칠 수 있으며, 이는 AI 모델이 학습하기 어려운 변수입니다. 이를 보완하기 위해 가격 급락이나 급등 시기에는 시장 개입 확률 변수를 추가했습니다. 실제 EDA 과정에서도 가격 급변 시기에 정부의 물량 방출이 있었음을 확인했고, 이를 통해 외부 개입 변수가 효과적으로 작동함을 검증했습니다.

모델링 단계에서는 정확도뿐만 아니라 추론 시간도 중요한 고려 사항이었습니다. 이에 따라, 1차 예선에서 사용했던 Extra Tree, Random Forest, SegRNN, iTransformer 등과 같은 무거운 모델 대신, 더 가볍고 빠른 LGBM과 DLinear 모델을 선택했습니다. DLinear 모델은 데이터를 추세와 계절성으로 분해해 학습할 수 있으며, 빠른 추론이 가능하고 Attention 메커니즘을 추가해 계절적 이벤트에 더 집중할 수 있도록 설계했습니다. LGBM 모델은 leaf-wise 성장 방식과 Histogram 기반 학습을 통해 비선형적 변화를 효과적으로 포착할 수 있으며, 계산량이 줄어들어 추론 시간이 단축되었습니다.

또한, 2차 예선에서 이번 Task에 맞춰 개발된 TIME-ML 모델은 기존의 시계열 모델과 머신러닝 모델의 한계를 극복하기 위한 혼합 모델입니다. 이 모델은 시계열 모델의 예측값을 머신러닝 모델의 입력 피처로 사용하는 구조로 설계되었습니다. 이를 통해 시계열 데이터의 시간적 흐름과 장기 추세를 더 잘 반영할 수 있습니다. TIME ML 모델은 자기회귀적 구조를 기반으로 하여, 기존 머신러닝 모델들과 달리 시계열 모델의 예측값을 직접 피처로 사용합니다. 이를 통해 시간 종속성을 효과적으로 학습할 수 있으며, 단기 예측에서는 시계열 모델의 패턴 학습 능력, 장기 예측에서는 머신러닝 모델의 관계 학습 능력을 결합해 성능을 극대화했습니다.

결과적으로, 2차 예선에서는 외부 데이터를 활용한 분석과 피처 엔지니어링 그리고 TIME ML 모델의 도입으로 농산물 가격 예측의 정확도를 크게 향상시킬 수 있었습니다. 이 전략은 단기와 장기 예측 모두에서 우수한 성과를 보였으며, 본선 진출을 위한 중요한 발판이 되었습니다.



# **본선
**

본선에서는 농산물 가격의 안정화를 목표로 한 **AI 기반 수매-방출 의사결정 시스템**을 개발했습니다. 이 시스템은 농민, 소비자, 그리고 정부 모두에게 이익을 제공할 수 있는 솔루션으로 설계되었습니다. 농산물 가격은 변동성이 크며, 가격 하락 시 농민의 수익 감소와 생산 의욕 저하가 발생하고, 가격 상승 시에는 소비자의 부담이 증가해 구매력이 낮아지는 문제가 있습니다. 따라서, AI 시스템을 활용해 가격 예측을 기반으로 시장 가격을 조정하여 이러한 문제를 해결하고자 했습니다.

**AI 기반 수매-방출 시스템**의 핵심 전략은 N 시점에서 예측된 가격이 농림축산식품부의 기준에 따른 ‘심각’ 구간에 도달할 경우, 그 이전 시점인 N-1 시점에서 선제적으로 조정 조치를 취하는 것입니다. '심각' 구간은 가격 변동성이 큰 상황을 의미하며, 이를 방지하기 위해 N-1 시점에서 미리 대응함으로써 시장 충격을 최소화하고 가격 안정화를 유도합니다.

조정 강도는 예측된 가격의 심각성에 따라 결정됩니다. 예를 들어, 가격이 상한선을 초과하면 방출량을 증가시켜 가격을 낮추고, 하한선 이하로 떨어지면 수매량을 늘려 가격을 상승시키도록 설계되었습니다. 이를 통해 시장 가격이 극단적인 변동을 보이지 않도록 조절하며, 필요한 조정 강도는 예측된 가격의 심각성에 비례해 설정됩니다.

본 시스템에서 중요한 역할을 하는 것은 **TIME ML 모델**입니다. 이 모델은 자기회귀 구조를 기반으로 하여, 이전 예측값을 피처로 사용해 시간 종속성을 반영할 수 있습니다. 최적화 과정은 다음과 같이 이루어집니다. 먼저, N-1 시점에서 예측된 가격을 바탕으로 수매와 방출을 조정한 뒤, 조정된 가격을 TIME ML 모델에 입력해 N 시점의 새로운 예측값을 산출합니다. 이 과정을 반복해 최적의 수매와 방출량을 결정하게 되며, 이를 통해 가격 안정화를 효과적으로 이끌어낼 수 있습니다.

비용 측면에서는 수매와 방출 과정에서 발생하는 직접적인 비용뿐만 아니라, 보관 및 폐기 비용도 고려했습니다. 시스템의 목적 함수는 가격이 ‘심각’ 구간에서 벗어나는 것을 목표로 하며, 동시에 모든 비용을 최소화하도록 설계되었습니다. 이를 통해, 비용 효율적인 의사결정을 내리면서도 시장 안정화를 달성할 수 있었습니다.

AI 기반 수매-방출 의사결정 시스템의 강점은, 실시간 예측을 통해 시장 상황에 맞춰 신속하게 대응할 수 있다는 점입니다. 기존의 수매-방출 시스템은 정해진 규칙에 따라 작동하기 때문에 시장 변화에 민첩하게 반응하기 어렵습니다. 반면, AI 기반 시스템은 실시간 데이터를 반영해 빠르고 유연한 의사결정이 가능하며, 이를 통해 가격 조정과 시장 안정화를 더 정확하게 수행할 수 있습니다.

본 시스템의 기대 효과는 다음과 같습니다. 먼저, 정부는 과잉 생산이나 가격 변동으로 인한 보관 및 폐기 비용을 절감할 수 있습니다. 농민은 최소 가격 보장이 이루어져 안정적인 소득을 확보할 수 있으며, 이를 통해 지속 가능한 생산 활동이 가능해집니다. 소비자는 가격 변동성이 줄어들어 안정된 가격으로 농산물을 구매할 수 있어, 생활비 부담이 줄어듭니다.

또한, 본선에서는 두 가지 추가적인 모델 활용 방안을 제안했습니다. 첫째, **선물 거래 시스템**은 소비자가 미리 정해진 가격에 농산물을 구매할 수 있도록 하여, 가격 급등의 리스크를 줄여줍니다. 농가는 미리 판매 계약을 통해 안정적인 수익을 확보할 수 있으며, 이를 통해 시장의 가격 변동성을 줄이고 수급을 예측 가능하게 합니다. 둘째, **맞춤형 농산물 보험**은 AI 예측 모델을 기반으로 가격 변동을 분석하고, 개별 농가나 유통업체의 특성에 맞춰 보장 범위를 설계합니다. 이는 일반적인 표준 보험보다 더 정확하게 리스크를 관리할 수 있으며, 가격 하락 시 손실을 보상해 농가의 재정적 안정성을 강화하고, 농업 생태계의 지속 가능성을 높이는 데 기여합니다.

결론적으로, AI 기반 수매-방출 의사결정 시스템은 농민, 소비자, 정부 모두에게 긍정적인 영향을 미치는 솔루션입니다. 단기와 장기 예측 모두에서 TIME ML 모델의 성능이 돋보였으며, 정확한 예측과 민첩한 대응을 통해 시장 가격을 안정화하는 데 성공했습니다. 이로 인해 본 시스템은 기존의 정적인 정책보다 더 효과적인 가격 안정화 방법으로 평가받을 수 있었습니다.

---





---

### 📖 **Dataset Info**

---

#### **1. Train Data (폴더)**

- **train_1.csv:**  
  - **품목:** 배추, 무, 양파, 감자, 대파
  - **데이터 출처:** 가락도매 시장의 평균 경락 가격 정보

- **train_2.csv:**  
  - **품목:** 건고추, 깐마늘, 상추, 사과, 배
  - **데이터 출처:** 건고추, 깐마늘은 중도매 가격, 상추, 사과, 배는 소매 가격의 평균 정보

---

#### **2. Meta Data (폴더)**

- **TRAIN_경락정보_가락도매_2018-2022.csv:**  
  - 가락도매 시장의 가격 정보 (타겟 품목 제외)
- **TRAIN_경락정보_산지공판장/전국도매_2018-2022.csv:**  
  - 산지 및 전국 도매 시장의 가격 정보
- **TRAIN_기상_2018-2022.csv:**  
  - 기상 데이터
- **TRAIN_소매/중도매_2018-2022.csv:**  
  - 소매 및 중도매 가격 정보 (타겟 품목 제외)
- **TRAIN_수출입/생산정보/물가지수_2018-2022.csv:**  
  - 수출입, 생산, 물가지수 정보 (Data Leakage 주의)

---

#### **3. Test Data (폴더)**

- **2023-2024년 평가 데이터:**  
  - 최대 3개월의 예측 입력 시점 포함 (총 52개 샘플)
  - **TEST_00_1.csv ~ TEST_51_2.csv:** 예측용 데이터

- **Meta Files:**
  - **TEST_경락정보/기상/소매/중도매 (00~51).csv:** 각 시점별 추가 메타 정보
  - **TEST_수출입/생산정보/물가지수_2023-2024.csv:** 최신 수출입, 생산, 물가지수 데이터 (Data Leakage 주의)

---

#### **4. Submission File**

- **sample_submission.csv:**  
  - 각 품목의 +1순, +2순, +3순 예측 가격 (TEST 파일별 예측 결과)

---

#### **5. 외부 데이터 활용**

- **공개 데이터 활용 가능:**  
  - 농식품 공공데이터 포털, 범정부 공공데이터 포털, 농사로, 농넷, 스마트팜코리아 등
  - **기타:** 자체 보유 데이터 및 민간 융·복합 데이터 활용 가능




## **🔧 Feature Engineering**

---

### 1. **자연적 요인 (Environmental Factors)**
- **누적 기상 데이터 생성**
  - 기상 데이터(강수량, 기온, 일조량, 일사량, 습도 등)는 계절성 변화와 날씨 변동의 영향을 강하게 받습니다. 이를 개선하기 위해 누적합 피처를 생성하여, 장기적인 기상 변화가 가격에 미치는 영향을 반영했습니다.
- **수확 시기 반영 피처**
  - 농산물의 수확 시기는 가격 변동의 중요한 요인입니다. 따라서 품목별로 수확 가능한 월을 기준으로 수확 시기의 특성을 나타내는 피처를 추가했습니다. 이는 수확 시기에 가격 급등/급락을 설명하는 데 도움이 됩니다.

---

### 2. **경제적 요인 (Economic Factors)**
- **유가 데이터**
  - 유가는 농산물의 재배, 운송, 유통 비용에 영향을 미치는 중요한 경제 지표입니다. 유가 데이터를 추가하여, 에너지 가격 상승이 농산물 가격에 미치는 영향을 반영했습니다. 특히 유가 상승 시 농산물 가격도 상승하는 경향이 있어, 이를 학습 데이터에 포함시켜 비선형적 관계를 학습할 수 있도록 했습니다.
- **총반입량 평균 피처**
  - 농산물 가격은 공급량에 따라 변동됩니다. 평균 가격과 총반입량의 평균값을 계산하여, 시장의 공급 측면을 반영하는 피처를 추가했습니다. 이는 수급 불균형 상황에서 가격 변동을 예측하는 데 도움이 됩니다.


---

### 3. **시계열적 요인 (Temporal Factors)**
- **계절성 반영 (sin/cos 변환)**
  - 연도, 월, 순(상순, 중순, 하순) 데이터를 주기적인 특성에 맞춰 sin/cos 변환하여 계절적 패턴을 학습할 수 있도록 했습니다. 이는 김장철, 수확기와 같은 반복적인 계절성 변화를 모델이 인식하게 합니다.
- **가격 차이 및 누적 변화량 계산**
  - 과거 8개월 간의 가격 차이와 누적 변화량을 계산해, 시간에 따른 가격 변동성을 반영했습니다. 이는 최근의 가격 급등/급락 패턴을 학습하는 데 도움을 줍니다.
- **Target Encoding 및 표준편차 인코딩**
  - 가격의 평균과 표준편차를 계산하여 인코딩 피처를 생성했습니다. 이를 통해 계절별, 수확기별 가격 패턴의 차이를 반영했습니다.
- **Low-Pass Filter 적용**
  - 가격 데이터에서 단기적인 노이즈를 제거하고 장기적인 추세를 추출하기 위해 Low-Pass Filter를 사용했습니다. 이를 통해 계절성 패턴과 장기적인 가격 변동을 더 정확하게 반영할 수 있었습니다.
---

### 4. **외부적 요인 (External Factors)**
- **시장 개입 확률 계산**
  - 급격한 가격 변화는 정부의 시장 개입을 유발할 수 있습니다. 가격 변화율, 이동 평균, 변동성 등을 기반으로 개입 확률을 계산하여, 시장 안정화 조치를 모델이 예측할 수 있도록 했습니다.
- **정부 개입 분석 및 피처 생성**
  - 급격한 가격 변동이 발생할 때 정부의 수매 및 방출 정책이 영향을 미칩니다. 이를 반영하기 위해 가격 변화율, 이동 평균, 변동성을 기반으로 시장 개입 확률 피처를 추가했습니다. 이를 통해 예측 모델이 외부 개입 상황을 고려할 수 있게 했습니다.
- **기후 개입 확률 피처 생성**
  - 기후 요인(기온, 강수량 등)의 평균과의 편차를 계산해, 이상 기후 상황에서의 개입 확률을 추정했습니다. 이는 폭염, 태풍과 같은 극단적 기상 현상이 가격에 미치는 영향을 반영합니다.

---

### **핵심 요약**
- **자연적 요인:** 누적 기상 데이터.
- **경제적 요인:** 유가 및 총반입량을 통해 경제적 요인을 반영.
- **시계열적 요인:** Low-Pass Filter 등을 활용해 환경적 영향을 반영 등을 통해 시간 흐름에 따른 변동성을 학습.
- **외부적 요인:** 정부 개입 정보를 반영해, 외부 요인에 따른 가격 변동성을 예측.



## 🎈 Modeling
```
**Time-Series**
 - DLinear(본선 사용)  
 - NLinear  
 - LSTFLinear  
 - SegRNN  
 - ITransformer  
 - Temporal Fusion Transformer  
 - PatchTST  

**Machine Learning**
 - LGBM(본선 사용)  
 - XGBoost  
 - CatBoost  
 - ExtraTree  
 - RandomForest  
 - Ridge  

**Time-ML**
 - LGBM기반 DLinear 예측값 활용  

```

