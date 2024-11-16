# Price_Forecasting
# LG-Demand_Forecasting [(Link)](https://dacon.io/competitions/official/236381/leaderboard)
# 인기상 [(Link)](https://dacon.io/competitions/official/236417/leaderboard)

## 🏆 Result
## **Public score 1st** 0.08605 | **Private score 1st** 0.07665 | 최종 1등

<img width="100%" src="https://github.com/user-attachments/assets/51ed525a-5ab5-439a-8086-cc0ac25b2eca"/>
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


좋습니다. 내용을 더 충실하게 보강하고, **개조식 형식**으로 더 완성도 높게 다시 정리하겠습니다.

---

알겠습니다. **장점만 포함**하여 개조식으로 다시 정리하겠습니다.

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

알겠습니다. 요청하신 대로 **"사용 이유"**로 수정하여 다시 정리하겠습니다.

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

확실히 이해했습니다. **2~5번 항목을 하나로 묶고**, 소분 항목으로 **-1, -2, -3** 형식으로 정리하겠습니다.

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

## **3. 파이프라인의 강점과 기대효과**
- **파이프라인 강점**:
  - N 시점의 예측 결과가 '심각' 구간에 도달하기 전에, **N-1 시점에서 조정**하여 시장 변동성 완화.
  - 실시간 예측과 조정을 통해 **가격 안정화**에 효과적.
- **기대 효과**:
  - **정부**: 과잉 생산과 가격 변동으로 인한 **보관 및 폐기 비용 절감**.
  - **농민**: 최소 가격 보장을 통해 **안정적인 소득 확보** 및 지속 가능한 생산 가능.
  - **소비자**: 가격 변동성이 줄어, **안정된 가격으로 농산물 구매 가능**.

---

## **4. 시스템의 차별화 요소**
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

---






ㅁㄴㅇㅁㄴㅇ

ㅁㄴㅇㅁㄴㅇ




# ```본선```   


본선 진출 이후, 예측 모델을 활용한 **‘AI 기반 수매-방출 의사결정 시스템’**을 개발했습니다. 이 시스템은 농산물 시장의 가격 안정화를 목표로 설계되었으며, 농민, 소비자, 정부 모두에게 이익이 되는 솔루션을 제공합니다.

#### **1. 문제 정의 및 해결 방향**
- 농산물 가격이 **낮을 때는 농민**, **높을 때는 소비자**가 문제를 겪습니다:
- 가격이 **하락**하면 농민의 수익이 감소하고, 가격이 **상승**하면 소비자의 부담이 커집니다.
- 이를 해결하기 위해, AI 기반 시스템은 **예측된 가격**을 바탕으로 **수매와 방출**을 결정해 시장 가격을 조정하고 안정화합니다.

#### **2. N-1 시점에서의 의사결정**
- **N 시점**에서 가격이 심각 구간에 진입할 경우, 심리적 불안감으로 인해 시장의 변동성이 커질 수 있습니다.
- 따라서, **N-1 시점**에서 미리 조정 조치를 취해 심각 구간에 도달하기 전에 가격을 안정화합니다.
- 이 방식은 **약간의 예측 오차**가 있더라도, 심각 구간 진입 전 조정이 이루어지므로 **시장 충격을 최소화**할 수 있습니다.

#### **3. 조정 강도 및 최적화 프로세스**
- **조정 강도**는 예측된 가격의 심각성 정도에 따라 다르게 설정됩니다:
- 가격이 심각 기준에서 크게 벗어나면, 더 강한 조정이 필요합니다. 예를 들어, 예측된 가격이 **상한선을 초과**하면 방출량을 늘리고, **하한선 이하**이면 수매량을 늘립니다.
- **TIME ML 모델**을 통해 반복적인 최적화 과정이 수행됩니다:
- N-1 시점의 조정된 가격을 기반으로 새로운 예측을 수행하고, 이를 반복해 **최적의 수매와 방출량**을 결정합니다.
- TIME ML은 자기회귀 구조로 설계되어, 조정된 예측값을 학습에 반영해 **시간 종속성을 더 잘 학습**할 수 있습니다.

#### **4. 목적 함수와 비용 고려**
- 시스템의 목적 함수는 가격이 **‘심각’ 구간에서 벗어나도록** 하는 것입니다.
- 또한, **수매와 방출 비용**, **보관 및 폐기 비용**을 최소화하여 **비용 효율적인 의사결정**을 목표로 합니다.

#### **5. 전체 파이프라인의 강점과 기대효과**
- 예측된 가격이 심각 구간에 들어오기 전에 미리 대응할 수 있어, **시장 안정화**에 효과적입니다.
- **기대 효과:**
- **정부:** 적절한 수매와 방출을 통해 시장 안정화 및 **비용 절감** 가능.
- **농민:** 최소 가격 보장을 통해 소득이 안정되며, 손해를 줄이고 지속 가능한 생산 활동을 이어갈 수 있습니다.
- **소비자:** 가격 변동성이 줄어 안정된 가격으로 농산물을 구매할 수 있어, 생활비 부담이 완화됩니다.

#### **6. 시스템의 차별화**
- 기존의 수매-방출 시스템은 정해진 규칙에 따라 작동하지만, AI 기반 시스템은 **실시간 가격 예측**과 시장 데이터를 반영해 더 **유연하고 민첩한 대응**이 가능합니다.
- 이로 인해, 시장 변화에 맞춘 빠르고 정확한 의사결정이 가능해져, 더 효과적인 가격 안정화를 실현할 수 있습니다.

---

### **요약:**

- **문제 정의:** 농민과 소비자의 문제 해결을 위해 AI 기반 수매-방출 시스템 설계.
- **N-1 시점에서의 의사결정:** 심리적 불안감을 줄이고 안정적인 시장 조정 가능.
- **조정 강도와 최적화:** TIME ML 모델을 통한 반복적인 최적화와 자기회귀 구조 활용.
- **목적 함수 및 비용 고려:** 가격 안정화와 비용 최소화를 동시에 달성.
- **기대 효과:** 정부, 농민, 소비자 모두에게 이익이 되는 솔루션.
- **차별화:** 실시간 예측과 민첩한 대응을 통해 기존 시스템 대비 우수한 성능 제공.

#### **그 밖의 모델 활용 제안: 두 가지 전략**

1. **선물 거래 시스템 (Futures Trading System)**
   - **이점:** 소비자는 미리 정해진 가격에 농산물을 구매해 **가격 급등 리스크**를 줄일 수 있고, 농가는 미리 판매 계약을 통해 **안정적인 수익**을 확보할 수 있습니다.
   - **효과:** 가격 변동성을 줄이고, 수급 예측 가능성을 높여 농산물 시장의 안정성을 강화합니다.

2. **맞춤형 농산물 보험 (Customized Agricultural Insurance)**
   - **차별점:** AI 예측 모델을 기반으로 가격 변동을 분석하고, **농가나 유통업체의 특성**에 맞춰 보장 범위를 설계합니다. 이는 일반적인 표준 보험보다 더 정확하게 리스크를 관리할 수 있습니다.
   - **효과:** 가격 하락 시 손실을 보상해 **농가의 재정적 안정성**을 높이고, 장기적인 농업 생산 활동을 지원해 농업 생태계의 안정화에 기여합니다.
  
... 





## **2차 예선: 외부 데이터 활용과 모델 경량화**

### **주요 접근 사항**
1. **외부 데이터 활용 분석**
   - **자연적 요인 분석**:
     - 기온, 강수량, 일조량 등의 **환경 변수**를 분석하여 농산물 가격의 변동 요인을 파악.
     - **누적값 피처** 생성: 계절성과 변동성을 반영하기 위해 누적값 피처를 추가하여 장기적인 영향을 반영.
     - **극단적 기후 분석**: 태풍, 폭염 등의 **이상 기후 이벤트**가 가격에 미치는 영향 분석 및 변수 추가.
   - **경제적 요인 분석**:
     - **유가, 환율, 총반입량** 등의 경제 지표 분석.
     - **Granger 인과 검정**을 통해, 유가와 총반입량이 가격의 중요한 **선행 지표**임을 확인.
   - **시계열적 요인 분석**:
     - **Low-Pass Filter**를 사용해 이동 평균보다 더 정확하게 **추세 성분**을 추출.
     - **sin, cos 변환**으로 계절성 및 주기성을 반영한 피처 생성.
   - **외부적 요인 분석**:
     - 정부의 수매 및 방출 정책, 보조금 지급 등의 **정부 개입 변수** 추가.
     - **가격 급변 시 개입 확률 변수** 생성하여, 외부 개입 상황을 더 잘 반영.

2. **모델 경량화 및 선택**
   - **DLinear 모델**:
     - 데이터를 추세와 계절성으로 분해하여, 빠른 학습과 추론 가능.
     - **Attention 메커니즘** 추가로 중요한 계절 이벤트(김장철 등)를 반영.
   - **LGBM 모델**:
     - **leaf-wise 성장 방식**으로 비선형적 관계와 급격한 변동을 잘 포착.
     - 다양한 파생 변수(누적 강수량, 개입 확률 등)를 효과적으로 학습.
   - **결정**: 빠른 추론과 높은 정확도를 위해, **DLinear**와 **LGBM** 모델로 최종 선택.

3. **TIME ML 전략 개발**
   - **구조**: 시계열 모델의 예측값을 머신러닝 모델의 피처로 사용.
   - **장점**:
     - 단기 예측에서는 시계열 모델의 패턴 학습 능력을 활용.
     - 장기 예측에서는 머신러닝 모델의 비선형 관계 학습 능력을 활용.
   - **특징**: 자기회귀 구조로 **시간 종속성**을 효과적으로 학습하고, 예측 성능을 극대화.

---

## **본선: AI 기반 수매-방출 의사결정 시스템**

### **주요 내용 및 전략**
1. **AI 기반 의사결정 시스템 설계**
   - **목표**: 농산물 가격의 급락 시 농민 보호, 급등 시 소비자 보호를 위한 **시장 안정화 시스템** 개발.
   - **접근 방법**: 예측된 가격을 바탕으로, **수매 및 방출 의사결정**을 통해 시장 가격 조정.

2. **N-1 시점에서의 사전 조정**
   - **조정 방식**: 가격이 심각 구간에 진입하기 전에 **N-1 시점**에서 미리 조정 수행.
   - **효과**: 예측 오차를 줄이고, 시장 변동성 및 심리적 불안감을 최소화.

3. **조정 강도 결정 및 최적화**
   - 예측된 가격의 **심각성 정도**에 따라 조정 강도 설정:
     - 상한선 초과 시 **방출량 증가**.
     - 하한선 이하 시 **수매량 증가**.
   - **TIME ML 모델**을 통해 반복적인 최적화 수행.

4. **비용 고려 및 목적 함수 설계**
   - **비용 최소화 목표**: 수매 및 방출 비용, 보관 비용, 폐기 비용 최소화.
   - 가격이 **‘심각’ 구간에서 벗어나도록** 하는 목적 함수 설정.

5. **기대 효과 및 차별성**
   - **정부**: 효율적인 수매/방출로 비용 절감 및 시장 안정화 가능.
   - **농민**: 최소 가격 보장으로 안정적인 소득 확보.
   - **소비자**: 가격 변동성이 줄어 안정된 가격으로 구매 가능.
   - **차별성**: 실시간 예측과 유연한 의사결정을 통해 기존 시스템보다 **더 빠르고 정확한 대응** 가능.

---

### **추가 제안 사항**
1. **선물 거래 시스템 도입**
   - **이점**: 소비자는 가격 리스크를 줄이고, 농가는 안정적인 수익을 확보할 수 있음.
   - **기대 효과**: 가격 변동성 감소 및 시장 안정성 강화.

2. **맞춤형 농산물 보험 제공**
   - **특징**: AI 예측 모델 기반으로 **맞춤형 보장 설계** 가능.
   - **효과**: 가격 하락 시 손실 보상, 농가의 재정 안정성 강화.

---


## 🔥**대회 접근법 및 배운점**

# ```1차 예선```

1. 농산물 데이터를 통합하여 모델링할지, 아니면 각 품목별로 개별 모델링을 할지  
2. 동일한 품목명 아래 **여러 품종명(예: 감자의 수미, 대지 품종 등)**이 존재할 때, 이 데이터를 어떻게 활용할지  
3. 예측에 사용할 모델로 시계열 모델과 머신러닝 모델 중 어느 것이 더 적합할지  



1 : 먼저, 모든 농산물을 하나의 통합 모델로 학습할지, 아니면 품목별로 개별 모델링할지를 결정하기 위해 고민했습니다. 이때 각 농산물이 유사한 추세, 계절성, 분포를 보인다면 통합 모델링이 적합할 수 있습니다. 통합 모델은 데이터를 더 많이 활용할 수 있어 일반화 성능이 높아지는 장점이 있기 때문입니다. 그러나, 실제 EDA를 통해 분석한 결과, 농산물마다 고유한 계절성 패턴과 가격 변동 특성이 다르게 나타났습니다. 예를 들어, 배추는 김장철에 가격이 급등하는 반면, 감자는 여름철 기후에 더 민감한 반응을 보였습니다. 이러한 품목별 특성의 차이를 무시하고 통합 모델링을 진행할 경우, 고유한 패턴을 제대로 반영하지 못해 오히려 예측의 정확도가 떨어지고 노이즈가 증가할 가능성이 높습니다. 특히, 한 품목에서의 계절적 급등 현상이 다른 품목에는 존재하지 않기 때문에, 이러한 특이점이 통합 모델에서는 혼란을 줄 수 있습니다.  

따라서, 시간에 따른 가격 변동 EDA 결과를 바탕으로, 농산물의 특성을 반영하기 위해 품목별 모델링을 선택했습니다. 이는 각 품목의 고유한 패턴을 효과적으로 학습하고, 예측 성능을 높이는 데 도움이 되었습니다.  


2 : 동일 품목명이라도 여러 품종이 존재(예: 감자의 '수미', '대지' 품종)하는 경우, 이 데이터를 학습에 어떻게 활용할지에 대해 고민했습니다. 먼저, 동일 품목 내에서는 유사한 추세, 계절성, 분포를 보일 것이라는 가정을 세웠습니다. 그 이유는, 같은 품목에 속하는 품종들은 비슷한 재배 환경과 수확 시기를 가지기 때문에, 가격 변동 패턴에서도 공통적인 특성이 나타날 가능성이 높기 때문입니다. 예를 들어, '수미'와 '대지' 감자는 재배 및 유통 과정에서 계절성 영향을 유사하게 받습니다.   

이 가정을 검증하기 위해, UMAP 분석과 시간에 따른 가격 변동 EDA를 수행했습니다. UMAP 분석을 통해 품종별 가격 변동 패턴을 시각화한 결과, 동일 품목 내의 품종들은 비슷한 군집을 형성하는 경향이 있음을 확인했습니다. 또한, 시간에 따른 EDA에서도 동일 품목 내 품종들 간의 계절성 패턴과 가격 변동이 유사하게 나타났습니다.  

따라서, 동일 품목명 내의 다른 품종명들은 하나의 모델에서 다변량 데이터로 학습시키는 것이 적합하다고 판단했습니다. 이를 통해 모델은 품종별로 미세한 차이를 반영하면서도, 공통적인 패턴을 학습할 수 있도록 하였고, 이 접근법을 통해 데이터의 양을 증가시켜 모델의 일반화 성능을 높이도록 의도했습니다.   

3 : 모델링 접근에서는 머신러닝 모델과 시계열 딥러닝 모델 중 어떤 것을 사용할지 고민했습니다. 먼저, 머신러닝 모델을 선택한 이유는 비선형 관계 학습과 다양한 파생 변수의 활용이 가능하기 때문입니다. 농산물 가격은 기후, 경제적 요인, 외부 개입 등 다양한 비선형적 영향을 받기 때문에, 이를 반영하기 위해 머신러닝 모델들을 사용했습니다. 반면, 시계열 딥러닝 모델은 최근의 시간적 흐름과 계절성 패턴을 학습하기에 유리해, 단기 예측의 정확도를 높이기 위해 사용했습니다.  

머신러닝 모델에서는 시계열 데이터를 다루는 방식에 대해 추가적인 고민이 필요했습니다. 이를 위해 3가지 방법을 고려했습니다:  

1) 시계열 데이터를 시점에 맞게 입력하는 방법:  

 - 각 시점의 데이터를 그대로 사용해 현재 시점의 예측에 활용했습니다. 이 방법은 간단하면서도 시점별 독립적인 피처를 반영할 수 있는 장점이 있습니다.  

2) 이전 시점 데이터를 Transpose하여 피처로 활용하는 방법:  

 - 과거 데이터를 Transpose(전치)하여 현재 시점의 피처로 추가했습니다. 이를 통해 시간적 흐름과 과거 패턴을 반영할 수 있으며, 시계열 데이터를 피처 형식으로 변환해 머신러닝 모델의 학습 성능을 높일 수 있었습니다.  

3) 다변량 형식으로 다른 품종명들의 가격을 포함시키는 방법:  

 - 동일 품목 내의 다른 품종명 데이터를 다변량 형식으로 함께 입력했습니다. 이를 통해 다양한 품종 간의 상호 관계를 학습할 수 있었으며, 특히 비슷한 계절성 패턴을 가진 품종들의 데이터를 함께 사용함으로써 모델의 예측 정확도를 높였습니다.  

각 방법의 장점은 다음과 같습니다:  

1. 첫 번째 방법은 시점별 독립성을 보장하면서, 간단한 구조로 빠른 학습이 가능합니다.  
2. 두 번째 방법은 시간적 패턴을 반영해, 과거의 영향을 효과적으로 학습할 수 있습니다.  
3. 세 번째 방법은 다변량 데이터의 장점을 활용해, 품종 간의 상호작용과 공통적인 패턴을 학습할 수 있어 예측 성능을 향상시켰습니다.  

따라서 머신러닝 모델은 3번 방법으로 학습시켰습니다.   

결과적으로, 시계열 딥러닝 모델은 최근의 시간적 흐름과 계절성을 학습해 단기 예측에 강점을 보였고, 머신러닝 모델은 다양한 비선형 관계를 반영해 장기 예측에서 우수한 성능을 나타냈습니다. 이를 바탕으로, 단기 예측에는 시계열 모델의 가중치를 더 두고, 장기 예측에는 머신러닝 모델의 가중치를 두는 앙상블 전략을 구축하여 최종 예측 성능을 극대화했습니다.  


### **농산물 가격 예측 모델링 접근 요약**

---

#### **3가지 주요 고민과 결정 사항**

1. **농산물 통합 모델링 vs. 품목별 모델링**
   - **통합 모델**은 데이터를 더 많이 활용할 수 있어 일반화 성능이 높아질 가능성이 있지만, **EDA 결과** 품목별로 계절성 패턴과 가격 변동 특성이 크게 달랐습니다.
   - 예를 들어, **배추**는 김장철에 급등하고, **감자**는 여름철 기후에 더 민감했습니다.
   - 통합 모델링 시 이러한 차이를 반영하지 못해 오히려 **노이즈**가 될 수 있다고 판단했습니다.
   - **결정:** 각 품목별로 모델링하여 고유한 패턴을 효과적으로 학습하고 예측 성능을 높였습니다.

2. **동일 품목명 내 여러 품종명 데이터 활용**
   - 동일 품목 내 여러 품종(예: 감자의 '수미', '대지')이 존재하며, 유사한 계절성, 추세, 분포를 보일 것이라 가정했습니다.
   - **UMAP 분석**과 **가격 변동 EDA** 결과, 동일 품목 내 품종들이 비슷한 패턴을 나타냄을 확인했습니다.
   - **결정:** 동일 품목명 내 품종 데이터를 **다변량 형식**으로 하나의 모델에 통합 학습하여, 데이터 양을 늘리고 모델의 일반화 성능을 높였습니다.

3. **머신러닝 모델 vs. 시계열 딥러닝 모델 선택**
   - **머신러닝 모델:** 비선형 관계 학습과 다양한 파생 변수(환경 변수, 경제적 변수 등) 반영에 강점이 있어 **장기 예측**에 유리했습니다.
   - **시계열 딥러닝 모델:** 최근의 시간적 흐름과 계절성 패턴 학습에 유리해 **단기 예측**에서 뛰어난 성능을 보였습니다.

---

#### **머신러닝 모델의 시계열 데이터 처리 방법**
- **3가지 접근법:**
  1. **시점별 독립 입력:** 현재 시점의 데이터를 그대로 입력해 간단한 구조로 빠른 학습 가능.
  2. **Transpose 방식:** 과거 데이터를 전치(Transpose)해 현재 시점의 피처로 사용, 시간적 흐름 반영 가능.
  3. **다변량 입력:** 동일 품목 내 다른 품종명 데이터를 다변량 형식으로 함께 입력해, 상호작용과 공통 패턴 학습 가능.

- **결정:** 머신러닝 모델은 **다변량 입력 방식**을 사용해 품종 간 관계를 반영하고 예측 성능을 높였습니다.

---

#### **최종 앙상블 전략**
- **시계열 딥러닝 모델:** 단기 예측에 강점이 있어, 단기 예측에서는 가중치를 더 부여했습니다.
- **머신러닝 모델:** 다양한 비선형 관계를 반영해 장기 예측에 강점을 보여, 장기 예측에서는 가중치를 더 부여했습니다.
- **결과:** 단기와 장기 예측의 장점을 결합한 앙상블 전략을 통해, 최종 예측 성능을 극대화했습니다.

---

### **핵심 요약**
- **품목별 모델링:** 각 품목의 고유한 패턴을 반영해 예측 성능 향상.
- **다변량 데이터 활용:** 품종 간 유사성을 바탕으로 데이터 양 증가 및 일반화 성능 강화.
- **앙상블 전략:** 단기 예측에는 시계열 모델, 장기 예측에는 머신러닝 모델의 가중치를 부여해 최적화.



# ```2차 예선``` 
2차 예선은 1차 예선과 다르게 외부 데이터 활용이 가능했습니다. 
따라서 2차 예선은 1차 예선을 기반으로 하되, 외부 데이터 활용 및 본선 발표를 위한 “현실 세계에서의 모델 활용”의 관점으로 포커싱했습니다.

먼저, 농산물 가격은 “자연적”, “경제적”, “시계열적”, “외부적”의 요인이 복합적으로 작용해 가격을 결정한다는 사실에 기반해 이 4가지의 카테고리를 중점으로 EDA 및 데이터 활용을 진행했습니다.

1. 자연적 요인
 - 환경 변수: 농산물 가격에 영향을 미치는 주요 자연적 요인으로 강수량, 기온, 일조량, 습도 등을 분석했습니다. 이는 작물의 생육과 생산량에 직접적인 영향을 주기 때문에 중요한 변수로 고려했습니다.  

 - 누적값 피처 생성: 자연적 요인의 데이터는 계절성과 날씨의 변동성으로 인해 노이즈가 많았습니다. 이를 개선하기 위해 누적값 피처를 생성해, 장기적인 환경 영향을 반영할 수 있도록 했습니다.  

 - 태풍과 같은 이상 기후 분석: 태풍이나 폭염 같은 극단적인 기상 이벤트가 농산물 가격에 미치는 영향을 분석하고, 이를 반영할 수 있는 변수를 추가했습니다.  

2. 경제적 요인
 - 경제 지표 분석: 농산물 가격에 영향을 줄 수 있는 유가, 환율, 총반입량 등의 경제적 지표를. 분석했습니다. 이 지표들은 농산물의 재배, 유통 비용과 밀접한 관계가 있어 가격 변동에 중요한 역할을 합니다.  

 - Granger 인과 검정: 경제적 요인들이 농산물 가격에 선행하는 지표인지 검증하기 위해. Granger 인과 검정을 실시했습니다. 그 결과, 유가와 총반입량이 농산물 가격의 중요한 선행 지표임을 확인했습니다.  
  
 - 경제적 외부 변수 추가: 주요 경제 지표들을 모델 학습에 반영해, 가격 예측의 정확도를 높였습니다.  

3. 시계열적 요인
 - 추세와 계절성 분석: 농산물 가격 데이터의 **추세(Trend)**와 **계절성(Seasonality)**을 분석했습니다. 이를 통해, 가격이 계절에 따라 반복적으로 변동하는 패턴을 확인했습니다.  

 - Low-Pass Filter 사용: 이동 평균 대신 Low-Pass Filter를 사용해 추세 성분을 추출했습니다. 이동 평균은 노이즈를 포함할 수 있어, 더 정확한 추세 분석을 위해 Low-Pass Filter를 선택했습니다.  

 - sin, cos 변환: 계절성과 주기성을 효과적으로 반영하기 위해 sin, cos 변환을 사용해 시계열 데이터를 주기적 패턴으로 변환했습니다.  

4. 외부적 요인
 - 정부 개입 분석: 수매, 방출, 보조금 지급 등의 정부 개입이 농산물 가격에 미치는 영향을 분석했습니다. 이는 인위적인 가격 조정으로, AI 모델이 학습하기 어려운 요인입니다.  

 - 개입 확률 피처 생성: 가격 급락이나 급등 시기에 시장 개입 확률 변수를 추가했습니다. 가격. 변화율, 변동성 등을 기준으로 개입 확률을 계산하여 모델이 외부 개입 상황을 더 잘 반영할 수 있도록 했습니다.  

 - 실제 개입 데이터 검증: EDA 과정에서 가격 급변 시기를 분석한 결과, 해당 시기에 실제로 정부의 물량 방출이 있었음을 확인했습니다. 이를 통해 외부 개입 변수가 효과적으로 작동함을 검증했습니다.  



본선 진출을 위해, 정확도뿐만 아니라 추론 시간을 고려해야 했습니다. 따라서, 1차 예선에서 사용했던 Extra Tree, Random Forest, SegRNN, iTransformer 같은 무거운 모델 대신, 더 가볍고 빠른 모델인 LGBM과 DLinear를 선택했습니다.  

먼저,  DLinear는 데이터를 추세(Trend)와 계절성(Seasonality)으로 분해하여 학습하는 선형 모델로, 학습과 추론 모두 빠르게 수행할 수 있습니다. 또한, 계절성 패턴을 더욱 강화하기 위해 Attention 메커니즘을 추가했습니다. 이를 통해, 모델이 계절적 이벤트에 더 집중하여 김장철과 같은 중요한 패턴을 학습할 수 있었으며, 정확도를 높이는 데 기여했습니다. 또한 DLinear 모델은 선형 구조로 인해 추론 시간이 매우 짧습니다.   

다음으로, LGBM 모델을 선택한 이유는 leaf-wise 성장 방식과 Histogram 기반 학습 때문입니다. LGBM은 농산물 가격은 기후, 수급, 외부 개입 등 다양한 비선형적 요인에 의해 크게 영향을 받습니다. LGBM의 leaf-wise 성장 방식은 이러한 비선형적 변화를 포착하는 데 유리합니다. 이 방식은 농산물 가격 데이터에서 나타나는 급격한 변동과 예외적인 패턴을 세밀하게 분할할 수 있어, 특정 이벤트나 계절성 변화에도 빠르게 적응할 수 있는 장점이 있습니다. 또한, Histogram 기반 학습은 계산량을 줄여 추론 시간을 크게 단축시킬 수 있어, 본선 평가 기준에 적합했습니다. 이와 함께, 다양한 파생 변수(강수량 누적값, 개입 확률 등)를 효과적으로 반영할 수 있어, 농산물 가격의 복잡한 패턴을 잘 학습할 수 있었습니다.   

그리고 이번 Task에서 특별하게 새로운 모델링 전략인 TIME ML을 개발했습니다. TIME ML은 기존의 시계열 모델과 머신러닝 모델의 한계를 극복하기 위해 설계된 혼합 모델입니다. 이 모델은 시계열 모델의 예측값을 머신러닝 모델의 입력으로 사용하는 방식으로, 시계열 데이터의 시간적 흐름과 장기 추세를 더 잘 반영할 수 있도록 구성되었습니다.  

먼저, 시계열 딥러닝 모델은 최근의 시간적 패턴을 학습해 단기 예측에서 뛰어난 성능을 보입니다. 반면, 머신러닝 모델은 다양한 파생 변수와 비선형 관계를 학습하는 데 강점이 있어, 장기 예측에서 더 좋은 성과를 냅니다. TIME ML은 이 두 모델의 장점을 결합해, 단기 예측에서는 시계열 모델의 패턴 학습 능력을, 장기 예측에서는 머신러닝 모델의 관계 학습 능력을 동시에 활용합니다.  

이 전략의 핵심은 자기회귀적 구조를 기반으로 한다는 점입니다. 기존의 머신러닝 모델들은 과거 데이터를 독립적인 피처로 사용하지만, TIME ML은 시계열 모델의 예측값을 직접 피처로 사용하여 시간의 흐름과 타겟 변수의 변화를 지속적으로 반영합니다. 이를 통해 시간 종속성을 더 잘 학습할 수 있고, 예측 정확도 또한 크게 향상되었습니다.  

결과적으로, TIME ML 전략은 단기와 장기 예측 모두에서 우수한 성능을 보였으며, 기존의 단일 모델 대비 앙상블 효과를 통해 최종 예측 성능을 극대화할 수 있었습니다. 이 접근 방식은 농산물 가격 예측의 복잡한 특성을 더 잘 반영할 수 있도록 설계된 새로운 모델링 전략입니다.  

### **요약: 2차 예선 및 본선 모델 접근 방식**

---

#### **1. 2차 예선 전략: 외부 데이터 활용 및 현실 세계 적용**

- **외부 데이터 활용:** 1차 예선과 달리 2차 예선에서는 외부 데이터를 사용할 수 있었습니다. 이를 바탕으로 **현실적인 모델 활용**을 염두에 두고 접근했습니다.
- **4가지 요인 기반 분석:**
  - **자연적 요인:** 기온, 강수량, 일조량 등 **환경 변수**를 분석하고, **누적값 피처**로 장기적인 환경 영향을 반영했습니다. 태풍 같은 **이상 기후**도 분석에 포함했습니다.
  - **경제적 요인:** 유가, 환율, 총반입량 같은 **경제 지표**를 분석하고, Granger 인과 검정으로 **선행 지표**를 파악했습니다. 주요 경제 변수는 모델 학습에 반영했습니다.
  - **시계열적 요인:** 가격의 **추세와 계절성**을 분석했으며, **Low-Pass Filter**로 추세를 추출하고, **sin, cos 변환**으로 주기성을 반영했습니다.
  - **외부적 요인:** 정부 개입(수매, 방출 등)을 분석하고, **개입 확률 변수**를 추가해, 가격 변동성에 대응할 수 있도록 했습니다.

#### **2. 모델 선택: 경량화와 추론 시간 단축**

- **DLinear 모델:**
  - **추세와 계절성**을 분해해 학습하는 선형 모델로, 학습과 추론 시간이 매우 짧아 효율적입니다.
  - **Attention 메커니즘**을 추가해 계절적 이벤트(김장철 등)에 더 집중하도록 설계했습니다.
- **LGBM 모델:**
  - **leaf-wise 성장 방식**으로 데이터를 세밀하게 분할해, 비선형적 관계와 급격한 변동을 잘 포착할 수 있습니다.
  - **Histogram 기반 학습**으로 계산량을 줄이고 추론 시간을 단축했습니다.
  - 다양한 파생 변수(강수량 누적값, 개입 확률 등)를 효과적으로 반영해, 농산물 가격의 복잡한 패턴을 잘 학습했습니다.

#### **3. 새로운 모델링 전략: TIME ML**

- **TIME ML 개발:** 기존의 시계열 모델과 머신러닝 모델의 한계를 극복하기 위해 설계한 혼합 모델입니다.
  - **구조:** 시계열 모델의 예측값을 머신러닝 모델의 입력 피처로 사용해, **시간적 흐름과 장기 추세**를 반영합니다.
- **모델 결합:** 
  - 시계열 모델은 **단기 예측**에서, 머신러닝 모델은 **장기 예측**에서 뛰어난 성능을 보입니다.
  - TIME ML은 두 모델의 장점을 결합해, 단기 예측에서는 시계열 모델의 패턴 학습 능력을, 장기 예측에서는 머신러닝 모델의 관계 학습 능력을 활용했습니다.
- **자기회귀적 구조:** 
  - 기존의 머신러닝 모델과 달리, TIME ML은 시계열 모델의 예측값을 직접 피처로 사용해 **시간 종속성**을 더 잘 학습할 수 있습니다.
- **성과:** 
  - TIME ML 전략은 단기와 장기 예측 모두에서 **우수한 성능**을 보였고, 앙상블 효과를 통해 최종 예측 성능을 극대화했습니다.

#### **4. 본선 전략: 정확도와 효율성의 균형**
- 본선에서는 **정확도뿐만 아니라 추론 시간도 평가**의 중요한 요소였습니다. 따라서 1차 예선에서 사용했던 무거운 모델 대신, **DLinear**와 **LGBM** 같은 경량 모델을 선택했습니다.
- 이 두 모델은 **추론 속도**와 **정확도**를 동시에 확보할 수 있어 본선 평가 기준에 적합했습니다.

---

### **결론: 모델링 전략의 차별성과 효과**
- **외부 데이터 활용:** 자연적, 경제적, 시계열적, 외부적 요인을 모두 반영해 더 정확한 예측 가능.
- **TIME ML 전략:** 단기와 장기 예측 모두에서 뛰어난 성능을 보이며, 기존 모델의 한계를 극복한 혼합 모델.
- **경량화 모델 선택:** DLinear와 LGBM 모델로 빠른 추론 시간 확보와 높은 예측 성능 달성.
- **최종 성과:** 정확도와 추론 시간을 모두 고려한 모델링 전략으로, 본선에서 우수한 성과를 거두었습니다.



# ```본선```   


본선 진출 이후, 예측 모델을 활용한 **‘AI 기반 수매-방출 의사결정 시스템’**을 개발했습니다. 이 시스템은 농산물 시장의 가격 안정화를 목표로 설계되었으며, 농민, 소비자, 정부 모두에게 이익이 되는 솔루션을 제공합니다.

#### **1. 문제 정의 및 해결 방향**
- 농산물 가격이 **낮을 때는 농민**, **높을 때는 소비자**가 문제를 겪습니다:
- 가격이 **하락**하면 농민의 수익이 감소하고, 가격이 **상승**하면 소비자의 부담이 커집니다.
- 이를 해결하기 위해, AI 기반 시스템은 **예측된 가격**을 바탕으로 **수매와 방출**을 결정해 시장 가격을 조정하고 안정화합니다.

#### **2. N-1 시점에서의 의사결정**
- **N 시점**에서 가격이 심각 구간에 진입할 경우, 심리적 불안감으로 인해 시장의 변동성이 커질 수 있습니다.
- 따라서, **N-1 시점**에서 미리 조정 조치를 취해 심각 구간에 도달하기 전에 가격을 안정화합니다.
- 이 방식은 **약간의 예측 오차**가 있더라도, 심각 구간 진입 전 조정이 이루어지므로 **시장 충격을 최소화**할 수 있습니다.

#### **3. 조정 강도 및 최적화 프로세스**
- **조정 강도**는 예측된 가격의 심각성 정도에 따라 다르게 설정됩니다:
- 가격이 심각 기준에서 크게 벗어나면, 더 강한 조정이 필요합니다. 예를 들어, 예측된 가격이 **상한선을 초과**하면 방출량을 늘리고, **하한선 이하**이면 수매량을 늘립니다.
- **TIME ML 모델**을 통해 반복적인 최적화 과정이 수행됩니다:
- N-1 시점의 조정된 가격을 기반으로 새로운 예측을 수행하고, 이를 반복해 **최적의 수매와 방출량**을 결정합니다.
- TIME ML은 자기회귀 구조로 설계되어, 조정된 예측값을 학습에 반영해 **시간 종속성을 더 잘 학습**할 수 있습니다.

#### **4. 목적 함수와 비용 고려**
- 시스템의 목적 함수는 가격이 **‘심각’ 구간에서 벗어나도록** 하는 것입니다.
- 또한, **수매와 방출 비용**, **보관 및 폐기 비용**을 최소화하여 **비용 효율적인 의사결정**을 목표로 합니다.

#### **5. 전체 파이프라인의 강점과 기대효과**
- 예측된 가격이 심각 구간에 들어오기 전에 미리 대응할 수 있어, **시장 안정화**에 효과적입니다.
- **기대 효과:**
- **정부:** 적절한 수매와 방출을 통해 시장 안정화 및 **비용 절감** 가능.
- **농민:** 최소 가격 보장을 통해 소득이 안정되며, 손해를 줄이고 지속 가능한 생산 활동을 이어갈 수 있습니다.
- **소비자:** 가격 변동성이 줄어 안정된 가격으로 농산물을 구매할 수 있어, 생활비 부담이 완화됩니다.

#### **6. 시스템의 차별화**
- 기존의 수매-방출 시스템은 정해진 규칙에 따라 작동하지만, AI 기반 시스템은 **실시간 가격 예측**과 시장 데이터를 반영해 더 **유연하고 민첩한 대응**이 가능합니다.
- 이로 인해, 시장 변화에 맞춘 빠르고 정확한 의사결정이 가능해져, 더 효과적인 가격 안정화를 실현할 수 있습니다.

---

### **요약:**

- **문제 정의:** 농민과 소비자의 문제 해결을 위해 AI 기반 수매-방출 시스템 설계.
- **N-1 시점에서의 의사결정:** 심리적 불안감을 줄이고 안정적인 시장 조정 가능.
- **조정 강도와 최적화:** TIME ML 모델을 통한 반복적인 최적화와 자기회귀 구조 활용.
- **목적 함수 및 비용 고려:** 가격 안정화와 비용 최소화를 동시에 달성.
- **기대 효과:** 정부, 농민, 소비자 모두에게 이익이 되는 솔루션.
- **차별화:** 실시간 예측과 민첩한 대응을 통해 기존 시스템 대비 우수한 성능 제공.

#### **그 밖의 모델 활용 제안: 두 가지 전략**

1. **선물 거래 시스템 (Futures Trading System)**
   - **이점:** 소비자는 미리 정해진 가격에 농산물을 구매해 **가격 급등 리스크**를 줄일 수 있고, 농가는 미리 판매 계약을 통해 **안정적인 수익**을 확보할 수 있습니다.
   - **효과:** 가격 변동성을 줄이고, 수급 예측 가능성을 높여 농산물 시장의 안정성을 강화합니다.

2. **맞춤형 농산물 보험 (Customized Agricultural Insurance)**
   - **차별점:** AI 예측 모델을 기반으로 가격 변동을 분석하고, **농가나 유통업체의 특성**에 맞춰 보장 범위를 설계합니다. 이는 일반적인 표준 보험보다 더 정확하게 리스크를 관리할 수 있습니다.
   - **효과:** 가격 하락 시 손실을 보상해 **농가의 재정적 안정성**을 높이고, 장기적인 농업 생산 활동을 지원해 농업 생태계의 안정화에 기여합니다.
... 


---

## 📖 **Dataset Info**

---

### **1. Train Data (폴더)**

- **train_1.csv:**  
  - **품목:** 배추, 무, 양파, 감자, 대파
  - **데이터 출처:** 가락도매 시장의 평균 경락 가격 정보

- **train_2.csv:**  
  - **품목:** 건고추, 깐마늘, 상추, 사과, 배
  - **데이터 출처:** 건고추, 깐마늘은 중도매 가격, 상추, 사과, 배는 소매 가격의 평균 정보

---

### **2. Meta Data (폴더)**

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

### **3. Test Data (폴더)**

- **2023-2024년 평가 데이터:**  
  - 최대 3개월의 예측 입력 시점 포함 (총 52개 샘플)
  - **TEST_00_1.csv ~ TEST_51_2.csv:** 예측용 데이터

- **Meta Files:**
  - **TEST_경락정보/기상/소매/중도매 (00~51).csv:** 각 시점별 추가 메타 정보
  - **TEST_수출입/생산정보/물가지수_2023-2024.csv:** 최신 수출입, 생산, 물가지수 데이터 (Data Leakage 주의)

---

### **4. Submission File**

- **sample_submission.csv:**  
  - 각 품목의 +1순, +2순, +3순 예측 가격 (TEST 파일별 예측 결과)

---

### **5. 외부 데이터 활용**

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

