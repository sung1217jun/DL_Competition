# Credit Score Classification using TabNet

고객의 금융 및 신용 데이터를 활용하여 신용등급(Credit Score)을 분류하는 딥러닝 기반 프로젝트를 진행하였다.
본 프로젝트에서는 정형 데이터 딥러닝 모델인 TabNet을 활용하여 고객의 금융 패턴을 학습하고, Good / Standard / Poor 신용등급을 예측하였다.

<br>

## 프로젝트 개요

금융 데이터에는 고객의 연체 정보, 부채 규모, 이자율, 투자 금액, 카드 사용률 등 다양한 정보가 포함되어 있다.
이러한 데이터를 기반으로 고객의 신용 상태를 예측하는 것은 금융권의 리스크 관리 및 고객 평가 측면에서 매우 중요한 문제이다.

본 프로젝트에서는 금융 데이터 특성을 반영한 전처리 및 파생변수 생성을 수행하였으며, TabNet 모델을 활용하여 신용등급 분류 성능을 향상시키는 것을 목표로 하였다.

---

## 사용 기술

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* PyTorch TabNet
* Matplotlib

---

## 데이터 전처리

데이터 전처리 단계에서는 고객 식별 목적의 변수인 ID, Customer_ID, Name, SSN 등을 제거하였다.
해당 변수들은 모델의 일반화 성능 향상에 도움이 되지 않고 과적합 가능성을 높일 수 있기 때문이다.

범주형 변수는 TabNet 학습이 가능하도록 Ordinal Encoding을 적용하였으며, 결측치는 train 데이터 기준 중앙값 및 "Unknown" 값으로 처리하였다. 또한 데이터 누수를 방지하기 위해 encoder와 scaler는 train 데이터에 대해서만 fit을 수행하였다.

추가적으로 금융 데이터 특성을 반영하기 위해 다음과 같은 파생변수를 생성하였다.

* Debt_to_Income
* EMI_to_Salary
* Delay_Payment_Risk
* Inquiry_per_Card

이를 통해 단순 원본 변수보다 고객의 금융 위험도를 효과적으로 반영할 수 있도록 구성하였다.

---

## EDA 및 데이터 분석

EDA 결과 신용등급은 Good, Standard, Poor의 다중분류 구조를 가지며, Standard 클래스의 비율이 가장 높은 불균형 데이터 형태를 보였다.
이때 LabelEncoder를 이용해서 Good -> 0, Poor -> 1, Standard ->2 로 변환하였다.
<p align="center"> <img src="https://github.com/user-attachments/assets/eb1da9f7-fe9c-4899-a0a4-c1d4265ed7ec" width="70%" /> </p>

또한 주요 변수 분석 결과 다음과 같은 특징을 확인하였다.

* Outstanding_Debt가 증가할수록 Poor 비율 증가
* Delay_from_due_date 증가 시 신용등급 하락 경향
* Interest_Rate가 높을수록 금융 위험 증가
* Num_of_Delayed_Payment가 신용등급과 높은 연관성 보임

<p align="center">
  <img src="https://github.com/user-attachments/assets/5f5505ed-228c-4b2b-8fdb-a1a7c7e9870e" width="45%">
  <img src="https://github.com/user-attachments/assets/43f7cd4c-5cce-4248-94fe-7fa606f4aed4" width="45%">
</p>

이를 기반으로 연체 및 부채 관련 파생변수를 추가 생성하고 모델 학습에 활용하였다.

---

## 모델링

모델은 정형 데이터 딥러닝 모델인 TabNet을 사용하였다.

TabNet은 Attention 기반 Feature Selection 구조를 활용하여 중요한 feature를 단계적으로 선택하며 학습할 수 있고, 범주형과 수치형 데이터가 혼합된 금융 데이터에 적합하다고 판단하였다.

모델 성능 향상을 위해 다음 요소들을 중심으로 튜닝을 진행하였다.

* n_d
* n_a
* n_steps
* learning rate
* batch size
* StepLR scheduler
* Early Stopping

또한 mutual information 기반 Feature Selection을 적용하여 타깃과 관련성이 높은 feature를 우선적으로 학습하도록 구성하였다.

<p align="center"> <img src="https://github.com/user-attachments/assets/f08ff7d5-8576-473a-b738-4ab3e56d70ee" width="75%"> </p>


---

## 성능 결과

최종 모델의 Validation Accuracy는 0.7564, Validation F1 Score는 0.7566으로 나타났다.

Classification Report 결과 Good, Poor, Standard 클래스 모두 비교적 안정적인 precision과 recall을 보였으며, 특정 클래스에 편향되지 않은 균형 잡힌 예측 성능을 확인할 수 있었다.

추가적으로 Credit_History_Age 개월 수 변환, Type_of_Loan 멀티핫 인코딩, 파생변수 강화 등의 실험도 진행하였으나 기존 모델 대비 Validation Score가 감소하여 최종적으로 기존 TabNet 모델을 채택하였다.

<p align="center"> <img src="https://github.com/user-attachments/assets/dd02fe02-b604-4756-b6e9-a624538b7367" width="55%"> </p>

---

## 결과 요약

* 정형 데이터 딥러닝 모델 TabNet 활용
* 금융 특화 파생변수 생성
* Mutual Information 기반 Feature Selection 적용
* Validation F1 Score : 0.7566 달성
* 추가 Feature Engineering 실험 진행 및 비교 분석 수행

---

## Feature Importance Analysis

TabNet Feature Importance 분석 결과, 금융 위험도와 직접적으로 관련된 변수들이 높은 중요도를 가지는 것을 확인하였다.

특히 `Outstanding_Debt`, `Interest_Rate`, `Debt_to_Income`, `Delay_from_due_date` 등의 변수는 고객의 신용 상태를 예측하는 데 핵심적인 역할을 수행하였다. 또한 `Delay_Payment_Risk`, `Inquiry_per_Card`와 같은 파생변수 역시 높은 중요도를 보이며, 단순 원본 변수보다 금융 위험 패턴을 효과적으로 반영하는 것을 확인할 수 있었다.

이는 EDA 과정에서 확인한 “부채 규모 증가”, “연체 증가”, “이자율 상승”이 신용등급 하락과 밀접한 관련이 있다는 분석 결과와 유사한 흐름을 보였다.

<p align="center"> <img src="https://github.com/user-attachments/assets/6f574347-b524-4bbf-8dfe-a618b3ea9675" width="70%"> </p>

