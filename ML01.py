# 머신러닝의 개념
# 프로그램이 어떤 것에 대한
# 학습을 통해 기존의 모델이나 결과물을 개선하거나
# 예측하게끔 구축하는 과정을 의미
# 데이터를 이용해서 명시적으로 정의되지 않은 패턴을
# 컴퓨터로 학습하여 결과를 만들어 내는 분야
# 직접적으로 프로그래밍 하지 않아도
# 컴퓨터가 스스로 학습할 수 있는 능력을 주는 학문

# 머신러닝 변천사
# 고전적 인공지능 시대[규칙기반](1950) - 신경망 시대(1960) -
# 머신러닝[통계기반](1990) - 빅데이터(2010) -
# 딥러닝, 고급신경망(2013)

# 머신러닝기반 데이터 분석 기법의 유형
# 머신러닝 - 지도학습(값/레이블을 예측하는 시스템 구축)
#            주어진 데이터와 정답(레이블)을 이용해서
#            미지의 상태나 값을 예측하는 학습 방법
#            현재 주식시장 변화 학습 - 미래 주식시장 변화 예측
#            사용자가 구매한 상품을 토대로 다음에 구입할 상품 예측

#            비지도학습(패턴 추출)
#            데이터 자체에서 유용한 패턴을 찾아내는 학습방법
#            비슷한 데이터를 묶는 군집화
#            데이터에서 이상한 점을 찾아내는 이상검출

#            강화 학습(상호작용가능한 시스템 구축)
#            기계가 환경과 상호작용을 통해 정기적으로 얻는 이득을
#            최대화하도록 하는 학습방법
#            바둑프로그램이 바둑을 두는 경우, 현재 수에서 다음 수를
#            선택하는 것은 지도학습에 속함 하지만, 단순히 다음 수만
#            고려하는 것이 아니고 게임의 승패까지 고려해서 전체수를
#            보게 하는 수를 학습하는 과정이 강화학습

# 지도학습 - 회귀(값 예측): 연속된 숫자값(실수) 예측
#           분류(항목선택): 입력된 데이터를 주어진 항목으로 나눔
#           순위/추천(순서배열): 대상에 대한 선호도를 예측

# 비지도학습 - 군집화(비슷한 데이터 묶음)
#              밀도추정(데이터 분포 예측)
#              각 국 학생들의 키/몸무게 통계자료에서
#              키/몸무게의 관계를 밀도로 분석
#              차원축소(데이터 차원 간추림)
#              데이터가 복잡해서 시각화가 어려울 때
#              2/3차원으로 축소

# 딥러닝 - 신경망을 층층히 쌓아서 문제를 해결하는 기법의 총칭
#          데이터의 양에 의존하는 기법으로 다양한 패턴과 경우에
#          유연하게 대응하는 구조로 만들어 많은 데이터를 이용해서
#          학습해야 성능이 향상되는 구조 채택

# 머신러닝 기반 데이터 분석 계획 및 절차

# 지도학습의 대표적인 알고리즘
#           분류              :         수치예측(회귀)
# K-최근접 이웃(K-Nearest Neighbors) : 선형 회귀(Linear Regression)
# 로지스틱 회귀(Logistic Regression) : 확장된 회귀분석(ex : 다항회귀, 비선형 회귀, 벌점화 회귀 등)
# 인공 신경망 분석(Artificial Neural Network) : 인공 신경망 분석(Artificial Neural Network)
# 의사결정트리(Decision Tree) : 의사결정트리(Decision Tree)
# 서포트 벡터 머신(Support Vector Machine) : 서포트 벡터 머신(회귀)(Support Vector Machine (Regression))
# 나이브 베이즈(Naive Bayes) : PLS(Partial Least Squares)
# 앙상블 기법(랜덤 포레스트 등) : 앙상블 기법(랜덤 포레스트 등)

# 비즈니스 이해 및 문제 정의 - 분석주제 선정, 판단
# 데이터 수집 - 데이터 확보
# 데이터 전처리와 탐색 - 결측/이상치 처리 변환, 표준화, 요약, 시각화
# 데이터에 대한 모델훈련 - train/test데이터 분리, 머신러닝 기법 적용
# 모델 성능 평가 - 평가지표 계산
# 모델 성능 향상 및 현업 적용 - 튜닝 및 성능 개선

# 데이터 세트 분할하기
#  일정 비율로
#
# 일반적으로 학습용과 평가용 데이터 각각의 분할은 전체 데이터에서
# 랜덤하게 특정 비율로 학습용 데이터를 추출하고, 학습용 데이터에
# 사용되지 않은 나머지 데이터를 평가용 데이터로 취하는 방법을 따른다
#
# 데이터분할 방법1: 연속적인 순서대로 나눔
# iris_train <- iris[1:105,]  #데이터의 첫 행부터 105행까지 총 70%를 학습용으로 할당
# iris_train <- iris[106:150,] #106행부터 마지막 행 까지 총 30%를 평가용으로 할당
#
# 데이터 분할 방법2: 무작위 추출로 나눔
# idx <- sample(1:nrow(iris),size=nrow(iris)*0.7,replace=F) #70%
# iris_train <- iris[idx,]
# iris_test <- iris[-idx,]

# 지도학습 모델 적용하기
# K-최근접 이웃 기법:KNN
# 상품 및 서비스 추천 등

# 목표변수의 범주를 알지 못하는 데이터 세트의 분류를 위해 해당
# 데이터 세트와 가장 유사한 주변 데이터 세트의 범주로 지정

# 해당 데이터 점과 주변 데이터 세트 간의 ‘유사성’을 어떻게 측정기준
# 목표변수의 범주를 분류할 때 주변 데이터 몇 개를 참고할지 여부

# iris데이터 집합을 이용해서 KNN알고리즘 적용
# iris: 1953년 통계학자의 붓꽃 유사성 분류 논문에 사용된 데이터
#
# 아마추어 식물학자가 들에서 발견한 붓꽃의 품종을 알고 싶다고 가정
# 학자는 붓 꽃의 꽃잎petal과 꽃받침sepal의 폭과 길이를 알고 있음
#
# 한편, 전문식물학자는 붓꽃을 setosa, versicolor, virginica 종으로
# 분류해서 측정한 데이터가 있다고 가정
# 식물학자가 측정한 값을 이용해서 이 붓꽃이 어떤 품종인지 구분

#파이썬에서 ML을 테스트/구현하려면
#numpy, scipy, matplotlib, ipython, scikit-learn, pandas
#ML을 위한 데이터집합 라이브러리: sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()          #iris데이터 불러옴
print(iris.keys())        #iris 데이터 키 출력
print(iris['data'][:5])   #iris 데이터 처음 5행만 출력
print(iris['target'])  #iris 타겟
print(iris['target_names'])  #iris 타겟명

#scikit-learn에서 train_test_spkit함수를 이용해서
#데이터집합을 일정비율(75:25)로 나눠 train/test로 작성
from sklearn.model_selection import train_test_split
print(iris['data'].shape)    #iris 데이터 크기
print(iris['target'].shape)  #iris 타겟 크기
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'],random_state=0)

print("학습데이터 크기",x_train.shape)
print("평가데이터 크기", y_train.shape)
print(x_test.shape)
print(y_test.shape)

#머신러닝 모델을 만들기 전에
#머신러닝을 이용해서 풀어도 되는 문제인지
#데이터에 이상은 없는지 여부 확인을 위해 시각화 도구 이용
#대표적인 시각화 도구 : 산포도/산점도
#하지만, 산점도로는 3개 이상의 특성을 표현하기 어려움
#모든 특성을 짝지어 만드는 산점도 행렬을 사용할 것을 추천
#4개의 특성을 가진 붓꽃의 경우에 적합

#단, 데이터가 행렬로 작성되어 있기 때문에
#dataframe으로 변환 필요
from pandas.plotting import scatter_matrix
iris_df = pd.DataFrame(x_train)
scatter_matrix(iris_df, c=y_train,figsize=(15,15),marker='o')
plt.show()

#k-최근접 알고리즘에서 k는 가장 가까운 이웃 '하나'가 아니라
#훈련데이터 중 새로운 데이터와
#가장 가까운 k개의 이웃을 찾는다는 의미
#머신러닝의 모든 모델은 scikit-learn의 Estimator클래스에 구현
#KNN알고리즘은 neighbors 모듈의 KNeighborClassifier클래스에 구현
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) #k=1로 설정

knn.fit(x_train,y_train) #train 데이터로 분류모델 학습시킴

x_new = np.array([[5, 2.9, 1, 0.2]])  #예측을 위한 데이터 생성

prediction = knn.predict(x_new)  #예측값 조사
print('예측결과',prediction)
print('예측결과 대비 품종',iris['target_names'][prediction])

#예측모델 평가 - 신뢰성 확인
#앞서 만든 test데이터집합을 이용
y_pred = knn.predict(x_test)
print('test 데이터를 이용한 예측값',y_pred)
print('test 데이터 대비 예측값 정확도',np.mean(y_test == y_pred))
print('test 데이터 대비 예측값 정확도',knn.score(x_test, y_test))

#대부분의 지도학습 알고리즘(의사결정나무,SVM,나이브베이지안)이
#그렇듯 주어진 학습자료들로 부터 모형(모델)을 추정하여
#새로운 실증자료가 주어지면 모형에 적합하여 예측값을 산출함
#이러한 학습방법을 eager방식이라 함

#하지만, KNN은 학습자료가 주어져도 아무런 움직임이 없다가
#실증자료가 주어져야만 그제야 움직이기 시작함
#이러한 학습방법을 lazy방식이라 함

#지도학습 분류방법 중 가장 간단한 방법: KNN
#많은 메모리 소요(대용량 데이터일 때 불리함)
#대안: 로지스틱 회귀, 딥러닝 이용

#분류의 개념
#미리 정의된, 가능성 있는 여러 결과값(레이블) 중 하나를 예측
#iris의 경우 결과값은 모두 3가지 : setosa,versicolor,virginica
#이진분류: 2가지 질문의 답 중 하나를 예측(스팸메일 분류)
#다중분류: 3가지 이상 질문의 답 중 하나를 예측

#일반화/과대적합/과소적합
# 일반화: 모델을 통해 처음 보는 데이터에 대해 정확히 예측할 수 있는 경우
# 예) 요트구매 고객정보: 나이, 보유차량수, 주택보유, 자녀수, 혼인여부, 애완견, 보트구매
#
# 요트를 구매한 고객과 구매의사가 없는 고객 데이터를 토대로
# 누가 요트를 구매할지 예측해보자
#
# 과대적합:나이가 45세 이상, 자녀가 3이상,
#          이혼하지 않은 고객은 요트를 구매할 것이다
#          너무 많은 특성을 이용해서 복잡한 모델을 만드는 경우
# 과소적합:
#          너무 작은 특성을 이용해서 단순한 모델을 만드는 경우

#KNN알고리즘: k값의 변화에 따른 산점도 비교
#mglearn 패키지 설치
import mglearn as mg
x, y = mg.datasets.make_forge()     #데이터집합 생성
mg.discrete_scatter(x[:,0], x[:,1], y)  #산점도 작성
plt.show()

#k값이 1일 때 KNN알고리즘 산점도
mg.plots.plot_knn_classification(n_neighbors=1)
plt.show()

#k값이 3일 때 KNN알고리즘 산점도
mg.plots.plot_knn_classification(n_neighbors=3)
plt.show()

#k값이 5일 때 KNN알고리즘 산점도
mg.plots.plot_knn_classification(n_neighbors=5)
plt.show()


#k값을 3으로 설정후 KNN알고리즘 적용
#데이터집합 생성
x,y = mg.datasets.make_forge()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)    #train 데이터로 학습

print(knn.predict(x_test))    #test로 데이터 예측

print('정확도', knn.score(x_test, y_test))  #예측 평가