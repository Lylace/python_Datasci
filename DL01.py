#딥러닝 기초 - 텐서플로, 케라스

#딥러닝 개발환경
#1. 파이썬 3.5이상 또는 아나콘다 4.4이상
#2. 파이참 - 파이썬 venv환경에서 실행
#3. 머신러닝/딥러닝 관련 패키지 설치
#numpy, scipy, maltplotlib, pandas, scikit-learn
#spyder, seaborn, h5py, pillow,
#tensorflow(tensorflow-gpu) keras

#https://www.anaconda.com/
#https://www.tensorflow.org/
#https://keras.io/
#https://www.gitbook.com/@tensorflowkorea

#4. 텐서플로우 GPU지원 사이트
#CUDA설치 - https://developer.nvidia.com/cuda-downloads
#cuDNN설치 - https://developer.nvidia.com/cudnn

#5. 설치확인
#파이참 - '파이썬 콘솔'에서 다음 실행
import tensorflow as tf
print(tf.__version__)      #텐서플로우 버젼 확인

import keras

hello = tf.constant('Hello, TensorFlow!!')
sess = tf.Session()             #텐서플로우 작업 생성
print(sess.run(hello))          #텐서플로우 작업 실행

#인공지능 - 관념적으로 컴퓨터가 인간이 사고를 모방하는 것
#           즉, 기계가 사람처럼 사고하고 행동하게 하는 것

#머신러닝 - 주어진 데이터를 통해 컴퓨터가 스스로 학습하는 것
#           학습: 데이터를 입력해서 패턴을 분석하는 과정
#머신러닝의 한계 - 인간도 파악하기 어려운 복잡한 문제는
#                  머신러닝으로도 풀기 어려움(이미지 인식)

#딥러닝 - 인공신경망을 이용해서 컴퓨터가 스스로 학습하는 것
#         인공신경망 : 인간 뇌의 동작방식을 착안해서 만듦
#         2012년 ImageNet에서 제공하는 1,000개의 카테고리로
#         분류된 100만개의 이미지를 인식하여 정확성을 겨루는
#         ILSVRC라는 이미지 인식대회에서 84.7%라는 인식률 달성
#         그 전까지는 75%대 였음 - 현재는 97%에 육박할 정도

#인공신경망은 이미 1940년대 부터 연구되던 기술
#그 전까지는 여러가지 문제에 부딪혀 암흑의 시대를 지내다가
#빅데이터와 GPGU의 발전, 수 백만, 조 개로 이뤄진 아주 간단한
#수식을 효율적으로 수 천번 이상해야 하는 딥러닝 알고리즘의
#발명 덕택에 급격히 발전되고 있음

#텐서플로우는 머신러닝 프로그램, 특히 딥러닝 프로그램을 아주
#쉽게 구현할 수 있도록 다양한 기능을 제공하는 머신러닝 라이브러리
#구글에서 제작하고 배포하고 있음

#케라스, 카페, 토치, MXNet, 체이너, CNTK
#텐서플로우를 좀 더 사용하기 쉽게 만들어 주는 보조 라이브러리

#인공신경망 구동 예제
# x1 = '여친에게 데이터 요청 전화옴 : 0.5'
# w1 = '여친을 너무 좋아한다 :7'
# w1 = '여친과 냉전상태 : 1'
#
# x2 = '지금 밖에는 눈이 온다 : 3'
# w2 = '난 눈을 맞는 것을 좋아해 : 5'
# w2 = '난 눈에 맞아 젖는 걸 싫어해 : 1'
# 
# x3 = '난 지금 점심을 안 먹었는데'
# w3 = '난 배고픈 거 싫어'
# w3 = '여친을 만난다면 배고픈것 쯤이야!'
#
# y = 0:'그냥 집에 있는다', 1:'여친과 데이트한다'
# 0.5 x 0.1 + 0.3 x 1 = 0.35 = 0

#딥러닝 간단 예제
#2013년 폴란드 브로츠와프 의과대학
#폐암환자의 수술 전 진단정보와 수술 후 생존여부

#딥러닝을 구동하는데 필요한 케라스 함수를 불러옴
from keras.models import Sequential
from keras.layers import Dense

#머신러닝 관련 라이브러리 불러옴
import numpy as np
import tensorflow as tf

#난수 생성을 위한 초기화
seed = 9898
np.random.seed(seed)
tf.set_random_seed

#준비된 환자 정보를 불러옴
#종양유형, 폐활량, 호흡곤란여부, 고통정도, 기침, 흡연 천식여부
data_set = np.loadtxt('data/ThoraricSurgery.csv', delimiter=',')

#환자기록과 수술결과를 x, y로 구분해서 저장
x = data_set[:, 0:17]
y = data_set[:, 17]

#딥러닝 실행 방식을 결정(모델 설정 및 실행방법 정의)
#https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/
model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
#입력데이터는 17, 은닉층 갯수는 30, 적용 알고리즘 함수는 relu
model.add(Dense(1, activation='sigmoid'))
#출력 데이터는 1, 적용 알고리즘 함수는 sigmoid

#딥러닝 실행
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#loss는 오차값 추적 방식, optimizer는 오차 수정 함수
model.fit(x,y, epochs=30, batch_size=10)
#검증값 epochs

#결과 검증 및 출력
print('정확도 : %.4f'%(model.evaluate(x,y)[1]))

