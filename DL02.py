#DL02
#iris 데이터 집합으로 품종 구분을 딥러닝으로 구현
#꽃잎의 모양과 길이로 여러가지 품종으로 나뉨
#앞서 푼 폐암환자 생존율(0/1) 계산과 달리
#범주 값이 3가지 임 - 다중 분류

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sbs
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

#아이리스 데이터 읽기
df = pd.read_csv('data/iris.csv', names=['sepal_length','sepal_width','petal_length','petal_width','species'])
print(df.head())

#아이리스 데이터 산점도 확인
sbs.pairplot(df, hue='species')
plt.show()

#난수 설정
seed = 9898
np.random.seed(seed)
tf.set_random_seed(seed)

#데이터 분류
dataset = df.values
x = dataset[:,0:4].astype(float)
y_tmp = dataset[:,4]     #Iris-setosa,...

#문자열을 숫자로 변환
#즉, species변수는 문자값으로 구성 - 숫자로 변환할 필요 존재
e = LabelEncoder()
e.fit(y_tmp)                                #array[값1, 값2, 값3]
y_trans = e.transform(y_tmp)                #array[0,1,2]
y = np_utils.to_categorical(y_trans)        #[ [1,0,0], [0,1,0], [0,0,1] ]

#딥러닝 실행 방식을 결정(모델 설정 및 실행방법 정의)
model = Sequential()
model.add(Dense(16, input_dim=4,activation='relu'))
model.add(Dense(3, activation='softmax'))
#최종 출력값이 3개 중 하나이어야 하므로 softmax함수 이용
#즉, 각 항목 값의 합은 1이 되도록 계산해주는 함수

#딥러닝 컴파일
#최종 출력값이 3개인 다중 분류이기 때문에
#loss는 categorical_cossentropy를 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#딥러닝 실행
model.fit(x,y, epochs=50, batch_size=1)

#결과 검증 및 출력
print('정확도: %.4f' % (model.evaluate(x, y)[1]))