#DL04
#손글씨 이미지 픽셀화 테스트
from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

#seed값 설정
seed = 9898
np.random.seed(seed)
tf.set_random_seed(seed)

#MNIST 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("학습셋 이미지 수 : %d"  % (x_train.shape[0]))
print("테스트셋 이미지 수 : %d" % (x_test.shape[0]))

#첫번째 데이터 그래프로 확인
plt.imshow(x_train[1112], cmap='Greys')  #이미지파일을 흑백으로 출력
plt.show()

#이미지를 컴퓨터는 어떻게 인식할까? - 픽셀단위로 확인
#이미지는 28x28, 총 784픽셀로 이루어져있음
#각 픽셀의 밝기 정도에 따라 0~255까지 숫자 중 하나로 표기
for x in x_train[1112]:
    for i in x:
        sys.stdout.write('%d\t' % i)
    sys.stdout.write('\n')

#이렇게 변환된 픽셀들의 집합을 고유의 숫자집합으로 바꿔야 함
#즉, 784개의 속성을 이용해서 10개의 결과집합이 나오도록 해야 함
#입력이미지(3) -> [0,0,0,1,0,0,0,0,0,0] = 3
#입력이미지(9) -> [0,0,0,0,0,0,0,0,0,1] = 9

#차원 변환 - 2차원 배열(28x28)을 1차원 배열(784)로 변환
#1차원으로 변환된 데이터 값들은 다시 0, 1 사이 값으로 변환
x_train = x_train.reshape(x_train.shape[0], 784)
x_train = x_train.astype('float64')
    #0,1로 변환하기 위해 255로 나누려고
    #정수 데이터를 실수 데이터로 바꿈
x_train = x_train / 255     #0, 1중 하나로 변환

x_test = x_test.reshape(x_test.shape[0],784).astype('float64') / 255

#선택한 데이터의 결과값 미리 확인
print("결과값 : %d" % (y_train[1112]))

#이미지 인식결과 확인
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(y_train[1112])