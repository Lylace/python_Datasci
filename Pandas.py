#Pandas
#효과적인 데이터 분석 기능을 제공하는 패키지
#R에서 자주사용하는 DataFrame을 파이썬에서도 사용할 수 있게 함
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_excel('c:/Java/sungjuk.xlsx')
# print(df)
#xlrd 패키지 설치 필요! - 엑셀 파일을 읽어 dataframe으로 생성

#총점, 평균 계산 후 df에 추가
subj = ['국어', '영어', '수학', '과학']
df['총점'] = df[subj].sum(axis=1)
df['평균'] = df['총점'] / len(subj)
df.sort_values(['평균'], ascending=[False]) #평균으로 정렬


import matplotlib as mpl
# font_name = mpl.font_manager.FontProperties(fname='c:/windows/fonts/malgun.ttf').get_name()
# mpl.rc('font',family=font_name)
mpl.rc('font',family='Malgun Gothic')
sj = df.sort_values(['평균'],ascending=[False])
sj.index = sj['이름']
sj['평균'].plot(kind='bar',figsize=(8,4))

#DataFrame 객체 생성: {'key' : ['val','val',...]}
data = {'이름':['수지','혜교','지현'],
        '국어':[99,98,99],
        '영어':[55,77,88],
        '수학':[44,77,99]}
sj = pd.DataFrame(data)

print(sj)
print(sj['이름'])  #특정 컬럼만 보기
# DataFrame(data,columns=['이름','국어','영어','수학'])

#Series: 1차원 자료구조, df에서 특정 컬럼 추출 시 Series 생성

data = [4,5,6,7,8,9,10]
print(data)             #그냥 1차원 배열
print(pd.Series(data))  #행번호가 있는 다 차원 배열

#데이터 프레임에 새로운 컬럼 추가
name = ['수지','혜교','지현']
kor = [99,98,99]
eng = [55,77,88]
mat = [44,77,99]

data={'이름':name,'국어':kor,'영어':eng, '수학':mat}
sj = pd.DataFrame(data, columns=['이름','국어','영어','수학'])
print(sj)

gender = pd.Series(['남','여','여'])
sj['성별'] = gender
print(sj)

#dataframe 행/열 삭제
sj = sj.drop(0, axis=0)    #첫번째 행 삭제
print(sj)

sj = sj.drop('성별',axis=1)   #'성별'열 삭제
print(sj)