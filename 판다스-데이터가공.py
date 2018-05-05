#데이터 취득 및 가공
import pandas as pd
import numpy as np

#데이터 읽어 들이기 - GAP_5year.csv(탭으로 구분)
gap = pd.read_csv('c:/Java/GAP_5year.tsv',sep='\t')
# print(gap) #많은 내용을 출력하기엔 다소 불편
print(gap.head())       #위에서 5줄
print(gap.tail())       #아래에서 5줄
print(gap.info())       #데이터 구조 표시
print(gap.describe())   #기술통계 요약

#데이터 조회
#통계자료에서 2007 한국 데이터
#R dplyr - filter(gap, country==''&year==)
kor = gap.query("country=='Korea, Rep.'")
print(kor)
kor = gap.query("country=='Korea, Rep.' & year == 2007")
print(kor)

#정렬해서 출력
#year, country 순으로 정렬 후 출력
sort = gap.sort_values(by=['year','country'])
print(sort.head())

#부분열 선택하기
#인구수, 1인당 GDP를 출력
#R dplyr - gap %>% select(pop, gdpPercap)
partcol = gap[['pop','gdpPercap']]
print(partcol.head())

#특정열 추가하기
#총GDP(gdp): 인구수 * 1인당 GDP
#GDP증가율(gdp_ratio): lifeExp / gdpPercap
#GDP증가비(%)(gdp_perc): gdp_ratio * 100

gap['gdp'] = gap['pop'] * gap['gdpPercap']

gap['gdp_ratio'] = gap.lifeExp / gap.gdpPercap

gap['gdp_perc'] = gap.gdp_ratio * 100
print(gap.head())

#통계량 계산하기
print(gap.aggregate(['mean','median']))
print(gap.agg(['sum','min','max']))

#표본 추출을 위한 샘플링: sample()
from datetime import datetime
timeseed = int(datetime.timestamp(datetime.now()))
np.random.seed(timeseed)
print(gap.sample(n=10))

#각 항목 한 번 씩 출력: 중복제거
print(gap.country)
print(gap.country.unique())

print(gap.year)
print(gap.year.unique())

print(gap.drop_duplicates(['country','year']).head())

#group by연산
#2007년 기준 대륙별 생활지수 평균값 분석
lifeExp2007 = gap.query('year==2007').groupby('continent').agg({'lifeExp':'mean'})
print(lifeExp2007)