#DL03b
#앞서 이미지를 average_hash로 픽셀화한 것으로
#여러 이미지에서 유사 이미지를 찾아봄

#http://www.vision.caltech.edu/

from PIL import Image
import numpy as np
import os, re

#이미지 파일경로 정의
find_dir = 'C:/Java/101_ObjectCategories'
cache_dir = 'C:/Java/101_ObjectCategories/cache_avghash'

#avrage cache 디렉토리가 없으면 자동생성
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

#이미지를 평균해쉬법에 의해 픽셀로 변환
#avghash 파일에 대한 cache파일 생성
def average_hash(fname, size=24):
    fname_new = fname[len(find_dir):]

    #이미지 캐시하기
    cache_file = cache_dir + "/" + fname_new.replace('\\','_') + ".csv"
    if not os.path.exists(cache_file):
        img = Image.open(fname) #이미지 파일을 읽기
        img = img.convert('L')  #이미지를 흑백으로 변환
        img = img.resize((size,size),Image.ANTIALIAS)   #이미지크기 변환

        pixel_data = img.getdata()      #픽셀 데이터 가져옴
        pixels = np.array(pixel_data)   #픽셀 데이터를 numpy 배열로 생성
        pixels = pixels.reshape((size, size))  #2차원 배열로 변환
        avg = pixels.mean()                    #평균값 구하기
        px = 1 * (pixels > avg)

        np.savetxt(cache_file, px, fmt='%.0f', delimiter=',')
        #생성된 캐시파일을 지정한 위치에 저장
    else:
        px = np.loadtxt(cache_file, delimiter=',')
        #캐시되어 있다면 지정한 위치에서 바로 파일 읽기
    return px

#비교 대상 이미지 간 유사성을 알아보기 위해
#해밍 거리 판별법을 사용함 - 서로 다른 기호의 갯수를 셈
#1011101과 1001001사이의 해밍거리 : 2
#2143896과 2233796사이의 해밍거리 : 3
#toned와 roses사이의 해밍거리 : 3
def hamming_dist(a, b):
    aa = a.reshape(1, -1)
    ab = b.reshape(1, -1)
    dist = (aa != ab).sum()
    return dist

def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root, f)
            if re.search(r'\.(jpg|jpeg|png)$', fname):
                yield fname

#지정한 이미지와 유사한 이미지 찾기
def find_image(fname, rate):
    src = average_hash(fname)
    for fname in enum_all_files(find_dir):
        dst = average_hash(fname)
        diff_r = hamming_dist(src, dst) / 256
        # print("[check] ",fname)
        if diff_r < rate:
            yield (diff_r, fname)

#실제 찾을 이미지파일 지정
srcfile = find_dir + '/image_0034.jpg'   #캥거루 이미지
html = ""
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x: x[0])
#찾은 이미지들은 html 형태로 작성
for r, f in sim:
    print(r, ">", f)
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>' + \
        '<p><a href="' + f + '"><img src="' + f + '" width=400>' + \
        '</a></p></div>'
    html += s

#작성된 html은 파일로 저장
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}' width=400></p>{1}
</body></html>""".format(srcfile, html)
with open("./avghash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("ok")