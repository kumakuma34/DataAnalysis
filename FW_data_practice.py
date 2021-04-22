
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

order_data = pd.read_csv('./52037_order_2018.csv' ,encoding='CP949')

#BIZPLC_ID , RSTR_ID로 key로 쓸 고객 정보 생성
order_data['CUST'] = order_data['BIZPLC_ID'].apply(str) + '_' + order_data['RSTR_ID'].apply(str)
#print(order_data)

goods_data = pd.read_csv('./GD_CLS_ID_INFO.csv' ,encoding='CP949')
goods_data['GD_CLS_NM'].apply(str)
#print(goods_data)

#코사인 유사도 검사에 쓸 수 있는 형태로 dataFrame 생성
df = pd.DataFrame(index = order_data['CUST'] , columns = goods_data['GD_CLS_NM'] )
df.loc['1003_1004']['대두유'] = 1
print(df)