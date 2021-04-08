
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import sys


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

order_data = pd.read_csv('./52037_order_2018.csv' ,encoding='CP949')
order_data['CUST'] = order_data['BIZPLC_ID'] + '_' + order_data['RSTR_ID']
print(order_data)

goods_data = pd.read_csv('./GD_CLS_ID_INFO.csv' ,encoding='CP949')
print(goods_data)

df = pd.DataFrame(columns = goods_data['GD_CLS_ID'])
print(df)