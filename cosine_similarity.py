import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('./movies_metadata.csv', low_memory = False)
data = data.head(20000)
#print(data['overview'].isnull().sum())
data['overview'] = data['overview'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix)
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index = data['title']).drop_duplicates()
print(indices)

def get_recommendations(title, cosine_sim = cosine_sim):
    #선택한 영화의 타이틀에 해당하는 인덱스 받아오기
    idx = indices[title]

    #모든 영화에 대해 해당 영화의 유사도 구하기
    sim_scores = list(enumerate(cosine_sim[idx]))

    #유사도에 따라 영화들을 정렬
    sim_scores = sorted(sim_scores , key=lambda x: x[1] , reverse = True)

    #가장 유사한 10개의 영화 받아오기
    sim_scores = sim_scores[1:11]

    #가장 유사한 10개의 영화의 인덱스 받아오기
    movie_indices = [i[0] for i in sim_scores]

    #가장 유사한 10개의 영화의 제목을 리턴
    return data['title'].iloc[movie_indices]


print(get_recommendations('The Dark Knight Rises'))


