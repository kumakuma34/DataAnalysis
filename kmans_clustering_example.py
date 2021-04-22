from sklearn import datasets
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np

iris = datasets.load_iris() #sklearn의 iris 데이터 가져옴
samples = iris.data
print(samples)

x = samples[:,0] #꽃잎의 length
y = samples[:,1] #꽃잎의 width
#plt.scatter(x,y,alpha = 0.5)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepla width(cm)')
#plt.show()

#Iris Dataset은 label이 제공되지만, 제공되지 않는다고 가정하고 kM돌림

#STEP1 : k개의 centroid를 임의로 지정
k = 3
#랜덤으로 x,y좌표를 3개 생성
centroids_x = np.random.uniform(min(x), max(x) , k)
centroids_y = np.random.uniform(min(y), max(y),k)
centroids = list(zip(centroids_x , centroids_y))



#STEP2 : Assgin Datas to Nearest Centroid
def distance(a,b):
    return sum([(el_a - el_b)**2 for el_a, el_b in list(zip(a, b))]) ** 0.5

#각 데이터 포인트를 그룹화할 label을 생성(0, 1, 2)
labels = np.zeros(len(samples))
sepal_length_width = np.array(list(zip(x,y)))
#각 데이터를 순회하면서 centroids와의 거리를 측정
for i in range(len(samples)):
    distances = np.zeros(k)
    for j in range(k):
        distances[j] = distance(sepal_length_width[i], centroids[j])
    cluster = np.argmin(distances) #가장 작은 값의 index를 반환
    labels[i] = cluster


#STEP3 : Update Centroids
centroids_old = deepcopy(centroids)

for i in range(k):
    #각 그룹에 속한 데이터들만 골라 points에 저장
    points = [sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] ==i]
    #points의 각 feature,즉 각 좌표의 평균 지점을 centroid로 지정
    centroids[i] = np.mean(points , axis = 0)

#STEP4 : 반복
centroids_old = np.zeros(centroids.shape)	# 제일 처음 centroids_old는 0으로 초기화 해줍니다
labels = np.zeros(len(samples))
error = np.zeros(k)
# error 도 초기화 해줍니다
for i in range(k):
  error[i] = distance(centroids_old[i], centroids[i])
# STEP 4: error가 0에 수렴할 때 까지 2 ~ 3 단계를 반복합니다
while(error.all() != 0):
  # STEP 2: 가까운 centroids에 데이터를 할당합니다
  for i in range(len(samples)):
    distances = np.zeros(k)	# 초기 거리는 모두 0으로 초기화 해줍니다
    for j in range(k):
      distances[j] = distance(sepal_length_width[i], centroids[j])
    cluster = np.argmin(distances)	# np.argmin은 가장 작은 값의 index를 반환합니다
    labels[i] = cluster
  # Step 3: centroids를 업데이트 시켜줍니다
  centroids_old = deepcopy(centroids)
  for i in range(k):
    # 각 그룹에 속한 데이터들만 골라 points에 저장합니다
    points = [ sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i ]
    # points의 각 feature, 즉 각 좌표의 평균 지점을 centroid로 지정합니다
    centroids[i] = np.mean(points, axis=0)
  # 새롭게 centroids를 업데이트 했으니 error를 다시 계산합니다
  for i in range(k):
    error[i] = distance(centroids_old[i], centroids[i])

colors = ['r', 'g', 'b']
for i in range(k):
    points = np.array([sepal_length_width[j] for j in range(len(sepal_length_width)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()