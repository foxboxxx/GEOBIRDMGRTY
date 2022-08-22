import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pygmt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from gifmaker import gifMaker


goldenPlover = pd.read_csv(r"plovercsv.csv")
goldenPlover = goldenPlover[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].fillna(0)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].fillna(0)
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].astype(int)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].astype(int)
goldenPlover['individualCount'] = goldenPlover['individualCount'].fillna(1)
goldenPlover['individualCount'] = goldenPlover['individualCount'].astype(int)
goldenPlover = goldenPlover.dropna(subset=['year'])
goldenPlover['year'] = goldenPlover['year'].astype(int)
goldenPlover = goldenPlover.query('5 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')

earth2000 = np.zeros(shape=(180, 360))
earth2020 = np.zeros(shape=(180, 360))

goldenPlover2000 = goldenPlover.query('2000 == year').copy()
print(goldenPlover2000)
for x in np.arange(0, goldenPlover2000.shape[0]):
        earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] = earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2000.iloc[x]['individualCount']
        print(x, " -> ", earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,])

goldenPlover2019 = goldenPlover.query('2019 == year').copy()
for x in np.arange(0, goldenPlover2019.shape[0]):
        earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,] = earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2019.iloc[x]['individualCount']
        print(x, " -> ", earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,])

tensor2000 = torch.from_numpy(earth2000)
tensor2020 = torch.from_numpy(earth2020)

print(torch.lerp(tensor2000, tensor2020, 0.5))
npinterpolate = torch.lerp(tensor2000, tensor2020, 0.5).numpy()
np.savetxt("tensor1.csv", npinterpolate, delimiter=",")

# plt.scatter(npinterpolate)
# plt.show()
plt.cla()

# plt.imsave("tensor.png",tensor2020, )
print(earth2020)
xs = earth2020[:,0] # Selects all xs from the array
ys = earth2020[:,1]  # Selects all ys from the array
plt.scatter(x=xs, y=ys)



# Cluster Identification

# y_kmeans = kmeans.predict(X)
# model = KMeans(n_clusters=5).fit(X)


arr = np.zeros(shape = (goldenPlover2019.shape[0], 2))
for i in np.arange(0, arr.shape[0]):
        arr[i,0] = goldenPlover2019['decimalLatitude'].iloc[i]
        arr[i,1] = goldenPlover2019['decimalLongitude'].iloc[i]
xs = arr[:,0] # Selects all xs from the array
ys = arr[:,1]  # Selects all ys from the array
# plt.xlim(-180, 180)
# plt.ylim(-90,90)

# kmeans = KMeans(n_clusters=3)
# kmeans.fit(arr)
# kmeans.labels_

clustering = DBSCAN(eps = 5, min_samples = 5).fit(arr)
clustering.labels_

plt.scatter(x = xs, y = ys, s = 50, c = clustering.labels_)
# plt.show()

testImages = []
for i in np.arange(2000, 2020):
        plt.cla()
        temp = goldenPlover.query('@i == year').copy()
        a = np.zeros(shape = (temp.shape[0], 2))
        for x in np.arange(0, a.shape[0]):
                a[x,0] = temp['decimalLatitude'].iloc[x]
                a[x,1] = temp['decimalLongitude'].iloc[x]
        xs = a[:,0] # Selects all xs from the array
        ys = a[:,1]  # Selects all ys from the array

        clustering = DBSCAN(eps = 5, min_samples = 5).fit(a)
        clustering.labels_ 
        plt.title("American Golden Plover Cluster ID Test {}".format(i))
        plt.scatter(x = xs, y = ys, s = 50, c = clustering.labels_)
        plt.savefig("AmericanGoldenPloverClusterIDTest{}.png".format(i))
        print("{} Complete.".format(i))
        testImages.append("AmericanGoldenPloverClusterIDTest{}.png".format(i))

gifMaker(testImages, "Test Image.gif", 0.2)
# def clusterID(minYear, maxYear, birdData, birdName,)

