import os
from xml.etree.ElementTree import ProcessingInstruction

from functions import getDistanceFromLatLonInKm
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import pygmt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from sklearn.cluster import MeanShift
from sklearn import metrics
from gifmaker import gifMaker
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import math
import seaborn as sb
from matplotlib.lines import Line2D

from geopy.distance import great_circle
from shapely.geometry import MultiPoint

        
goldenPlover = pd.read_csv(r"plovercsv.csv")
goldenPlover = goldenPlover[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].fillna(0)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].fillna(0)
goldenPlover['decimalLongitude'] = goldenPlover['decimalLongitude'].astype(float)
goldenPlover['decimalLatitude'] = goldenPlover['decimalLatitude'].astype(float)
goldenPlover['individualCount'] = goldenPlover['individualCount'].fillna(1)
goldenPlover['individualCount'] = goldenPlover['individualCount'].astype(int)
goldenPlover = goldenPlover.dropna(subset=['year'])
goldenPlover['year'] = goldenPlover['year'].astype(int)
goldenPlover = goldenPlover.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
print(goldenPlover)

##############################################tensor flow
# for index, row in goldenPlover.iterrows():
#         print("work in progress")

# earth2000 = np.zeros(shape=(180, 360))
# earth2020 = np.zeros(shape=(180, 360))

# goldenPlover2000 = goldenPlover.query('2000 == year').copy()
# print(goldenPlover2000)
# for x in np.arange(0, goldenPlover2000.shape[0]):
#         earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] = earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2000.iloc[x]['individualCount']
#         print(x, " -> ", earth2000[goldenPlover2000.iloc[x]['decimalLatitude'] + 90, goldenPlover2000.iloc[x]['decimalLongitude'] + 180,])

# goldenPlover2019 = goldenPlover.query('2019 == year').copy()
# for x in np.arange(0, goldenPlover2019.shape[0]):
#         earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,] = earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,] + goldenPlover2019.iloc[x]['individualCount']
#         print(x, " -> ", earth2020[goldenPlover2019.iloc[x]['decimalLatitude'] + 90, goldenPlover2019.iloc[x]['decimalLongitude'] + 180,])

# tensor2000 = torch.from_numpy(earth2000)
# tensor2020 = torch.from_numpy(earth2020)

# print(torch.lerp(tensor2000, tensor2020, 0.5))
# npinterpolate = torch.lerp(tensor2000, tensor2020, 0.5).numpy()
# np.savetxt("tensor1.csv", npinterpolate, delimiter=",")

# # plt.scatter(npinterpolate)
# # plt.show()
# plt.cla()

# # plt.imsave("tensor.png",tensor2020, )
# print(earth2020)
# xs = earth2020[:,0] # Selects all xs from the array
# ys = earth2020[:,1]  # Selects all ys from the array
# plt.scatter(x=xs, y=ys)
#####################################################################

# Cluster Identification

# y_kmeans = kmeans.predict(X)
# model = KMeans(n_clusters=5).fit(X)


# arr = np.zeros(shape = (goldenPlover2019.shape[0], 2))
# for i in np.arange(0, arr.shape[0]):
#         arr[i,0] = goldenPlover2019['decimalLatitude'].iloc[i]
#         arr[i,1] = goldenPlover2019['decimalLongitude'].iloc[i]
# xs = arr[:,0] # Selects all xs from the array
# ys = arr[:,1]  # Selects all ys from the array
# # plt.xlim(-180, 180)
# # plt.ylim(-90,90)

# # kmeans = KMeans(n_clusters=3)
# # kmeans.fit(arr)
# # kmeans.labels_

# clustering = DBSCAN(eps = 5, min_samples = 5).fit(arr)
# clustering.labels_

# plt.scatter(x = xs, y = ys, s = 50, c = clustering.labels_)
# # plt.show()

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


# Cluster Code (not specific to American Golden Plover) including all migratory birds
listOfBirds = ['plovercsv.csv', 'pectoralSandpiperUnfiltered.csv', 'swainsonHawk.csv', 'whiteRumpedRaw.csv', 'forktailedUnfiltered.csv',]
listOfNames = ['American Golden Plover', 'Pectoral Sandpiper', 'Swainson\'s Hawk', 'White-Rumped Sandpiper', 'Fork-tailed Flycatcher']
def clusterID(minYear, maxYear, birdList, birdNames):
        for idx, x in enumerate(birdList):
                testImages = []
                birdDataFrame = pd.read_csv("{}".format(x))
                birdDataFrame = birdDataFrame[['decimalLongitude', 'decimalLatitude', 'individualCount', 'year']]
                birdDataFrame['decimalLongitude'] = birdDataFrame['decimalLongitude'].fillna(0)
                birdDataFrame['decimalLatitude'] = birdDataFrame['decimalLatitude'].fillna(0)
                birdDataFrame['decimalLongitude'] = birdDataFrame['decimalLongitude'].astype(float)
                birdDataFrame['decimalLatitude'] = birdDataFrame['decimalLatitude'].astype(float)
                birdDataFrame['individualCount'] = birdDataFrame['individualCount'].fillna(1)
                birdDataFrame['individualCount'] = birdDataFrame['individualCount'].astype(int)
                
                birdDataFrame = birdDataFrame.dropna(subset=['individualCount'])
                birdDataFrame = birdDataFrame.dropna(subset=['year'])
                
                birdDataFrame['year'] = birdDataFrame['year'].astype(int)
                birdDataFrame = birdDataFrame.query('8 > decimalLatitude >= -60 & -90 <= decimalLongitude <= -30 & individualCount < 1000')
                l21, l22, l23, B31, B32, B33 = [], [], [], [], [], []
                for y in np.arange(minYear, maxYear):
                        year = y
                        plt.cla()
                        temp = birdDataFrame.query('@y == year').copy()
                        a = np.zeros(shape = (temp.shape[0], 2))
                        for j in np.arange(0, a.shape[0]):
                                a[j,0] = temp['decimalLatitude'].iloc[j]
                                a[j,1] = temp['decimalLongitude'].iloc[j]
                        xs = a[:,0] # Selects all xs from the array
                        ys = a[:,1]  # Selects all ys from the array
                        latlon = temp[['decimalLatitude', 'decimalLongitude']].to_numpy()
                        # print(latlon)
                        # print("A", a)


                        # Code that identifies the clustering using the DBSCAN algorithm
                        clustering = DBSCAN(eps = 1, min_samples = 5).fit(latlon, y = None, sample_weight = temp['individualCount'].tolist())
                        clustering.labels_
                        clusterLabels = clustering.labels_ 
                        unique_labels = set(clusterLabels)
                        #print(unique_labels)
                        num_clusters = len(set(clusterLabels))
                        print(num_clusters)
                        n_clusters_ = len(set(clusterLabels)) - (1 if -1 in clusterLabels else 0)
                        n_noise_ = list(clusterLabels).count(-1)
                        
                        print(f'Estimated number of clusters: {n_clusters_}')

                        print(f'Estimated number of noise points: {n_noise_}')
                        labels, counts = np.unique(clusterLabels[clusterLabels>=0], return_counts=True)
                        print("Top Three Clusters: ", labels[np.argsort(-counts)[:3]])
                        #print("1,2,3", labels[np.argsort(-counts)[0]], labels[np.argsort(-counts)[1]])
                        # print(metrics.silhouette_score(clustering, clustering['labels']))
                        
################################################################################## OLD DEGREE TO KM.
                        # kms_per_radian = 6371.0088
                        # epsilon = 4 / kms_per_radian
                        # db = DBSCAN(eps=epsilon, min_samples=15, algorithm='ball_tree', metric='haversine').fit(np.radians(latlon))
                        # cluster_labels = db.labels_
                        # clusterList = cluster_labels.tolist()
                        # print(clusterList)

                        # num_clusters = len(set(cluster_labels))
                        # clusters = pd.Series([latlon[cluster_labels == n] for n in range(num_clusters)])
                        # print('Number of clusters: {}'.format(num_clusters))
###################################################################################     
###################################################################################
                        # PROBABLY USELESS... KEEP JUST IN CASE
                        def get_centermost_point(cluster):
                                centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
                                centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
                                return tuple(centermost_point)
                        # centermost_points = clusters.map(get_centermost_point)
                        # centermost_points = get_centermost_point(clusters[1])
                        # print(centermost_points)
##################################################################################
                        

                        #kmeans Identification
                        # clusteringKMeans = KMeans(n_clusters = 3)
                        # # cLMean = clusteringKMeans.labels_
                        # clusteringKMeans.fit(a)
                        # print(clusteringKMeans.fit(a).cluster_centers_)
                        
                        # Mean-shift testing
                        

                        # Plotting Code (DBSCAN)
                        clusterLabels = clusterLabels.tolist()
                        print(latlon)
                        index_list = [index for (index, num) in enumerate(clusterLabels) if num == 0]
                        testfil = [latlon[i] for i in index_list]
                        meanN = [sum(x)/len(x) for x in zip(*testfil)]
                        print(testfil)
                        print(meanN)

                        numC = len(set(clusterLabels))
                        print(numC)
                        print(clusterLabels)
                        if numC > 1: 
                                one = labels[np.argsort(-counts)[0]]
                                indOne = [index for (index, num) in enumerate(clusterLabels) if num == one]
                                oneList = [latlon[i] for i in indOne]
                                meanOne = [sum(x)/len(x) for x in zip(*oneList)]
                                l21.append([year, meanOne])
                        if numC > 2: 
                                two = labels[np.argsort(-counts)[1]]
                                indTwo = [index for (index, num) in enumerate(clusterLabels) if num == two]
                                twoList = [latlon[i] for i in indTwo]
                                meanTwo = [sum(x)/len(x) for x in zip(*twoList)]
                                l22.append([year, meanTwo])
                        if numC > 3: 
                                three = labels[np.argsort(-counts)[2]]
                                indThree = [index for (index, num) in enumerate(clusterLabels) if num == three]
                                threeList = [latlon[i] for i in indThree]
                                meanThree = [sum(x)/len(x) for x in zip(*threeList)]
                                l23.append([year, meanThree])
                        
                        for i in range(len(clusterLabels)):
                                # Noise
                                if clusterLabels[i] == -1:
                                        clusterLabels[i] = 'grey'
                                        continue
                                # Largest Cluster
                                if numC > 1:
                                        if clusterLabels[i] == one:
                                                clusterLabels[i] = '#a83e44'
                                                continue
                                # Second Largest Cluster
                                if numC > 2: 
                                        if clusterLabels[i] == two:
                                                clusterLabels[i] = '#201e1e'
                                                continue
                                # Third Largest Cluster
                                if numC > 3: 
                                        if clusterLabels[i] == three:
                                                clusterLabels[i] = '#4a68c3'
                                                continue
                                clusterLabels[i] = 'green' 


                        plt.title(f"{birdNames[idx]} DBSCAN Clustering - {y}")
                        ax1 = sb.scatterplot(x = xs, y = ys, s = 100, c = clusterLabels)
                        plt.legend(labels=['legendEntry1', 'legendEntry2', 'legendEntry3'])
                        custom_lines = [Line2D([0], [0],marker='o', color="#a83e44", label='Scatter', lw = 0),Line2D([0], [0],marker='o',color="#201e1e", label='Scatter', lw = 0), Line2D([0], [0], marker='o',color="#4a68c3", label='Scatter', lw = 0), Line2D([0], [0], marker='o',color="green", label='Scatter', lw = 0), Line2D([0], [0], marker='o',color="grey", label='Scatter', lw = 0)]
                        ax1.legend(custom_lines, ['1st Largest', '2nd Largest', '3rd Largest', 'Additional', 'Noise'], title = "Cluster Identification", )
                        sb.move_legend(ax1, "upper right")
                        ax1.set(xlabel="Latitude", ylabel="Longitude")
                        ax1.grid(color = 'grey', linestyle = "--")
                        plt.ylim([-95, -30])
                        plt.xlim([-60,15])
                        plt.savefig("{}ClusterIDTest{}.png".format(birdNames[idx],y))
                        
                        print("{} {} Complete.".format(birdNames[idx], y))
                        testImages.append(f"{birdNames[idx]}ClusterIDTest{y}.png")
                        

                        # print(l21)
                        # l21 = np.add(l21, [30,60])
                        # print(l21)
                        # dist21 = math.sqrt(((l21[0])**2) + ((l21[1])**2))
                        # print(dist21)

                        # Plotting Code (KMEANS CENTROIDS)
                        # plt.cla()
                        # plt.scatter(clusteringKMeans.cluster_centers_[:, 0], clusteringKMeans.cluster_centers_[:, 1], s=100, c='black')
                        # plt.title("{} KMean Centroids {}".format(birdNames[idx], y))                       
                        # plt.ylim([-95, -30])
                        # plt.xlim([-60,15])
                        # # plt.scatter(x = xs, y = ys, s = 50, c = cLMean)
                        # plt.savefig("{}kmeanCentroid{}.png".format(birdNames[idx],y))
                        # print("{} {} Complete.".format(birdNames[idx], y))
                        
                        
                        #New tes

                # Distance Plotting
                print(l21)
                print(l22)
                print(l23)
                
                for elem in l21:
                        points = elem[1]
                        dist21 = getDistanceFromLatLonInKm(points[0], points[1], -30, -60)
                        B31.append([elem[0], dist21])
                for elem in l22:
                        points = elem[1]
                        dist22 = getDistanceFromLatLonInKm(points[0], points[1], -30, -60)
                        B32.append([elem[0], dist22])
                for elem in l23:
                        points = elem[1]
                        dist23 = getDistanceFromLatLonInKm(points[0], points[1], -30, -60)
                        B33.append([elem[0], dist23])
                print(f"bb:{B31}")
                # for elem in l21:
                #         newb = np.add(elem[1], [30,60])
                #         dist21 = math.sqrt(((newb[0])**2) + ((newb[1])**2))
                #         B32.append([elem[0], dist21])
                # print(f"cc:{B32}")
                # for elem in l22:
                #         newb = np.add(elem[1], [30,60])
                #         dist22 = math.sqrt(((newb[0])**2) + ((newb[1])**2))
                #         B32.append([elem[0], dist22])     
                # for elem in l23:
                #         newb = np.add(elem[1], [30,60])
                #         dist23 = math.sqrt(((newb[0])**2) + ((newb[1])**2))
                #         B33.append([elem[0], dist23])    
                       
                plt.cla()
                # print(f"YEARS:{B31[0]}")
                # input("ENTER.")
                lis1, lis2, lis3 = list(zip(*B31)),list(zip(*B32)),list(zip(*B33))

                # input("ENTER.")
                frameOfDistance1 = pd.DataFrame(columns = ['year', 'dist', 'Cluster Size'])
                frameOfDistance1['year'] = lis1[0]
                frameOfDistance1['dist'] = lis1[1]
                frameOfDistance1['Cluster Size'][0:len(lis1[0])] = "1st Largest"
                frameOfDistance2 = pd.DataFrame(columns = ['year', 'dist', 'Cluster Size'])
                frameOfDistance2['year'] = lis2[0]
                frameOfDistance2['dist'] = lis2[1]
                frameOfDistance2['Cluster Size'][0:len(lis2[0])] = "2nd Largest"
                frameOfDistance3 = pd.DataFrame(columns = ['year', 'dist', 'Cluster Size'])
                frameOfDistance3['year'] = lis3[0]
                frameOfDistance3['dist'] = lis3[1]
                frameOfDistance3['Cluster Size'][0:len(lis3[0])] = "3rd Largest"
                
                frameOfDistanceFinal = pd.concat([frameOfDistance1, frameOfDistance2, frameOfDistance3], ignore_index= True)
                print(frameOfDistanceFinal)

                # input("ENTER.")
                zip(*B31)
                zip(*B32)
                zip(*B33)
                # plt.plot(*zip(*B31), color = "green")
                # plt.plot(*zip(*B32), color = "yellow")
                # plt.plot(*zip(*B33), color = "red")
                # print(*zip(*B31))
                # ax = sb.lineplot(*zip(*B31), color = "#261C2C")
                # ax = sb.lineplot(*zip(*B32), color = "#3E2C41")
                # ax = sb.lineplot(*zip(*B33), color = "#5C527F")

                # plt.xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018])
                # ax.set(xlabel="Year", ylabel="Distance from SAA (km)")
                # plt.title(f"{birdNames[idx]} Cluster Distance Analysis (2000 - 2020)")
                # plt.show()
                
                #New test with hues
                #sb.set_theme(style='darkgrid')
                plt.cla()
                ax = sb.lineplot(data = frameOfDistanceFinal, x = "year", y = "dist", hue = "Cluster Size", palette="icefire_r", legend='full')
                plt.xticks([2000, 2003, 2006, 2009, 2012, 2015, 2018])
                plt.yticks([0, 1000, 2000, 3000, 4000, 5000])
                ax.set(xlabel="Year", ylabel="Distance from SAA (km)")
                ax.grid(color = 'grey', linestyle = "--")
                plt.title(f"{birdNames[idx]} Cluster Distance Analysis (2000 - 2020)")
                sb.move_legend(ax, "upper right")
                #plt.show()
                
                
                # plt.plot(np.arange(2000,2020), B32)
                # plt.plot(np.arange(2000,2020), B33)
                plt.savefig(f"_{birdNames[idx]}Cluster")
                gifMaker(testImages, f"{birdNames[idx]}Clusters.gif", 0.2)

                # centroid_of_cluster_0 = np.mean(points_of_cluster_0, axis=0) 
                # plt.scatter(x = xs, y = ys, s = 50, c = clusteringKMeans.labels_)



clusterID(2000, 2020, listOfBirds, listOfNames)