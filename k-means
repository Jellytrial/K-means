@@ -0,0 +1,108 @@
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:58:22 2018

@author: Jelly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys


sys.setrecursionlimit(1000)
dataSet = pd.read_csv('input.csv')
value=dataSet.values

# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
    return np.sqrt(np.sum(np.power(vector2 - vector1, 2)))

# init centroids with random samples  
def initCentroids(dataSet, k):
    count = 0
    while (count < 100):
        numSamples, dim = dataSet.shape  
        centroids = np.zeros((k, dim))
        for i in range(k):  
            index = int(random.uniform(0, numSamples))  
            centroids[i, :] = dataSet[index, :]
        count = count+1    
    print('initcentroids:\n', centroids)
    return centroids  

# k-means cluster  
def kmeans(dataSet, k):  
    numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = np.mat(np.zeros((numSamples, 2)))  
    clusterChanged = True  
  
    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)  
  
    while clusterChanged:  
        clusterChanged = False  
        ## for each sample  
        for i in range(numSamples):  
            minDist  = 100.0  
            minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if np.all(distance < minDist):  
                    minDist  = distance  
                    minIndex = j  
              
            ## step 3: update its cluster  
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2  
  
        ## step 4: update centroids  
        for j in range(k):  
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)
    print('centorids: \n', centroids)
    return centroids, clusterAssment

# compulte the minimum of objective fuction
def J(centroids, dataSet):   
    sum_cluster_distance = 0  
    for i in range(k):  
        for j in range(k):
            sum_cluster_distance = sum_cluster_distance + euclDistance(centroids[j, :], dataSet[i, :]) 
    return sum_cluster_distance  

# show your cluster only available with 2-D data  
def showCluster(dataSet, k, centroids, clusterAssment):  
    numSamples, dim = dataSet.shape  
    if dim != 2:   
        return 1  
  
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):   
        return 1  
  
    # draw all samples  
    for i in range(numSamples):  
        markIndex = int(clusterAssment[i, 0])  
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex], alpha = 0.6)  
  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
  
    plt.show() 


dataSet = np.mat(dataSet) 
k = 5 
centroids, clusterAssment = kmeans(dataSet, k)
print('minimum of objective function: \n', J(centroids, dataSet))
showCluster(dataSet, k, centroids, clusterAssment)
