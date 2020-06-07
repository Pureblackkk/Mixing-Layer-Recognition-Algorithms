import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import pywt.data
from sklearn.cluster import KMeans
import os
import os.path
import re
import sys
import codecs

class BvPrepocess:
    def __init__(self, matrix):
        self.matrix = matrix
        self.lenT = len(matrix[0, :])
        self.lenR = len(matrix[:, 0])

    def runningAvg(self, length):
        # caculate the running average
        newLenR = self.lenR - (length - 1)
        newMatrix = np.zeros((newLenR, self.lenT))
        for j in range(0, self.lenT):
            for i in range(0, newLenR):
                newMatrix[i, j] = self.matrix[i:i + length, j].mean()
        return newMatrix


class Wavelet:
    def __init__(self, matrix, levels, lowPercent, highPercent):
        self.matrix = matrix
        self.lenT = len(matrix[0, :])
        self.lenR = len(matrix[:, 0])
        self.levels = levels
        self.lowPercent = lowPercent
        self.highPercent = highPercent

    def dwtMaxPoint(self):
        localMaxlist = []
        for j in range(self.lenT):
            cofList = pywt.wavedec(self.matrix[:, j], 'haar', mode='smooth', level=self.levels)
            # maxList means in single time scale the height indicateing possible abrupt
            maxList = self.findMax(cofList[1:-1])
            # add this time's maxList into the total list
            localMaxlist.append(maxList)
        return localMaxlist

    def findMax(self, scaleList):
        maxList = []
        for singleList in scaleList:
            listLen = len(singleList)
            lowPass = np.percentile(singleList, self.lowPercent)
            highPass = np.percentile(singleList, self.highPercent)
            for i in range(listLen):
                if (singleList[i] > highPass):
                    if (i != 0 and i != (listLen - 1)):
                        if (singleList[i] > singleList[i + 1] and singleList[i] > singleList[i - 1]):
                            # there's a problem about the corresbonding height
                            tempHeight = 10 * (i + 1) * self.lenR / listLen
                            maxList.append(tempHeight)
        return maxList


# using K-means clustering methods to find the special points
class Cluster:
    def __init__(self, clusterList, clusterNumber, groupNumber):
        self.clusterList = clusterList
        self.clusterNumber = clusterNumber
        self.groupNumber = groupNumber

    def cluster(self):
        # convert list into groups
        if (len(self.clusterList) % self.groupNumber != 0):
            print("Wrong group numbers!")
            return -1
        else:
            step = self.groupNumber
            self.reList = []
            for i in range(0, len(self.clusterList), step):
                temp = []
                for j in self.clusterList[i:i + step]:
                    for k in j:
                        temp.append(k)
                self.reList.append(temp)
        totalNumber = len(self.clusterList) // self.groupNumber
        finalList = []
        for i in range(totalNumber):
            clusterRes = KMeans(n_clusters=self.clusterNumber)
            clusterRes.fit(np.array(self.reList[i]).reshape(-1, 1))
            temp = clusterRes.cluster_centers_
            temp = temp.flatten()
            temp = np.sort(temp)
            finalList.append(temp)
        return finalList
        # sse=[]
        # for k in range(1,9):
        #     estimator = KMeans(n_clusters=k)  # 构造聚类器
        #     estimator.fit(np.array(self.reList[45]).reshape(-1,1))
        #     sse.append(estimator.inertia_)
        # x = range(1,9)
        # plt.xlabel('k')
        # plt.ylabel('sse')
        # plt.plot(x,sse,'o-')
        # plt.show()

# class Bravo:
#   def __init__():