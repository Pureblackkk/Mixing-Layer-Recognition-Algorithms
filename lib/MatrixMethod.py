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
from sklearn import preprocessing
import datetime

class matrixMethod:
    def __init__(self,matrix):
        self.matrix=matrix
        if np.ndim(matrix)!=1:
            self.lenT=len(matrix[0,:])
            self.lenR=len(matrix[:,0])

    def wavSmooth(self,threshold,levels):
        smoothMatrix = np.zeros((self.lenR, self.lenT))
        w = pywt.Wavelet('db8')
        maxlev = pywt.dwt_max_level(self.lenR, w.dec_len)
        #threshold = 0.8
        for i in range(self.lenT):
            coeffs = pywt.wavedec(self.matrix[:, i], 'db8', level=levels)
            for j in range(1, len(coeffs)):
                coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
                smoothMatrix[:, i] = pywt.waverec(coeffs, 'db8')
        return smoothMatrix

    def gausSmooth(self):
        smoothMatrix = np.zeros((self.lenR, self.lenT))
        for i in range(self.lenR):
            for j in range(self.lenT):
                if i==0 and j==0:
                    smoothMatrix[i,j]=(self.matrix[i,j]+self.matrix[i,j+1]+self.matrix[i+1,j])/3
                elif i==0 and j==self.lenT-1:
                    smoothMatrix[i, j] = (self.matrix[i, j] + self.matrix[i, j - 1] + self.matrix[i + 1, j]) / 3
                elif i==0:
                    smoothMatrix[i, j]=(self.matrix[i,j]+self.matrix[i,j-1]+self.matrix[i,j+1]+self.matrix[i+1,j])/4
                elif j==0 and i!=self.lenR-1:
                    smoothMatrix[i,j]=(self.matrix[i,j]+self.matrix[i-1,j]+self.matrix[i+1,j]+self.matrix[i,j+1])/4
                elif j==self.lenT-1 and i!=self.lenR-1:
                    smoothMatrix[i, j] = (self.matrix[i, j] + self.matrix[i - 1, j] + self.matrix[i + 1, j] +
                                          self.matrix[i, j - 1]) / 4
                elif i==self.lenR-1 and j==0:
                    smoothMatrix[i, j]= (self.matrix[i,j]+self.matrix[i-1,j]+self.matrix[i,j+1])/3
                elif i==self.lenR-1 and j==self.lenT-1:
                    smoothMatrix[i, j] = (self.matrix[i, j] + self.matrix[i - 1, j] + self.matrix[i, j - 1]) / 3
                elif i==self.lenR-1:
                    smoothMatrix[i, j] = (self.matrix[i, j] + self.matrix[i - 1, j] + self.matrix[i, j + 1] +
                                          self.matrix[i, j - 1]) / 4
                else:
                    smoothMatrix[i, j] = (self.matrix[i, j] + self.matrix[i - 1, j] + self.matrix[i, j + 1] +
                                          self.matrix[i, j - 1] +self.matrix[i+1,j]) / 5
        return smoothMatrix





    def standard(self):
        standardMatrix = np.zeros((self.lenR, self.lenT))
        for i in range(self.lenT):
            standardMatrix[:,i]=preprocessing.scale(self.matrix[:,i])
        return standardMatrix

    def converto1(self):
        convertMatrix=np.zeros((self.lenR, self.lenT))
        min_max_scaler = preprocessing.MinMaxScaler()
        convertMatrix = min_max_scaler.fit_transform(self.matrix.transpose()).transpose()
        return convertMatrix


    def deRange(self):
        deRangeMatrix=np.zeros((self.lenR, self.lenT))
        for j in range(self.lenT):
            for i in range(self.lenR):
                corR=((i+1)*10)**2
                deRangeMatrix[i,j]=self.matrix[i,j]/corR
        return deRangeMatrix

    def unixToDate(self):
        dateList=[]
        for member in self.matrix:
            date=datetime.datetime.fromtimestamp(member)
            dateList.append(date.strftime("%Y-%m-%d %H:%M:%S"))
        return dateList

