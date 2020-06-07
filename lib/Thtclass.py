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

class tht:
    def __init__(self, t, r, matrix, window, interval):
        self.t = t  # time
        self.r = r  # range
        self.matrix = matrix  ###in version of (range  x   time)
        self.window = int(window)
        self.interval = int(interval)
        self.lenT = len(self.t)
        self.lenR = len(self.r)

    def convertTime(self):
        self.date = []
        for stamp in self.t:
            stamp = time.localtime(stamp)
            self.date.append(time.strftime("%Y-%m-%d %H:%M:%S", stamp))

    def __pr2G(self):  # caculate the graident profile in range scales
        self.pr2G = self.matrix[:, :].astype(float)
        for j in range(self.lenT):
            for i in range(self.lenR):
                if i == 0:
                    self.pr2G[i, j] = (self.matrix[i + 1, j] - self.matrix[i, j]) / 10
                elif i == (self.lenR - 1):
                    self.pr2G[i, j] = (self.matrix[i, j] - self.matrix[i - 1, j]) / 10
                else:
                    self.pr2G[i, j] = (self.matrix[i + 1, j] - self.matrix[i - 1, j]) / 20

    def __logpr2G(self):  # caculate the graident profile of log data in range scales
        self.logpr2G = np.log(self.matrix[:, :])
        for j in range(self.lenT):
            for i in range(self.lenR):
                if i == 0:
                # if i !=(self.lenR - 1):
                    self.logpr2G[i, j] = (self.matrix[i + 1, j] - self.matrix[i, j]) / 10
                elif i == (self.lenR - 1):
                    self.logpr2G[i, j] = (self.matrix[i, j] - self.matrix[i - 1, j]) / 10
                else:
                    self.logpr2G[i, j] = (self.matrix[i + 1, j] - self.matrix[i - 1, j]) / 20

    def __var(self):  # caculate the variance of pr2
        self.var = np.zeros((self.lenR, self.lenT))
        if (self.interval % 2) != 0:  # if the interval is odd
            halfInterval = (self.interval - 1) // 2
            for i in range(self.lenT):
                if i < halfInterval:
                    self.var[:, i] = self.matrix[:, 0:i + halfInterval + 1].var(axis=1)
                elif i > (self.lenT - halfInterval):
                    self.var[:, i] = self.matrix[:, i - halfInterval:self.lenT].var(axis=1)
                else:
                    self.var[:, i] = self.matrix[:, i - halfInterval:i + halfInterval].var(axis=1)
        else:  # if the interval is even
            halfInterval1 = self.interval // 2
            halfInterval2 = self.interval // 2 - 1
            for i in range(self.lenT):
                if i < halfInterval1:
                    self.var[:, i] = self.matrix[:, 0:i + halfInterval2 + 1].var(axis=1)
                elif i > (self.lenT - halfInterval2 - 1):
                    self.var[:, i] = self.matrix[:, i - halfInterval2:self.lenT].var(axis=1)
                else:
                    self.var[:, i] = self.matrix[:, i - halfInterval1:i + halfInterval2].var(axis=1)

    def avergae(self):  # caculate the height reference for the first start position in each goup
        if (self.lenT % self.window) != 0:
            print("The windosize false")
            return 0
        else:
            self.avvar = np.zeros((self.lenR, self.lenT // self.window))
            self.avlogpr2G = np.zeros((self.lenR, self.lenT // self.window))
            self.hrefIndex = np.zeros(self.lenT // self.window)
            self.__logpr2G()
            self.__var()
            for i in range(self.lenT // self.window):
                self.avvar[:, i] = self.var[:, i * self.window:i * self.window + self.window].mean(axis=1)
                self.avlogpr2G[:, i] = self.logpr2G[:, i * self.window:i * self.window + self.window].mean(axis=1)
                # need to delete the layer over 4000
                # self.__hrefIndex[i]=(np.where(self.avvar[:,i]==np.max(self.avvar[:,i]))[0][0]+np.where(self.avlogpr2G[:,i]==np.min(self.avlogpr2G[:,i]))[0][0])/2
                #self.hrefIndex[i] = (np.where(self.avvar[0:300, i] == np.max(self.avvar[0:300, i]))[0][0] +
                                    #np.where(self.avlogpr2G[0:300, i] == np.min(self.avlogpr2G[0:300, i]))[0][0]) / 2
                self.hrefIndex[i]=self.findHref(self.avlogpr2G[:,i],self.avvar[:,i])

    def findHref(self,gs,var):
        #find min gs
        gsMin=np.nan
        for i in range(self.lenR):
            if i>5 and i<len(gs)-5:
                if gs[i]<gs[i-1] and gs[i]<gs[i+1]:
                    halfInter = 20
                    startPos = i - halfInter
                    endPos = i + halfInter
                    if startPos < 0:
                        startPos = 0
                    if endPos > len(gs):
                        endPos = len(gs)
                    locMin = np.where(gs[startPos:endPos] == np.min(gs[startPos:endPos]))[0][
                                 0] + startPos
                    if (locMin == i):
                        gsMin=i
                        break
        #find max var
        varMax=np.nan
        for i in range(self.lenR):
            if i>5 and i<len(var)-5:
                if var[i]>var[i-1] and var[i]>var[i+1]:
                    halfInter = 20
                    startPos = i - halfInter
                    endPos = i + halfInter
                    if startPos < 0:
                        startPos = 0
                    if endPos > len(gs):
                        endPos = len(gs)
                    locMax = np.where(var[startPos:endPos] == np.max(var[startPos:endPos]))[0][
                                 0] + startPos
                    if (locMax == i):
                        varMax=i
                        break
        if(np.isnan(gsMin) or np.isnan(varMax)):
            return np.nan
        else:
            #print((varMax+gsMin)/2)
            return (varMax+gsMin)/2


    def variance(self):  # caculate the varience reference at the height href
        ret = self.avergae()
        if ret == 0:
            return 0
        self.varref = np.zeros(self.lenT // self.window)
        for i in range(self.lenT // self.window):
            index = int(self.hrefIndex[i])
            varVr = self.var[index, i * self.window:i * self.window + self.window].var()
            varLogPr2 = self.logpr2G[index, i * self.window:i * self.window + self.window].var()
            self.varref[i] = math.sqrt(varVr + varLogPr2)

    def mlh(self):  # caculate the mixing layer height
        ret = self.variance()
        if ret == 0:
            return 0
        self.mixHeight = np.zeros((self.lenT))
        self.rangelow = np.zeros((self.lenT))
        self.rangehigh = np.zeros((self.lenT))
        for i in range(self.lenT // self.window):
        #for i in range(0,1,1):
            tempHref = self.hrefIndex[i]
            tempVarref = self.varref[i] / 10
            if(tempVarref>300):
                tempVarref=60
            for j in range(self.window):
            #for j in range(0,1,1):
                rangeLow = math.ceil(0.85 * (tempHref - tempVarref))  # find the bigger int
                # print('rangelow is:')
                # print(rangeLow)
                rangeHigh = math.floor(1.15 * (tempHref + tempVarref))  # find the smaller int
                # print('rangehigh is')
                # print(rangeHigh)
                if rangeHigh == rangeLow:
                    mixIndex = rangeLow
                else:
                    if rangeLow < 0:
                        rangeLow = 0
                    # if rangeHigh>=450:
                    #     rangeHigh=449
                    if rangeHigh >= 250:
                        rangeHigh = 249
                    varGheight_index = np.where(self.var[rangeLow:rangeHigh + 1, i * self.window + j] == max(
                        self.var[rangeLow:rangeHigh + 1, i * self.window + j]))[0][0] + rangeLow
                    # print('varheigth index is ')
                    # print(varGheight_index)
                    logpr2Gheight_index = np.where(self.logpr2G[rangeLow:rangeHigh + 1, i * self.window + j] == min(
                        self.logpr2G[rangeLow:rangeHigh + 1, i * self.window + j]))[0][0] + rangeLow
                    # print('logpr2height index is ')
                    # print(logpr2Gheight_index)
                    mixIndex = (varGheight_index + logpr2Gheight_index) / 2
                    # print('mixindex')
                    # print(mixIndex)
                self.mixHeight[i * self.window + j] = mixIndex * 10
                self.rangelow[i * self.window + j] = rangeLow
                self.rangehigh[i * self.window + j] = rangeHigh
                # update the new ref
                tempHref = mixIndex
                # tempVarref=math.sqrt(self.logpr2G[int(mixIndex),i:i+self.window].var()+self.var[int(mixIndex),i:i+self.window].var())
                # tempVarref = math.sqrt(
                #     self.logpr2G[int(mixIndex), i * self.window:i * self.window + self.window].var() + self.var[
                #                                                                                        int(mixIndex),
                #                                                                                        i * self.window:i * self.window + self.window].var())
                tempVarref =self.logpr2G[int(mixIndex), i * self.window:i * self.window + self.window].std() + self.var[
                                                                                                       int(mixIndex),
                                                                                                       i * self.window:i * self.window + self.window].std()
                tempVarref/=10
                if (tempVarref > 300):
                    tempVarref = 100
                # print('tempVarref is ')
                # print(tempVarref)