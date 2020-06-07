#import source file
import lib.Bravoclass as bravo
from lib.Thtclass import tht
from lib.MatrixMethod import matrixMethod
import lib.Readfile as read
import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import math
from matplotlib import font_manager as fm, rcParams
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


class STRAT:
    def __init__(self, t, r, matrix, scale):
        self.matrix = matrix
        self.lenT = len(matrix[0, :])
        self.lenR = len(matrix[:, 0])
        self.t = t  # time
        self.r = r  # range
        self.scale = scale  # scale

    def smooth(self):
        self.smoothMatrix = np.zeros((self.lenR, self.lenT))
        threshold = 0.8
        for i in range(self.lenT):
            coeffs = pywt.wavedec(self.matrix[:, i], 'db8', level=4)
            for j in range(len(coeffs) - 1):
                coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
            self.smoothMatrix[:, i] = pywt.waverec(coeffs, 'db8')

    def smooth2(self, matrix):
        col = len(matrix[:, 0])
        row = len(matrix[0, :])
        smoothMatrix = np.zeros((col, row))
        threshold = 0.1
        for i in range(col):
            coeffs = pywt.wavedec(matrix[i, :], 'haar', level=2)
            for j in range(len(coeffs) - 1):
                coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
            smoothMatrix[i, :] = pywt.waverec(coeffs, 'haar')
        return smoothMatrix

    def smooth3(self, matrix):
        col = len(matrix[:, 0])
        row = len(matrix[0, :])
        smoothMatrix = np.zeros((col, row))
        for i in range(col):
            for j in range(row):
                if (j != 0 and j != (row - 1)):
                    smoothMatrix[i, j] = (matrix[i, j + 1] + matrix[i, j] + matrix[i, j - 1]) / 3
        return smoothMatrix

    def getThreshold(self, line):
        return np.std(line[400:450])

    def drawMap(self, number, ridge, wavelet):
        # prepare the line for drawing
        font = {'family': 'SimHei',
                'weight': 'bold',
                'size': 13
                }
        font_1 = {
            'size': 12
        }
        xLine = []
        yLine = []
        for line in ridge:
            x = []
            y = []
            for pos in line:
                x.append(pos[1])
                y.append(pos[0])
            xLine.append(x)
            yLine.append(y)
        # draw the map
        coef, freqs = pywt.cwt(self.matrix[:, number], np.arange(1, self.scale, 1), wavelet)
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.matrix[:, number])
        ax1.set_xticks(np.arange(0, 500, 100))
        ax1.set_xlim(0, 450)
        ax1.set_yticks([])
        ax1.set_ylabel('RCS(a.u.)', font)
        ax1.set_xticklabels(['0', '1', '2', '3', '4'])
        ax1.set_xlabel('高度(km)', font)

        ax2 = plt.subplot(2, 1, 2)
        ax2.matshow(coef)
        ax2.set_xticks(np.arange(0, 500, 100))
        ax2.set_yscale("log",subsy=[0])
        ax2.set_xticks([])
        ax2.set_xticklabels(['0', '1', '2', '3', '4'])
        ax2.set_xlabel('(b)', font_1)
        lineNum = len(xLine)
        for i in range(lineNum):
            ax2.plot(xLine[i], yLine[i], color='black')
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_ylabel('ln(a)', font)
        plt.savefig('/Users/pureblack/Desktop/毕业论文/图/StratAss1', dpi=500, bbox_inches='tight')
        plt.show()

    def drawMap2(self,number, ridge, wavelet, mlh, hMol, hPar):
        font = {'family': 'SimHei',
                'weight': 'bold',
                'size': 13
                }
        font_1 = {
            'size': 12
        }
        xLine = []
        yLine = []
        for line in ridge:
            x = []
            y = []
            for pos in line:
                x.append(pos[1])
                y.append(pos[0])
            xLine.append(x)
            yLine.append(y)
        # draw the map
        coef, freqs = pywt.cwt(self.matrix[:, number], np.arange(1, self.scale, 1), wavelet)
        plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.matrix[:, number])
        ax1.set_xticks(np.arange(0, 500, 100))
        ax1.set_xlim(0, 450)
        ax1.set_yticks([])
        ax1.set_ylabel('RCS(a.u.)', font)
        ax1.set_xticklabels(['0', '1', '2', '3', '4'])
        ax1.set_xlabel('高度(km)', font)
        ax1.axvline(mlh,lw=2,ls='--',c='black')
        ax1.axvline(hPar,lw=2,ls='--',c='black')
        ax1.axvline(hMol,lw=2,ls='--',c='black')

        ax2 = plt.subplot(2, 1, 2)
        ax2.matshow(coef)
        ax2.set_xticks(np.arange(0, 500, 100))
        ax2.set_yscale("log", subsy=[0])
        ax2.set_xticks([])
        ax2.set_xticklabels(['0', '1', '2', '3', '4'])
        #ax2.set_xlabel('(b)', font_1)
        lineNum = len(xLine)
        for i in range(lineNum):
            ax2.plot(xLine[i], yLine[i], color='black')
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.set_ylabel('ln(a)', font)
        plt.savefig('/Users/pureblack/Desktop/毕业论文/图/StratAss4', dpi=500, bbox_inches='tight')
        plt.show()

    def cwtRidge(self, line, wavelet):
        self.coef, freqs = pywt.cwt(line, np.arange(1, self.scale), wavelet)
        self.coef = self.smooth3(self.coef)
        ridgeList = []
        for i in range(self.scale - 1):
            rowList = []
            for j in range(self.lenR):
                if (j != 0 and j != (self.lenR - 1)):
                    if ((self.coef[i, j] > self.coef[i, j + 1] and self.coef[i, j] > self.coef[i, j - 1]) or (
                            self.coef[i, j] < self.coef[i, j + 1] and self.coef[i, j] < self.coef[i, j - 1])):
                        # delete the wrong data
                        halfInter = 20
                        startPos = j - halfInter
                        endPos = j + halfInter
                        if startPos < 0:
                            startPos = 0
                        if endPos > len(self.coef[0, :]):
                            endPos = len(self.coef[0, :])
                        locMax = np.where(self.coef[i, startPos:endPos] == np.max(self.coef[i, startPos:endPos]))[0][
                                     0] + startPos
                        locMin = np.where(self.coef[i, startPos:endPos] == np.min(self.coef[i, startPos:endPos]))[0][
                                     0] + startPos
                        if (locMax == j or locMin == j):
                            if j < self.lenR - 2 and j > 1:
                                rowList.append([i, j])
            ridgeList.append(rowList)
        # insert the different ridge and judge the ridge
        ridge = []
        # initial the ridge and keep a posList
        for loc in ridgeList[6]:
            ridge.append([loc])
        # print("---"+str(2)+"---")
        # print('---ridegList---')
        # print(ridgeList[2])
        # keep a posList
        posList = range(len(ridgeList[6]))
        # keep a booleanList
        booleanList = np.ones(len(ridgeList[6]))

        for i in range(7, self.scale - 1, 1):
            # for i in range(7,60,1):
            # print("---"+str(i)+"---")
            # print('---ridegList---')
            # print(ridgeList[i])

            # equal to the origin layer
            if (len(ridgeList[i]) == len(ridgeList[i - 1])):
                for j in range(len(booleanList)):
                    if (booleanList[j]):
                        # mutiply=self.coef[i,ridgeList[i][j][1]]*self.coef[i-1,ridgeList[i-1][j][1]]
                        # if mutiply > 0:
                        num = int(np.sum(booleanList[0:j]))
                        position = posList[num]
                        mutiply = self.coef[i, ridgeList[i][j][1]] * self.coef[
                            ridge[position][-1][0], ridge[position][-1][1]]
                        if mutiply > 0:
                            ridge[position].append(ridgeList[i][j])
            # some line point reduce
            else:
                point = 0
                upLayer = len(ridgeList[i - 1])
                newposList = []
                newbooleanList = np.zeros(len(ridgeList[i]))
                k = 0
                for loc in ridgeList[i]:
                    # find the correct ridge
                    for j in range(point, upLayer):
                        distance = abs(loc[1] - ridgeList[i - 1][j][1])
                        # mutiply=self.coef[i,loc[1]]*self.coef[i-1,ridgeList[i-1][j][1]]
                        # if(distance<10 and mutiply>0):
                        if (distance < 10):
                            if (booleanList[j]):
                                num = int(np.sum(booleanList[0:j]))
                                position = posList[num]
                                mutiply = self.coef[i, loc[1]] * self.coef[
                                    ridge[position][-1][0], ridge[position][-1][1]]
                                if mutiply > 0:
                                    ridge[position].append(loc)
                                    point = j + 1
                                    # creat new posList
                                    newposList.append(position)
                                    newbooleanList[k] = 1
                                break
                    # #creat new posList
                    # newposList.append(position)
                    k = k + 1
                posList = newposList
                booleanList = newbooleanList
            # print('---posList---')
            # print(posList)
            # print('---booleanList---')
            # print(booleanList)
        return ridge

    def applyThreshold(self, line, threshold, ridge):
        # print(threshold)
        # define the data type
        peakType = np.dtype({
            'names': ['number', 'peakHeight', 'baseHeight', 'topHeight', 'average'],
            'formats': ['i', 'i', 'i', 'i', 'f']}, align=True)

        # find the peak number
        peakList = []
        averageList = []
        for i in range(len(ridge)):
            # caculate the average
            # print('-----'+str(i)+'-----')
            # print(ridge[i][0][1])
            average = 0
            for loc in ridge[i]:
                # print(self.coef[loc[0],loc[1]])
                average += self.coef[loc[0], loc[1]]
            average /= len(ridge[i])
            # print('-----')
            # print(average)
            if (average) > 0:
                peakList.append(i)
            averageList.append(average)

        # add the peak info
        self.peak = np.zeros(len(peakList), dtype=peakType)
        num = 0
        for i in peakList:
            # basic info
            self.peak[num]['number'] = i
            self.peak[num]['average'] = averageList[i]
            self.peak[num]['peakHeight'] = ridge[i][0][1]
            # find top
            top = len(self.coef[0, :])
            for j in range(i + 1, len(ridge), 1):
                if (averageList[j] < 0):
                    top = ridge[j][0][1]
                    break
            self.peak[num]['topHeight'] = top
            # find base
            base = 1
            for k in range(i - 1, -1, -1):
                if (averageList[k] < 0):
                    base = ridge[k][0][1]
                    break
            self.peak[num]['baseHeight'] = base
            # increment
            num += 1

        # merge some peak
        typeList = np.zeros(len(self.peak))
        cluster = 1
        for i in range(len(self.peak)):
            if (int(typeList[i]) == 0):
                typeList[i] = cluster
                for j in range(i + 1, len(self.peak), 1):
                    if (self.peak[i]['baseHeight'] == self.peak[j]['baseHeight'] and self.peak[i]['topHeight'] ==
                            self.peak[j]['topHeight']):
                        typeList[j] = cluster
                    # elif(self.peak[i]['topHeight']==self.peak[j]['baseHeight']):
                    # typeList[j]=cluster
                cluster += 1

        # correct the peak
        tempPeak = np.zeros(cluster - 1, dtype=peakType)
        correctList = []
        for i in range(1, cluster, 1):
            temp = np.where(typeList == i)
            tempPeak[i - 1]['number'] = i - 1
            tempPeak[i - 1]['average'] = 0
            tempPeak[i - 1]['topHeight'] = np.max(self.peak[temp]['topHeight'])
            tempPeak[i - 1]['baseHeight'] = np.min(self.peak[temp]['baseHeight'])
            # find the maxP
            maxloc = np.where(line[self.peak[:]['peakHeight']] == np.max(line[self.peak[temp]['peakHeight']]))[0][0]
            tempPeak[i - 1]['peakHeight'] = self.peak[maxloc]['peakHeight']
            pRPeak = line[tempPeak[i - 1]['peakHeight']]
            pRBase = line[tempPeak[i - 1]['baseHeight']]
            if pRPeak - pRBase > threshold:
                correctList.append(1)
            else:
                correctList.append(0)

        # apply threshold
        self.newPeak = np.zeros(np.sum(correctList), dtype=peakType)
        j = 0
        for i in range(len(tempPeak)):
            if (correctList[i]):
                self.newPeak[j] = tempPeak[i]
                j += 1

    def cloundAerosol(self, line):
        layerType = np.dtype({
            'names': ['number', 'peakHeight', 'baseHeight', 'topHeight', 'average', 'ratio'],
            'formats': ['i', 'i', 'i', 'i', 'f', 'f']}, align=True)
        self.clAer = np.zeros(len(self.newPeak), dtype=layerType)
        for i in range(len(self.newPeak)):
            ratio = line[self.newPeak[i]['peakHeight']]
            ratio /= line[self.newPeak[i]['baseHeight']]
            self.clAer[i]['number'] = i
            self.clAer[i]['peakHeight'] = self.newPeak[i]['peakHeight']
            self.clAer[i]['baseHeight'] = self.newPeak[i]['baseHeight']
            self.clAer[i]['topHeight'] = self.newPeak[i]['topHeight']
            self.clAer[i]['average'] = self.newPeak[i]['average']
            self.clAer[i]['ratio'] = ratio

    def BLH(self, line, number):
        # use gaus wavelet
        ridge = self.cwtRidge(line, 'gaus1')
        ridgeType = np.dtype({
            'names': ['number', 'startHeight', 'endHeight', 'average'],
            'formats': ['i', 'i', 'i', 'f']}, align=True)
        ridgeList = np.zeros(len(ridge), dtype=ridgeType)
        i = 0
        for line in ridge:
            ridgeList[i]['number'] = i
            ridgeList[i]['startHeight'] = line[0][1]
            ridgeList[i]['endHeight'] = line[-1][1]
            average = 0
            for loc in line:
                average += self.coef[loc[0], loc[1]]
            average /= len(ridge[i])
            ridgeList[i]['average'] = average
            i += 1
        # print(ridgeList)


        ############classified discussion
        # caculate the reference height
        hMol = self.molList[0]
        if (len(self.clAer) == 0):
            hPar = 450
        elif (self.clAer[0]['baseHeight'] == 1 and len(self.clAer)==1):
            hPar=450
        elif (self.clAer[0]['baseHeight'] == 1):
            hPar = self.clAer[1]['baseHeight']
        else:
            hPar = self.clAer[0]['baseHeight']
        mlh = np.nan
        maxPositive = 0
        # 1. Hmin_mol< Hmin_part
        if hMol < hPar:
            # if exist Mcwt>0 and propogates up to range r<Hmin_mol
            for r in ridgeList:
                if r['average'] > maxPositive and r['startHeight'] < hMol and len(ridge[r['number']]) > 5:
                    mlh = r['startHeight']
                    maxPositive = r['average']
                elif r['startHeight'] > hMol:
                    break
        # 2. Hmin_part < Hmin_mol
        elif hPar < hMol:
            # if exist Mcwt<0 and propogates up to range r<Hmin_Par
            flag = 0
            for r in ridgeList:
                if r['average'] > maxPositive and r['startHeight'] < hPar and len(ridge[r['number']]) > 5:
                    mlh = r['startHeight']
                    maxPositive = r['average']
                    flag = 1
                elif r['startHeight'] > hPar:
                    break
            if (flag == 0):
                mlh = hPar
        #self.drawMap2(number, ridge, 'gaus1',mlh,hMol,hPar)
        return mlh

    def molecular(self, line):
        n = 11
        st = n
        ed = len(line) - n
        originLine = []
        # denormalized
        for i in range(len(line)):
            originLine.append(line[i] / ((i + 1) * 10 ** 2))
        # caculate the threshold
        threshold = 3 * np.var(originLine[400:450])
        # caculate the molecular layer
        self.molList = []
        for i in range(st, ed, 1):
            var = 0
            for j in range(i - n, i + n + 1, 1):
                var += (originLine[j] - np.mean(originLine[i - n:i + n + 1])) ** 2
            var /= 2 * n + 1
            if var < threshold:
                self.molList.append(i)

    def strat(self):
        self.mlh = []
        # for i in range(self.lenT):
        for i in range(0, 5400, 1):
            # find molecular layer
            self.molecular(self.matrix[:, i])
            # caculate the threshold
            threshold = self.getThreshold(self.matrix[:, i])
            # find particle layer
            ridge1 = self.cwtRidge(self.matrix[:, i], 'mexh')
            #self.drawMap(i,ridge1,'mexh')
            # apply a threshold and distinct cloud and aersol
            self.applyThreshold(self.matrix[:, i], 5 * threshold, ridge1)
            self.cloundAerosol(self.matrix[:, i])
            # print(self.newPeak)
            # print(self.clAer)
            # find the mlh
            mlh = self.BLH(self.matrix[:, i], i)
            #print(mlh)
            self.mlh.append(mlh)
            # print('---' + str(i) + '----')
            # print('molHeight:')
            # print(self.molList[0])
            # print('MLH:')
            # print(mlh)
        # print(self.mlh)
        #caculate the right mean
        self.meanMHL=np.zeros(5400)
        for i in range(0,5400,225):
            countNull=0
            s=0
            for j in range(i,i+225,1):
                if(np.isnan(self.mlh[j])):
                    countNull+=1
                else:
                    s+=self.mlh[j]*10
            s/=225-countNull
            self.meanMHL[i:i+225]=s


















