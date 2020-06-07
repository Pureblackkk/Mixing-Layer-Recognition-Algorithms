import os
import os.path
import re
import sys
import codecs
from netCDF4 import Dataset
import numpy as np

class ReadFile:
    def __init__(self,path,type):
        self.path=path
        self.type=type

    def readFile(self):
        if self.type =='single':
            ncData=Dataset(self.path,'r')
            profileData = np.array(ncData['Bs_profile_data']).transpose()
            comData = np.array(ncData['Mean_Layer_Height'])
            t = np.array(ncData['time'])
            r = np.array(ncData['range'])
            step = len(comData) // 24
            meanLayerHeight = []
            for i in range(24):
                meanLayerHeight.append(comData[i * step][0])
            return profileData, meanLayerHeight, comData, t, r

        elif(self.type=='package'):
            files = os.listdir(self.path)
            files.sort(key=lambda x: int(x[17:29]))
            return files

        else:
            print('Wrong type!')
            return -1

