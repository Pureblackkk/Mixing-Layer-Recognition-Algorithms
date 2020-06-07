#import source file
import lib.Bravoclass as bravo
from lib.Thtclass import tht
from lib.MatrixMethod import matrixMethod
from lib.Stratclass import STRAT as strat
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
from sklearn.linear_model import LinearRegression

class drawPic:
    def __init__(self,matrix,path,date,t,r,comData,meanHeight):
        self.matrix=matrix
        self.lenT=len(matrix[0,:])
        self.lenR=len(matrix[:,0])
        self.path=path
        self.date=date
        self.t=t
        self.r=r
        self.comData=comData
        self.meanHeight=meanHeight

    def caculateMLH(self,matrix):
        print('start')
        # smooth
        m = matrixMethod(matrix)
        self.smoothMatrix = m.wavSmooth(0.8, 4)

        #####THT
        testTht = tht(self.t, self.r, self.smoothMatrix, 120, 20)
        ret = testTht.mlh()
        if ret != 0:
            self.thtMLH = np.zeros((self.lenT))
            for i in range(0, self.lenT, 225):
                self.thtMLH[i:i + 225] = (testTht.mixHeight[i:i + 225].mean())

        ######Bravo
        level = 2
        position = 95
        clusterNumber = 2
        testBravo = bravo.Wavelet(self.smoothMatrix, level, 25, position)
        tempList1 = testBravo.dwtMaxPoint()
        tempList2 = bravo.Cluster(tempList1, clusterNumber, 225)
        resList = tempList2.cluster()
        listDF = pd.DataFrame(resList)
        listDF['time'] = np.arange(len(resList)) + 1
        # bravoMLH = listDF.loc[:, 0]
        self.bravoMLH = np.zeros((self.lenT))
        for i in range(self.lenT):
            self.bravoMLH[i] = listDF.loc[i // 225, 0]

        ######STRAT
        testStrat = strat(self.t, self.r, self.smoothMatrix, 50)
        testStrat.strat()
        self.stratMLH = testStrat.meanMHL

        print('done')

    def drawSmoothCompare(self,number,select):
        # 2 smooth + 1 L_3
        if select==0:
            print('draw begain')
            m=matrixMethod(self.matrix)
            smooth1=m.wavSmooth(0.8,4)
            smooth2=m.gausSmooth()
            font_title = {'family': 'SimHei',
                          'weight': 'bold',
                          'size': 14
                          }
            plt.figure(figsize=(5,8))
            plt.plot(smooth1[:,number],range(450),color='orangered',label='Wavelet Smmoth ')
            plt.plot(smooth2[:,number],range(450),color='blue',label='Gaus Smooth')
            plt.plot(self.matrix[:, number], range(450), color='olivedrab', label='After Preprocessing')
            plt.ylabel('Height(km)',font_title)
            plt.yticks(range(0,500,100),['0','1','2','3','4'])
            plt.xlabel('RCS',font_title)
            plt.xticks([])
            plt.title('RCS Profile', font_title)
            plt.legend(loc='SouthWest')
            plt.show()

        #before pre + after pre
        if select==1:
            print('draw begain')
            font_title = {'family': 'SimHei',
                          'weight': 'bold',
                          'size': 14
                          }
            plt.figure(figsize=(5, 8))
            plt.plot(self.matrix[:, number], range(450), color='orangered', label='After Preprocessing')
            singlePath1 = "/Users/pureblack/bs_data/L2_06610_201811010000.nc"
            ncData=Dataset(singlePath1,'r')
            beforeMatrix= np.array(ncData['profile_data']).transpose()
            plt.plot(beforeMatrix[:,number],range(450),color='olivedrab',label='Before Preprocessing')
            plt.ylabel('Height(km)', font_title)
            plt.yticks(range(0, 500, 100), ['0', '1', '2', '3', '4'])
            plt.xlabel('RCS(a.u.)', font_title)
            plt.xticks([])
            plt.title('RCS Profile', font_title)
            plt.legend(loc='SouthWest')
            plt.show()

        # 2 smooth
        if select==3:
            print('draw begain')
            m = matrixMethod(self.matrix)
            smooth1 = m.wavSmooth(0.8, 4)
            smooth2 = m.gausSmooth()
            font_title = {'family': 'SimHei',
                          'weight': 'bold',
                          'size': 14
                          }
            plt.figure(figsize=(5, 8))
            plt.plot(smooth1[:, number], range(450), color='orangered', label='Wavelet Smmoth ')
            plt.plot(smooth2[:, number], range(450), color='olivedrab', label='Gaus Smooth')
            plt.ylabel('Height(km)', font_title)
            plt.yticks(range(0, 500, 100), ['0', '1', '2', '3', '4'])
            plt.xlabel('RCS(a.u.)', font_title)
            plt.xticks([])
            plt.title('RCS Profile', font_title)
            plt.legend(loc='SouthWest')
            plt.show()

        #map L2
        if select==4:
            singlePath1 = "/Users/pureblack/bs_data/L2_06610_201811010000.nc"
            ncData = Dataset(singlePath1, 'r')
            beforeMatrix = np.array(ncData['profile_data']).transpose()
            plt.figure(figsize=(10, 4))
            map = matrixMethod(beforeMatrix)
            newM = map.standard()
            lev = np.arange(-10, 10.5, 0.5)
            yaxis = np.arange(0, self.lenR, 50)
            ystring = yaxis * 10
            xaxis = np.arange(0, self.lenT, 225)
            xstring = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                       '24', '01', '02', '03', '04', '05', '06', '07']
            font = {'family': 'SimHei',
                    'weight': 'bold',
                    'size': 12
                    }


            plt.contourf(newM, levels=lev, extend='both', cmap='seismic')
            cbar = plt.colorbar(extend='both', ticks=None)
            cbar.ax.set_ylabel('RCS(A.U.)')
            cbar.set_ticks([])
            plt.xticks(xaxis, xstring, rotation=0)
            plt.xlabel('2018-11-01 Local Time(hour, UTC+8)', font)
            plt.yticks(yaxis, ystring)
            plt.ylabel('Height(m)', font)
            plt.title('RCS Profile on 2018-11-01', font)
            plt.show()

        #map wavelet
        if select==5:
            print('draw begain')
            plt.figure(figsize=(10, 4))
            map = matrixMethod(self.matrix)
            smooth = map.wavSmooth(0.8,4)
            newMap=matrixMethod(smooth)
            newM = newMap.standard()
            lev = np.arange(-10, 10.5, 0.5)
            yaxis = np.arange(0, self.lenR, 50)
            ystring = yaxis * 10
            xaxis = np.arange(0, self.lenT, 225)
            xstring = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                       '24', '01', '02', '03', '04', '05', '06', '07']
            font = {'family': 'SimHei',
                    'weight': 'bold',
                    'size': 12
                    }

            font_title = {'family': 'SimHei',
                          'weight': 'bold',
                          'size': 14
                          }
            plt.contourf(newM, levels=lev, extend='both', cmap='seismic')
            cbar = plt.colorbar(extend='both', ticks=None)
            cbar.ax.set_ylabel('RCS(A.U.)')
            cbar.set_ticks([])
            plt.xticks(xaxis, xstring, rotation=0)
            plt.xlabel('2018-11-01 Local Time(hour, UTC+8)', font)
            plt.yticks(yaxis, ystring)
            plt.ylabel('Height(m)', font)
            plt.title('RCS Profile on 2018-11-01', font)
            plt.show()

        #map all
        if select==6:
            print('draw begain')
            singlePath1 = "/Users/pureblack/bs_data/L2_06610_201811010000.nc"
            ncData = Dataset(singlePath1, 'r')
            beforeMatrix = np.array(ncData['profile_data']).transpose()
            map1 = matrixMethod(beforeMatrix)
            newM1 = map1.standard()
            m = matrixMethod(self.matrix)
            smooth1 = m.wavSmooth(0.8, 4)
            smooth2 = m.gausSmooth()
            map2=matrixMethod(smooth1)
            newM2=map2.standard()
            font_number = {'family': 'SimHei',
                          'size': 14
                          }
            font_title= {'family': 'SimHei',
                         'weight': 'bold',
                         'size': 14
                         }
            lev = np.arange(-10, 10.5, 0.5)
            yaxis = np.arange(0, self.lenR, 50)
            ystring =[ '0',  '500', '1000', '1500', '2000', '2500', '3000', '3500', '4000']
            xaxis = np.arange(0, self.lenT+225, 225)
            xstring = ['08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                       '24', '01', '02', '03', '04', '05', '06', '07','08']
            font = {'family': 'SimHei',
                    'weight': 'heavy',
                    'size': 13
                    }

            #####draw
            fig=plt.figure(figsize=(15,20))

            #sub1
            ax1=plt.subplot(3,3,1)
            ax1.plot(self.matrix[:, number], range(450), color='orangered', label='预处理后')
            ax1.plot(beforeMatrix[:, number], range(450), color='olivedrab', label='预处理前')
            #ax1.set_ylabel('Height(km)', font)
            ax1.set_yticks(range(0, 500, 100))
            ax1.set_yticklabels(['0', '1', '2', '3', '4'])
            #ax1.set_xlabel('RCS(a.u.)', font_title)
            ax1.set_xticks([])
            ax1.set_title('(a)', font_number)
            ax1.legend(loc='lower left')

            #sub2
            ax2=plt.subplot(3,3,2)
            ax2.plot(smooth1[:, number], range(450), color='orangered', label='离散小波平滑')
            ax2.plot(smooth2[:, number], range(450), color='blue', label='高斯平滑')
            ax2.plot(self.matrix[:, number], range(450), color='olivedrab', label='未平滑')
            #ax2.set_ylabel('Height(km)', font_title)
            ax2.set_yticks([])
            ax1.set_yticklabels(['0', '1', '2', '3', '4'])
            ax2.set_xlabel('RCS (a.u.)', font)
            ax2.set_xticks([])
            ax2.set_title('(b)', font_number)
            ax2.legend(loc='lower left')

            #sub3
            ax3=plt.subplot(3,3,3)
            ax3.plot(smooth1[:, number], range(450), color='orangered', label='离散小波平滑')
            ax3.plot(smooth2[:, number], range(450), color='olivedrab', label='高斯平滑')
            #ax3.set_ylabel('Height(km)', font_title)
            ax3.set_yticks([])
            ax1.set_yticklabels(['0', '1', '2', '3', '4'])
            #ax3.set_xlabel('(c)', font_title)
            ax3.set_xticks([])
            ax3.set_title('(c)', font_number)
            ax3.legend(loc='lower left')

            #sub4
            ax4 = plt.subplot(3, 1, 2)
            m1=ax4.contourf(newM1, levels=lev, extend='both', cmap='seismic')
            ax4.set_xticks([])
            #ax4.set_xlabel('(d)', font)
            ax4.set_yticks(range(0, 500, 100))
            ax4.set_yticklabels(['0', '1', '2', '3', '4'])
            #ax4.set_ylabel('Height(km)', font)
            ax4.set_ylabel('高度（千米）', font)
            ax4.set_title('(d)', font_number)


            # sub5
            ax5 = plt.subplot(3, 1, 3)
            m2=ax5.contourf(newM2, levels=lev, extend='both', cmap='seismic')
            ax5.set_xticks(xaxis)
            ax5.set_xlabel('2018-11-01 当地时间(小时, 东八区)', font)
            ax5.set_xticklabels(xstring)
            ax5.set_yticks(range(0, 500, 100))
            ax5.set_yticklabels(['0', '1', '2', '3', '4'])
            #ax5.set_ylabel('Height(km)', font)
            ax5.set_title('(e)', font_number)

            #color bar
            cbar = fig.colorbar(m1, ax=[ax4,ax5], extend='both', ticks=None)
            cbar.ax.set_ylabel('RCS(A.U.)',font)
            cbar.set_ticks([])


            #plt.suptitle('数据预处理及平滑方法比较',fontsize=15,fontweight='black')
            plt.savefig(self.path+'compare1.png', dpi=400, bbox_inches='tight')
            plt.show()
            print('end')

    def drawWavalet(self,type):

        font_title = {'family': 'SimHei',
                      'weight': 'bold',
                      'size': 14
                      }

        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 13
                }

        if type == 'haar' or type == 'db8':
            wavelet = pywt.Wavelet(type)
            phi, psi, x = wavelet.wavefun(level=3)
            plt.figure(figsize=(8, 7))
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(x, psi, color='black')
            ax1.grid()
            ax1.set_title(type + "母小波", font_title)
            # ax1.set_ylabel('(a)')
            ax2 = plt.subplot(2, 1, 2)
            ax2.set_title(type + "父小波", font_title)
            ax2.plot(x, phi, color='black')
            ax2.grid()
            # ax2.set_ylabel('(b)')
            plt.savefig(self.path + type + '.png', dpi=400, bbox_inches='tight')
            plt.show()
        elif type=='mexh' or type=='gaus1':
            wavelet = pywt.ContinuousWavelet(type)
            psi, x = wavelet.wavefun(level=10)
            plt.figure(figsize=(7, 4))
            plt.plot(x, psi, color='black')
            plt.grid()
            #plt.title("Mexh 小波基函数", font_title)
            plt.xlabel("X",font)
            plt.ylabel("${ψ_{a,b}'(x)}$",font)
            plt.savefig(self.path + type + '.png', dpi=400, bbox_inches='tight')
            plt.show()

    def drawBravoDWT(self,number,levels):
        smooth=matrixMethod(self.matrix)
        smoothMatrix=smooth.wavSmooth(0.8,4)
        r=range(450)
        cA5, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(smoothMatrix[:, number], 'haar', mode='smooth', level=levels)
        total = levels + 1
        xTicks=range(0,450,100)
        xString=['0 km','1 km','2 km','3 km','4 km']
        plt.figure(figsize=(9,10))
        for i in range(total):
            exec("ax" + str(i + 1) + "=plt.subplot(6,1," + str(i + 1) + ")")
        exec("ax1.plot(r, smoothMatrix[:, 0], color='black')")
        exec("ax1" + ".set_yticks([])")
        exec("ax1" + ".set_ylabel('RCS (a.u.)')")
        exec("ax1" + ".set_xticks(xTicks)")
        exec("ax1" + ".set_xticklabels(xString)")
        for i in range(2, 7):
            exec("ax" + str(i) + ".scatter(range(len(cD" + str(i - 1) + ")),cD" + str(i - 1) + ",s=10,color='black')")
            exec("ax" + str(i) + ".axhline(np.percentile(cD" + str(i - 1) + ", 75),color='red')")
            exec("ax" + str(i) + ".set_yticks([])")
            exec("ax" + str(i) + ".set_xticks([])")
            exec("ax" + str(i) + ".set_ylabel('Level='+str(i-1))")
            #exec("ax" + str(i) + ".axhline(np.percentile(cD" + str(i - 1) + ", 25))")
        plt.savefig(self.path +'dwtNote.png', dpi=400, bbox_inches='tight')
        plt.show()

    def drawBravoCluster(self):
        smooth = matrixMethod(self.matrix)
        smoothMatrix = smooth.wavSmooth(0.8, 4)
        level1 = 2
        level2=5
        position = 95
        clusterNumber1 = 2
        clusterNumber2=5
        plt.figure(figsize=(8,10))
        ax1=plt.subplot(2,1,1)
        ax2 = plt.subplot(2, 1, 2)


        testBravo1 = bravo.Wavelet(smoothMatrix, level1, 25, position)
        tempList1a = testBravo1.dwtMaxPoint()
        tempList1b = bravo.Cluster(tempList1a, clusterNumber1, 225)
        resList1 = tempList1b.cluster()
        listDF1 = pd.DataFrame(resList1)
        listDF1['time'] = np.arange(len(resList1)) + 1
        # bravoML1 = []
        # for i in range(0, 5400, 225):
        #     bravoML1[i:i + 225] = listDF1.loc[i//225,0]
        for i in range(clusterNumber1):
            ax1.scatter(listDF1['time'],listDF1.loc[:,i],label='class'+str(i))
        ax1.plot(listDF1['time'],self.meanHeight,label='BL-VIEW',color='black',marker='.')
        ax1.legend(loc=5)
        ax1.set_title('(a)聚类数: '+str(clusterNumber1))
        ax1.set_yticks(np.arange(0,5000,1000))
        ax1.set_yticklabels(['0','1','2','3','4'])
        ax1.set_ylabel('高度(千米)')
        ax1.set_xticks(np.arange(1,26,4))
        ax1.set_xticklabels(['08', '12', '16', '20', '24','04','08'])
        ax1.set_xlabel('时间（小时）')

        testBravo2 = bravo.Wavelet(smoothMatrix, level2, 25, position)
        tempList2a = testBravo1.dwtMaxPoint()
        tempList2b = bravo.Cluster(tempList2a, clusterNumber2, 225)
        resList2 = tempList2b.cluster()
        listDF2 = pd.DataFrame(resList2)
        listDF2['time'] = np.arange(len(resList2)) + 1
        # bravoML2 = []
        # for i in range(0, 5400, 225):
        #     bravoML2[i:i + 225] = listDF2.loc[i//225,0]
        for i in range(clusterNumber2):
            ax2.scatter(listDF2['time'],listDF2.loc[:,i],label='class'+str(i))
        ax2.plot(listDF2['time'],self.meanHeight,label='BL-VIEW',color='black',marker='.')
        ax2.legend(loc=5)
        ax2.set_title('(b)聚类数: '+str(clusterNumber2))
        ax2.set_yticks(np.arange(0, 5000, 1000))
        ax2.set_yticklabels(['0', '1', '2', '3', '4'])
        ax2.set_ylabel('高度(千米)')
        ax2.set_xticks(np.arange(1, 26, 4))
        ax2.set_xticklabels(['08', '12', '16', '20', '24', '04', '08'])
        ax2.set_xlabel('时间（小时）')

        plt.savefig(self.path + 'differentCluster', dpi=400, bbox_inches='tight')
        plt.show()

    def drawSingleDayExample(self,time):
        length=len(time)
        timeIndex=[]
        for t in time:
            timeIndex.append((t-8)*225)
        m = matrixMethod(self.matrix)
        smoothMatrix=m.wavSmooth(0.8,4)
        #smoothMatrix = m.gausSmooth()
        plt.figure(figsize=(18,8))
        font_title = {'family': 'SimHei',
                      'weight': 'bold',
                      'size': 14
                      }
        title=['(a)','(b)','(c)','(d)','(e)']
        for i in range(length):
            exec("ax" + str(i + 1) + "=plt.subplot(1,length," + str(i + 1) + ")")
        #line 1
        for i in range(1, length+1):
            exec("ax" + str(i) + ".plot(smoothMatrix[:,int(timeIndex[i-1])],range(450),color='black')")
            if(i==1):
                exec("ax" + str(i) + ".set_ylabel('高度(km)', font_title)")
                exec("ax" + str(i) + ".set_yticks(range(0, 500, 100))")
                exec("ax" + str(i) + ".set_yticklabels(['0', '1', '2', '3', '4'])")
            exec("ax" + str(i) + ".set_xticks([])")
            if(i==3):
                exec("ax" + str(i) + ".set_xlabel('RCS', font_title)")
            exec("ax" + str(i) + ".set_yticks(range(0, 500, 100))")
            exec("ax" + str(i) + ".set_yticklabels(['0', '1', '2', '3', '4'])")
            exec("ax" + str(i) + ".set_title(title[i-1]+' '+self.date[int(timeIndex[i-1])])")
        plt.savefig(self.path + 'lineExample1.png', dpi=400, bbox_inches='tight')
        plt.show()

    def drawSingleDayCompare(self):
        fig=plt.figure(figsize=(10,4))
        map=matrixMethod(self.smoothMatrix)
        newM=map.standard()
        lev=np.arange(-10,10.5,0.5)
        yaxis=np.arange(0,self.lenR,50)
        ystring=yaxis*10
        xaxis=np.arange(0,self.lenT+225,225)
        xstring=['08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','01','02','03','04','05','06','07','08']
        font = {'family' : 'SimHei',
        'weight' : 'bold',
        'size'   : 12
        }
        font_title={'family' : 'SimHei',
        'weight' : 'bold',
        'size'   : 14
        }
        ax1 = fig.add_subplot(111)
        plt.contourf(newM,levels=lev,extend = 'both', cmap='seismic')

        cbar=plt.colorbar(extend = 'both', ticks=None)
        cbar.ax.set_ylabel('RCS(A.U.)')
        cbar.set_ticks([])
        plt.xticks(xaxis,xstring,rotation=0)
        plt.xlabel('2018-06-14 当地时(小时, 东八区)', font)
        plt.yticks(yaxis,ystring)
        plt.ylabel('高度(米)',font)

        #plt.title('2018-06-14',font)

        ax2 = ax1.twinx()
        ax2.plot(self.thtMLH,color='green',label='THT',linewidth=2)
        ax2.plot(self.bravoMLH,color='blue',label='BRAVO',linewidth=2)
        ax2.plot(self.stratMLH, color='yellow', label='STRAT', linewidth=2)
        ax2.plot(self.comData,color='black', label='BL-VIEW',linewidth=2)
        ax2.set_ylim(0,4500)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        plt.legend()

        plt.savefig("/Users/pureblack/Desktop/毕业论文/图/singleDayCompareChinese",dpi=500, bbox_inches = 'tight')
        plt.show()

    def lineCompare(self):
        #average 24
        thtMLH=[]
        bravoMLH=[]
        stratMLH=[]
        for i in range(0, 5400, 225):
            thtMLH.append(self.thtMLH[i])
            bravoMLH.append(self.bravoMLH[i])
            stratMLH.append(self.stratMLH[i])
        font = {'family': 'SimHei',
                'weight': 'bold',
                'size': 14
                }
        thtMLTemp = np.sort(np.array(thtMLH))
        bravoMLTemp = np.sort(np.array(bravoMLH))
        stratMLTemp=np.sort(np.array(stratMLH))
        comDataTemp = np.sort(np.array(self.meanHeight))

        #t vs b
        plt.scatter(thtMLH, bravoMLH, color='tomato', label='THT vs BRAVO')
        model = LinearRegression()
        model.fit(thtMLTemp.reshape(-1, 1), bravoMLTemp.reshape(-1, 1))
        ret = model.predict(thtMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(thtMLTemp[1] - thtMLTemp[0]))
        plt.plot(thtMLTemp, ret, color='tomato', label='coeff=' + "%.2f" % float(coeff))

        #t vs s
        plt.scatter(thtMLH, stratMLH, color='tomato', label='THT vs BRAVO')
        model = LinearRegression()
        model.fit(thtMLTemp.reshape(-1, 1), stratMLTemp.reshape(-1, 1))
        ret = model.predict(thtMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(thtMLTemp[1] - thtMLTemp[0]))
        plt.plot(thtMLTemp, ret, color='tomato', label='coeff=' + "%.2f" % float(coeff))

        # t vs r
        plt.scatter(thtMLH, self.meanHeight, color='darkcyan', label='THT vs REFER')
        model = LinearRegression()
        model.fit(thtMLTemp.reshape(-1, 1), comDataTemp.reshape(-1, 1))
        ret = model.predict(thtMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(thtMLTemp[1] - thtMLTemp[0]))
        plt.plot(thtMLTemp, ret, color='darkcyan', label='coeff=' + "%.2f" % float(coeff))

        #b vs s
        plt.scatter(bravoMLH, stratMLH, color='gold', label='BRAVO vs REFER')
        model = LinearRegression()
        model.fit(bravoMLTemp.reshape(-1, 1), stratMLTemp.reshape(-1, 1))
        ret = model.predict(bravoMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(bravoMLTemp[1] - bravoMLTemp[0]))
        plt.plot(bravoMLTemp, ret, color='gold', label='coeff=' + "%.2f" % float(coeff))

        #b vs r
        plt.scatter(bravoMLH, self.meanHeight, color='gold', label='BRAVO vs REFER')
        model = LinearRegression()
        model.fit(bravoMLTemp.reshape(-1, 1), comDataTemp.reshape(-1, 1))
        ret = model.predict(bravoMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(bravoMLTemp[1] - bravoMLTemp[0]))
        plt.plot(bravoMLTemp, ret, color='gold', label='coeff=' + "%.2f" % float(coeff))

        #s vs r
        plt.scatter(stratMLH, self.meanHeight, color='gold', label='BRAVO vs REFER')
        model = LinearRegression()
        model.fit(stratMLTemp.reshape(-1, 1), comDataTemp.reshape(-1, 1))
        ret = model.predict(stratMLTemp.reshape(-1, 1)).flatten()
        coeff = float(float((ret[1] - ret[0])) / float(stratMLTemp[1] - stratMLTemp[0]))
        plt.plot(stratMLTemp, ret, color='gold', label='coeff=' + "%.2f" % float(coeff))

        plt.title("Comparison of different algorithm results on 2018-06-14", font)
        plt.legend()

    def timeCaculate(self,time):
        #preparing task
        #order: tht-bravo-strat-compare
        self.time=time
        self.dateDict={}
        path1="/Users/pureblack/bs_data/L3_data/L3_DEFAULT_06610_2018"
        path2="0000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
        self.timeMatrix=np.zeros((self.lenR,1))
        for t in time:
            path=path1+t+path2
            file1 = read.ReadFile(path, "single")
            profileData, meanLayerHeight, comData, ti, r = file1.readFile()
            self.caculateMLH(profileData)
            listMLH=[]
            listMLH.append(self.thtMLH)
            listMLH.append(self.bravoMLH)
            listMLH.append(self.stratMLH)
            listMLH.append(np.array(comData).flatten())
            self.dateDict[t]=listMLH
            #whole smoothmatrix
            self.timeMatrix=np.hstack((self.timeMatrix,self.smoothMatrix))

    def timeDayCompare(self):
        fig = plt.figure(figsize=(10, 4))
        map = matrixMethod(self.timeMatrix)
        newM = map.standard()
        lev = np.arange(-10, 10.5, 0.5)
        yaxis = np.arange(0, self.lenR, 50)
        ystring = yaxis * 10
        xaxis = np.arange(1, len(newM[0,:])+1, 5400)
        xstring =self.time
        font = {'family': 'SimHei',
                'weight': 'bold',
                'size': 12
                }
        font_title = {'family': 'SimHei',
                      'weight': 'bold',
                      'size': 14
                      }
        ax1 = fig.add_subplot(111)
        plt.contourf(newM, levels=lev, extend='both', cmap='seismic')

        cbar = plt.colorbar(extend='both', ticks=None)
        cbar.ax.set_ylabel('RCS(A.U.)')
        cbar.set_ticks([])
        plt.xticks(xaxis, xstring, rotation=0)
        plt.xlabel('2018-06-14 Local Time(hour, UTC+8)', font)
        plt.yticks(yaxis, ystring)
        plt.ylabel('Height(m)', font)

        plt.title('RCS Profile on 2018-06-14', font)

        #merge the line
        thtMLH =[]
        bravoMLH =[]
        stratMLH =[]
        comDataMLH =[]
        for t in self.time:
            thtMLH=np.append(thtMLH,self.dateDict[t][0])
            bravoMLH = np.append(bravoMLH, self.dateDict[t][1])
            stratMLH = np.append(stratMLH , self.dateDict[t][2])
            comDataMLH  = np.append(comDataMLH , self.dateDict[t][3])

        ax2 = ax1.twinx()
        ax2.plot(thtMLH, color='green', label='THT', linewidth=2)
        ax2.plot(bravoMLH, color='blue', label='BRAVO', linewidth=2)
        ax2.plot(stratMLH, color='yellow', label='STRAT', linewidth=2)
        ax2.plot(comDataMLH, color='black', label='REFER', linewidth=2)
        ax2.set_ylim(0, 4500)
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        plt.legend()

        # plt.savefig("/Users/pureblack/Desktop/中期图/phrase20.png",dpi=500, bbox_inches = 'tight')
        plt.show()

    def timeStatic(self,time):
        #bias｜std| consistency | r2
        self.bias=np.zeros(6)
        self.std=np.zeros(6)
        self.consistency=np.zeros(6)
        self.r2=np.zeros(6)
        for t in time:
            res=self.dateDict[t]
            count=0
            for i in range(3):
                for j in range(i+1,4):
                    b = 0
                    s = 0
                    c = 0
                    r = 0
                    for k in range(24):
                        b+=abs(res[i][k]-res[j][k])
                        s+=(res[i][k]-res[j][k])**2
                        if(abs(res[i][k]-res[j][k])<150):
                            c+=1
                    b/=24
                    s=math.sqrt(s)/24
                    c=c/24
                    model = LinearRegression()
                    model.fit(res[i].reshape(-1, 1), res[j].reshape(-1, 1))
                    ret = model.predict(res[i].reshape(-1, 1)).flatten()
                    r= float(float((ret[1] - ret[0])) / float(res[i][1] -res[i][0]))
                    self.bias[count]+=b
                    self.std[count]+=s
                    self.consistency[count]+=c
                    self.r2[count]+=r
                    count+=1
        self.bias/=4
        self.std/=4
        self.consistency/=4
        self.r2/=4

    def fourHourTimeStatic(self,time):
        #divded the time into every four hour
        self.fourBias = np.zeros((6,6))
        self.fourStd = np.zeros((6,6))
        self.fourConsistency = np.zeros((6,6))
        self.fourR2 = np.zeros((6,6))
        for t in time:
            res = self.dateDict[t]
            for h in range(0,24,4):
                count = 0
                for i in range(3):
                    for j in range(i + 1, 4):
                        b = 0
                        s = 0
                        c = 0
                        r = 0
                        for k in range(h*4,4*(h+1),1):
                            b += abs(res[i][k] - res[j][k])
                            s += (res[i][k] - res[j][k]) ** 2
                            if (abs(res[i][k] - res[j][k]) < 150):
                                c += 1
                        s = math.sqrt(s) / 4
                        c = c / 4
                        model = LinearRegression()
                        model.fit(res[i][4*h:4*h+4].reshape(-1, 1), res[j][4*h:4*h+4].reshape(-1, 1))
                        ret = model.predict(res[i][4*h:4*h+4].reshape(-1, 1)).flatten()
                        r = float(float((ret[1] - ret[0])) / float(res[i][4*h+1] - res[i][4*h]))
                        self.bias[count][h]+= b
                        self.std[count][h]+= s
                        self.consistency[count][h]+= c
                        self.r2[count][h]+= r
                        count += 1
        self.fourBias/=4
        self.fourStd/=4
        self.fourConsistency/=4
        self.fourR2/=4

    def thtStep(self):
        font_title = {'family': 'SimHei',
                      'weight': 'bold',
                      'size': 12
                      }
        m = matrixMethod(self.matrix)
        self.smoothMatrix = m.wavSmooth(0.8, 4)
        testTht = tht(self.t, self.r, self.smoothMatrix, 120, 20)
        ret = testTht.mlh()

        ##1
        # avvar=preprocessing.scale(testTht.avvar[:, 20])
        # avlog=preprocessing.scale(testTht.avlogpr2G[:, 20])
        # plt.figure(figsize=(5, 8))
        # plt.plot(avvar,range(450),label='平均方差',color='red')
        # plt.plot(avlog,range(450),label='平均梯度',color='blue')
        # plt.legend()
        # plt.axhline(testTht.hrefIndex[20],color='black',lw=2)
        # plt.yticks(range(0, 500, 100), ['0', '1', '2', '3', '4'])
        # plt.ylabel('高度(千米)', font_title)
        # plt.xlabel('RCS (a.u.)',font_title)
        # plt.savefig(self.path + 'thtstep1.png', dpi=400, bbox_inches='tight')
        # plt.show()

        ##2
        avvar = preprocessing.scale(testTht.var[:, 2401])
        avlog = preprocessing.scale(testTht.logpr2G[:, 2401])
        plt.figure(figsize=(5, 8))
        plt.plot(avvar, range(450), label='方差2', color='red')
        plt.plot(avlog, range(450), label='梯度2', color='blue')
        plt.legend()
        plt.axhline(testTht.mixHeight[2401]/10,color='black',lw=2)
        plt.axhline(testTht.rangehigh[2401], color='black',lw=2)
        plt.axhline(testTht.rangelow[2401], color='black',lw=2)
        plt.yticks(range(0, 500, 100), ['0', '1', '2', '3', '4'])
        plt.ylabel('高度(千米)', font_title)
        plt.xlabel('RCS (a.u.)', font_title)
        plt.savefig(self.path + 'thtstep3.png', dpi=400, bbox_inches='tight')
        plt.show()

    def stratMexhDraw(self):
        m = matrixMethod(self.matrix)
        self.smoothMatrix = m.wavSmooth(0.8, 4)
        testStrat = strat(self.t, self.r, self.smoothMatrix, 50)
        testStrat.strat()











































