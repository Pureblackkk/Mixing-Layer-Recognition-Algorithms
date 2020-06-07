#import source file
import lib.Bravoclass as bravo
from lib.Thtclass import tht
from lib.MatrixMethod import matrixMethod
from lib.DrawPic import drawPic
import lib.Readfile as read
import numpy as np
############################################################
#read CL51 NC File 
singlePath1 = "/Users/pureblack/bs_data/L3_data/L3_DEFAULT_06610_201806170000_1_360_1_3120_10_30_4000_3_0_1_500_1000_4000_60.nc"
file1 = read.ReadFile(singlePath1,"single")
profileData, meanLayerHeight,comData, t, r = file1.readFile()
date=matrixMethod(t)
dateList=date.unixToDate()

#test as an example
draw1=drawPic(profileData,'/Users/pureblack/Desktop/',dateList,t,r,comData,meanLayerHeight)
#caculate the mixing layer height using three different methods
draw1.caculateMLH(profileData)
#draw picture
draw1.drawSingleDayCompare()
