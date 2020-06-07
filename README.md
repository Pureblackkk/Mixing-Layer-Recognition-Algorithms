# Mixing-Layer-Recognition-Algorithms

### 一. 数据

使用`NC`格式的激光雷达回波数据，利用封装在`ReadFile`类中的`readFile`读取，可读取**单个文件**或**整个文件夹**

----

### 二. 数据操作

封装在`MatrixMethod`中：

​	`wavSmooth`：**小波去噪声**

​	`gausSmooth`：**高斯去噪声**

​	`standard` ：**数据标准化**

​	`converto1`：**归一化**

​	`deRange`：**去距离矫正化**

​	`unixToDate`：**时间戳转换**

---

### 三. 三种识别算法

+ 时空高度追踪（**THT**）算法

+ 基于垂直梯度和分组的混合层高度识别（**BRAVO**）算法

+ 大气结构（**STRAT**）算法

  分别封装在`Thtclass`、`Bravoclass`、`Stratclass`内，由`DrawPic`中的`caculateMLH`函数统一调用

---

### 四. 绘图函数

封装在`DrawPic`中，主要方法包涵：

​	`caculateMLH`：分别**计算三种算法**得出的结果

​	`drawSingDayCompare`：**绘制一日**识别结果对比图

​	`timeCaculate`：**多日**识别结果**计算**

​	`timeDayCompare`：**绘制多日**识别结果对比图

​	`timeStatic`：多日**统计量**计算

---

### 五. 例子

在`Main`文件中给出了绘制一日算法识别结果对比图的例子





