import numpy as np
import struct
import math
import codecs
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D

path = 'F:/llc/test/'
#Paraments Setting
Maximum_R = 10#设置查找圆的最大半径/说明：半径的设置必须考虑到使圆最大直径在重叠区域内
Maximum_MoveXY = 200#为了缩小查找点的范围，给定拼接图像之间相对最大错位

#输入点坐标，获得以该点为圆心不同半径范围内的点的数量，并生成特征向量
def ChracterVector_circle(circle_list, RecPointData):

    #KDTree
    KDTREE = spatial.KDTree(RecPointData)
    Vector0 = KDTREE.query_ball_point(circle_list, Maximum_R-8)
    Vector1 = KDTREE.query_ball_point(circle_list, Maximum_R-6)
    Vector2 = KDTREE.query_ball_point(circle_list, Maximum_R-4)
    Vector3 = KDTREE.query_ball_point(circle_list, Maximum_R-2)
    Vector4 = KDTREE.query_ball_point(circle_list, Maximum_R)
    if len(circle_list)==2:
        Vector = [len(Vector0), len(Vector1), len(Vector2), len(Vector3), len(Vector4)]
        return Vector
    Vector_0 = []
    Vector_1 = []
    Vector_2 = []
    Vector_3 = []
    Vector_4 = []
    for i in range(len(circle_list)):
        Vector_0.append(len((Vector0[i])[0]))
        Vector_1.append(len((Vector1[i])[0]))
        Vector_2.append(len((Vector2[i])[0]))
        Vector_3.append(len((Vector3[i])[0]))
        Vector_4.append(len((Vector4[i])[0]))
    Vector = []
    Vector.append(Vector_0)
    Vector.append(Vector_1)
    Vector.append(Vector_2)
    Vector.append(Vector_3)
    Vector.append(Vector_4)
    return Vector


def lateral_stitch(Image_i, Image_j):

    picture_left = path+'img'+str(Image_i)+'.txt'
    picture_right = path+'img'+str(Image_j)+'.txt'

    # ----------------------打开待拼接的两幅图------------------------------
    f = codecs.open(picture_left, mode='rb')#左图
    line_temp = f.read(12 * 4)  # 读取单个定位点信息/单个定位点有12个定位信息，每个信息占4个字节
    line = struct.unpack("12f", line_temp)
    list1 = []
    list2 = []
    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0]))
        y = math.ceil(float(y[0]))
        list1.append(x)
        list2.append(y)
        line_temp = f.read(12 * 4)

        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)
    Coordinate_left = np.array(list(zip(list1, list2)))#转置矩阵
    f.close()

    ff = codecs.open(picture_right, mode='rb')#右图
    line_temp = ff.read(12 * 4)
    line = struct.unpack("12f", line_temp)
    list11 = []
    list22 = []
    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0]))
        y = math.ceil(float(y[0]))
        list11.append(x)
        list22.append(y)
        line_temp = ff.read(12 * 4)
        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)
    Coordinate_right = np.array(list(zip(list11, list22)))  # 转置矩阵
    ff.close()

    #-----------------特征矢量图匹配-------------------------
    # 选取左图圆心点并构造特征向量
    """
    1、特征点的选取必须保证所画圆最大半径在重叠区域内
    2、点圆范围内的分子数适中，分子数过小选中的圆心点可能为噪声，过大则圆域内信号太密集算法误差增大
    """
    max_x = int(max(Coordinate_left[:, 0]))
    min_x = int(min(Coordinate_left[:, 0]))
    max_y = int(max(Coordinate_left[:, 1]))
    min_y = int(min(Coordinate_left[:, 1]))

    first_match = 20# 因为边缘有一些点太少，经过多次尝试，初始位置从20开始（可调整）
    CircleLeft_X = max_x - first_match
    listEdge = []
    countList = []
    count = 0

    for i in Coordinate_left:#获取边缘重叠区域
        if (i[0] > CircleLeft_X - 2*Maximum_R - 2 and i[0] < CircleLeft_X):
            listEdge.append(i)

    circleX_left = CircleLeft_X - Maximum_R - 1#选取左图圆心点[circleX_left, circleY_left]
    for circleY_left in range(min_y+Maximum_R, max_y-Maximum_R, 10):
        for i in listEdge:
            if (abs(i[0] - circleX_left) < Maximum_R and abs(i[1] - circleY_left) < Maximum_R):
                count+=1
        countList.append([circleY_left, count])
        count = 0
    countList.sort(key=lambda x:x[1], reverse=False)#按照第二项升序排序
    maxList = countList[len(countList)-1]
    for i in countList:
        if i[1] > maxList[1]/5:#框选的区域分子密度为最大密度的1/4左右
            circleY_left = i[0]
            break

    RecPointList = []

    for i in listEdge:#提取圆心附近的坐标点
        if (abs(i[0] - circleX_left) <= Maximum_R and abs(i[1] - circleY_left) <= Maximum_R):
            RecPointList.append(i)
    RecPointData_Left = np.array(RecPointList)

    circle_left = [circleX_left, circleY_left]
    Vector_left = ChracterVector_circle(circle_left, RecPointData_Left)#计算特征向量

    #遍历右图区域获得特征矩阵Matrix
    List_right = []
    List_right_around = []
    for i in Coordinate_right:
        if (i[1] < circleY_left + Maximum_MoveXY/2 + Maximum_R and i[1] > circleY_left - Maximum_MoveXY/2 - Maximum_R and
            i[0] < first_match + Maximum_MoveXY/4 + Maximum_R):
            List_right_around.append(list(i))
            if (i[1] < circleY_left + Maximum_MoveXY/2 and i[1] > circleY_left - Maximum_MoveXY/2 and
                i[0] < first_match + Maximum_MoveXY/4):
                List_right.append(list(i))
    List_right_around = np.array(List_right_around)


    #————————搜索优化-------------------
    """
    由于List_right中点的数量过多，导致计算缓慢，为了优化采用构建的网格作为搜索中心
    """
    ttt = np.array(List_right)
    MaxListX = max(ttt[:, 0])
    MinListX = min(ttt[:, 0])
    MaxListY = max(ttt[:, 1])
    MinListY = min(ttt[:, 1])
    X_grid, Y_grid = np.meshgrid(range(MinListX, MaxListX, 1), range(MinListY, MaxListY, 1))
    X_grid = X_grid.reshape(1, -1)
    Y_grid = Y_grid.reshape(1, -1)
    List_right = []
    List_right.append(list(X_grid))
    List_right.append(list(Y_grid))
    List_right = np.array(List_right).T
    #-----------------------------------
    Matrix = ChracterVector_circle(List_right, List_right_around)
    #求取匹配点
    pinjun_left = sum(Vector_left)/len(Vector_left)
    left = np.array(Vector_left)-pinjun_left
    right = np.array(Matrix).T
    junzhi_right = np.mean(right, axis=1)
    junzhi_right0 = list(junzhi_right)
    junzhi_right = []
    junzhi_right.append(junzhi_right0)
    junzhi_right.append(junzhi_right0)
    junzhi_right.append(junzhi_right0)
    junzhi_right.append(junzhi_right0)
    junzhi_right.append(junzhi_right0)
    junzhi_right = np.array(junzhi_right).T
    Matrix = np.array(Matrix).T-junzhi_right

    C = cosine_similarity(left.reshape(1,-1), Matrix)
    C = list(C[0])

    #C = list(np.dot(np.array(Matrix).T, np.array(Vector_left)))
    MaxIndex = C.index(max(C))
    circle_right = List_right[MaxIndex]

    # 绘制相似度曲线
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    for t in range(len(List_right)):
        x.append((List_right[t])[0, 0])
        y.append((List_right[t])[0, 1])
    x = np.array(x)
    y = np.array(y)
    z = np.array(C)
    ax.plot(x,y,z,'ro', c='black', alpha=1, marker=',')

    #计算相对位移
    x_distance = circle_right[0, 0] - circleX_left
    y_distance = circle_right[0, 1] - circleY_left

    return x_distance, y_distance, circleX_left, circleY_left,

x_distance,y_distance,a,b =lateral_stitch(0, 1)
print(x_distance, y_distance)

#--------------------------绘图测试--------------------------------------

stitch = "F:/llc/test/image.jpg"
XY_move = np.zeros((2,2))#偏移量
XY_move[1][0] = x_distance
XY_move[1][1] = y_distance

plt.figure(2)
plt.xlim(xmax=5000, xmin=-100)
plt.ylim(ymax=5000, ymin=-100)
for num in range(2):

    Single_Image = codecs.open(path+'img'+str(num)+'.txt', mode='rb')
    line_temp = Single_Image.read(12 * 4)
    line = np.empty((1, 12))
    line = struct.unpack("12f", line_temp)
    list11 = []
    list22 = []

    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0])) - XY_move[num, 0]
        y = math.ceil(float(y[0])) - XY_move[num, 1]
        list11.append(x)
        list22.append(y)
        line_temp = Single_Image.read(12 * 4)
        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)

    Single_Image.close()
    # 作图，打点方式为像素点，600dpi，可以设置保存路径
    plt.plot(list11, list22, 'ro', c='black', alpha=0.1, marker=',')

plt.plot(a-10, b, c='r', markersize=0.1, marker=',')
plt.plot(a+10, b, c='r', markersize=0.1, marker=',')
plt.plot(a, b, c='r', markersize=0.1, marker=',')
plt.savefig(stitch, dpi=600)
plt.show()
