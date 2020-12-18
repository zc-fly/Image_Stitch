"""
To complete the final image stitching, we use the Breadth First Algorithm
to stitching all the image which can be stitched.
"""
import os
import math
import codecs
import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial

path = 'F:/llc/test/'
stitch = 'F:/llc/TotalStitch.jpg'
MovingDistance_stitch = "F:/llc/test/MovingDistance_stitch.txt"
dataDir = os.listdir(path)
fileNum = len(dataDir)
Image_width = 5#待拼接图像的尺寸为Image_width*Image_length
Image_length = 5

#数据文件重命名
def Rename_dataFile():
    i = 0
    os.listdir(path)
    for file in os.listdir(path):
        os.renames(os.path.join(path,file), os.path.join(path, 'img'+ str(i))+'.txt')
        i = i + 1

#-----------------------------------------------垂直方向两幅图拼接-------------------------------------
def straight_stitch(Image_i, Image_j):
    # 待拼接图片（下方）
    picture_down = path + 'img' + str(Image_i) + '.txt'
    # 待拼接图片（上方）
    picture_up = path + 'img' + str(Image_j) + '.txt'

    # 拼接效果不稳定时对拼图进行调整(目前主要是中位数算法不稳定)
    x_adjust = 0
    y_adjust = 0

    # 打开拼接的第一幅图（下方）
    # 第一幅图x坐标列表为listx_1，y坐标为listy_1
    f = codecs.open(picture_down, mode='rb')
    line_temp = f.read(12 * 4)  # 读取单个定位点信息/单个定位点有12个定位信息，每个信息占4个字节
    line = np.empty((1, 12))
    line = struct.unpack("12f", line_temp)
    listx_1 = []
    listy_1 = []
    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0]))
        y = math.ceil(float(y[0]))
        listx_1.append(x)
        listy_1.append(y)
        line_temp = f.read(12 * 4)
        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)
    f.close()

    vector1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vector1_len = len(vector1)
    # 打开拼接的第二幅图（上方）
    ff = codecs.open(picture_up, mode='rb')
    line_temp = ff.read(12 * 4)  # 读取单个定位点信息/单个定位点有12个定位信息，每个信息占4个字节
    line = np.empty((1, 12))
    line = struct.unpack("12f", line_temp)
    listx_2 = []
    listy_2 = []
    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0]))
        y = math.ceil(float(y[0]))
        listx_2.append(x)
        listy_2.append(y)
        line_temp = ff.read(12 * 4)
        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)
    ff.close()

    vector2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vector2_len = len(vector2)
    # 计算y列表中的最小值，判断大概的初始位置
    min_x = min(listx_2)
    min_y = min(listy_2)

    # 因为边缘有一些点太少，经过多次尝试，初始位置从40开始（可调整）

    first_match = 40
    j = min_y + first_match
    first = j
    k = 0
    # 统计符合条件的向量元素个数
    while j:
        for i in listy_2:
            if (i == j):
                vector2[k] += 1
        k += 1
        j += 1
        if (j == (first + vector2_len)):
            break

    # 找到相应y的所有x坐标
    list_xx = []
    q = 1
    for d in listx_2:
        if (d == min_y + first_match):
            list_xx.append(listy_2[q])
        q += 1
    # 计算所有y坐标的中位数
    q_2 = np.median(list_xx)
    #
    list_max = max(listy_1)
    # 边缘范围为%4-%5，所以初始位置为x坐标最大值减去120
    first = list_max - 80
    # 初始位置备份
    first_y = first
    cos_s = []
    # 循环控制变量
    k = 0
    j = first
    z = 1

    # 向量循环匹配
    while z:
        # 统计符合条件的向量元素个数
        while j:
            for i in listy_1:
                if (i == j):
                    vector1[k] += 1
            k += 1
            j += 1
            if (j == (first + vector1_len)):
                break
        # 每一次计算的余弦相似度，存放在cos_s列表里
        cos = 1 - spatial.distance.cosine(vector1, vector2)
        # 显示向量
        cos_s.append(cos)
        # 更新循环控制变量
        z += 1
        k = 0
        first += 1
        j = first
        i = 0
        # 向量置空
        vector1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 这里的向量元素个数为10
        if ((z + vector1_len + first_y) > list_max):
            break

    # 计算cos_s中最大相似度并返回其下标
    indexx = cos_s.index(max(cos_s))

    if(max(cos_s) < 0.85):
        print("两张图无法拼接")
        return 404, 404
    # 计算y坐标的偏移量
    y_displace = indexx + 1 + first_y
    # 找到相应y的所有x坐标
    #####################################################
    #测试内容
    temp = y_displace
    flag = 0
    list_similar = []
    list_index = []
    for n in range(listy_1.count(temp)):
        sec = flag
        flag = listy_1[flag:].index(temp)
        list_index.append(flag + sec)
        flag = list_index[-1:][0] + 1

    for ind in list_index:
        list_similar.append(listx_1[ind])

    temp = first_match
    flag = 0
    list_similar2 = []
    list_index2 = []
    for n in range(listy_2.count(temp)):
        sec = flag
        flag = listy_2[flag:].index(temp)
        list_index2.append(flag + sec)
        flag = list_index2[-1:][0] + 1

    for ind3 in list_index2:
        list_similar2.append(listx_2[ind3])

    len1 = len(list_similar)-1
    len2 = len(list_similar2)-1
    if len1 >= len2:
        list_similar = list_similar[0:len2]
    else:
        list_similar2 = list_similar2[0:len1]


    def diss():
        l = list_similar
        l2 = list_similar2
        l2 = sorted(list(set(l2)))
        l = sorted(list(set(l)))
        list_vcr = [1] * 2000
        list_vcr2 = [1] * 2000
        cha = min(listx_1)
        l = [ii - cha for ii in l]
        l2 = [jj - cha for jj in l2]
        for kk in l:
            list_vcr[kk] = 1000

        for kkk in l2:
            list_vcr2[kkk] = 1000

        start = 70
        end = 1900

        list_compare2 = list_vcr2[70:1900]
        maxx = 0
        while 1:
            list_compare1 = list_vcr[start:end]
            cos_cc = 1 - spatial.distance.cosine(list_compare1, list_compare2)
            if (cos_cc > maxx):
                maxx = cos_cc
                max_set = start
            start += 1
            end += 1
            if end > len(list_vcr):
                break
            down_point = max_set

        x_displacement = down_point - 50 + cha
        return x_displacement

    ################################################################

    list_x = []
    q = 0
    for d in listy_1:
        if (d == (y_displace + len(vector1))):
            list_x.append(listx_1[q])
        q += 1
    # 对所有x坐标取中位数
    p_1 = np.median(list_x)
    x_displace = p_1 - q_2

    x_displace += x_adjust
    y_displace += y_adjust
    result = pd.value_counts(listx_1)

    x_displace = diss()
    return x_displace, y_displace

#------------------------------------------水平方向两幅图拼接----------------------------------------
def lateral_stitch(Image_i, Image_j):
    picture_left = path + 'img' + str(Image_i) + '.txt'
    picture_right = path + 'img' + str(Image_j) + '.txt'
    x_adjust = 0
    y_adjust = 0

    # 打开拼接的第一幅图（左边）
    f = codecs.open(picture_left, mode='rb')
    line_temp = f.read(12 * 4)  # 读取单个定位点信息/单个定位点有12个定位信息，每个信息占4个字节
    line = np.empty((1, 12))
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
    f.close()

    vector1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vector1_len = len(vector1)

    ff = codecs.open(picture_right, mode='rb')
    line_temp = ff.read(12 * 4)  # 读取单个定位点信息/单个定位点有12个定位信息，每个信息占4个字节
    line = np.empty((1, 12))
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
    ff.close()

    vector2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    vector2_len = len(vector2)
    # 计算x列表中的最小值，判断大概的初始位置
    min_x = min(list11)
    # 因为边缘有一些点太少，经过多次尝试，初始位置从20开始（可调整）
    first_match = 20
    j = min_x + first_match
    first = j
    k = 0
    # 统计符合条件的向量元素个数
    while j:
        for i in list11:
            if (i == j):
                vector2[k] += 1
        k += 1
        j += 1
        if (j == (first + vector2_len)):
            break

    # 找到相应x的所有y坐标
    list_yy = []
    q = 1
    for d in list11:
        if (d == min_x + first_match):
            list_yy.append(list22[q])
        q += 1
    # 计算所有y坐标的中位数
    q_2 = np.mean(list_yy)

    list_max = max(list1)
    # 边缘范围为%4-%5，所以初始位置为x坐标最大值减去70
    first = list_max - 70
    # 初始位置备份
    first_x = first
    cos_s = []
    # 循环控制变量
    k = 0
    j = first
    z = 1
    # 向量循环匹配
    while z:
        # 统计符合条件的向量元素个数
        while j:
            for i in list1:
                if (i == j):
                    vector1[k] += 1
            k += 1
            j += 1
            if (j == (first + vector1_len)):
                break
        # 每一次计算的余弦相似度，存放在cos_s列表里
        cos = 1 - spatial.distance.cosine(vector1, vector2)
        # 显示向量
        cos_s.append(cos)
        # 更新循环控制变量
        z += 1
        k = 0
        first += 1
        j = first
        i = 0
        # 向量置空
        vector1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 这里的向量元素个数为10
        if ((z + vector1_len + first_x) > list_max):
            break

    # 计算cos_s中最大相似度并返回其下标
    indexx = cos_s.index(max(cos_s))

    if (max(cos_s) < 0.85):
        print("两张图无法拼接")
        return 404, 404
    # 计算x坐标的偏移量
    x_displace = indexx + 1 + first_x
    # 找到相应x的所有y坐标
    ################################################################
    # 测试
    temp = x_displace
    flag = 0
    list_similar = []
    list_index = []
    for n in range(list1.count(temp)):
        sec = flag
        flag = list1[flag:].index(temp)
        list_index.append(flag + sec)
        flag = list_index[-1:][0] + 1

    for ind in list_index:
        list_similar.append(list2[ind])

    temp = first_match
    flag = 0
    list_similar2 = []
    list_index2 = []
    for n in range(list11.count(temp)):
        sec = flag
        flag = list11[flag:].index(temp)
        list_index2.append(flag + sec)
        flag = list_index2[-1:][0] + 1

    for ind3 in list_index2:
        list_similar2.append(list22[ind3])

    def diss():
        l = list_similar
        l2 = list_similar2
        l2 = sorted(list(set(l2)))
        l = sorted(list(set(l)))

        len1 = len(l) - 1
        len2 = len(l2) - 1
        if len1 >= len2:
            l = l[0:len2]
        else:
            l2 = l[0:len1]
        list_vcr = [1] * 1046
        list_vcr2 = [1] * 1046
        cha = min(list2)
        l = [ii - cha for ii in l]
        l2 = [jj - cha for jj in l2]
        for kk in l:
            list_vcr[kk] = 100

        for kkk in l2:
            list_vcr2[kkk] = 100

        start = 40
        end = 1000

        list_compare2 = list_vcr2[40:1000]
        maxx = 0
        while 1:
            list_compare1 = list_vcr[start:end]
            cos_cc = 1 - spatial.distance.cosine(list_compare1, list_compare2)
            if (cos_cc > maxx):
                maxx = cos_cc
                max_set = start
            start += 1
            end += 1
            if end > len(list_vcr):
                break
            left_point = max_set

        y_displacement = left_point - 50 + cha
        return y_displacement

    ################################################################
    list_y = []
    q = 0

    for d in list1:
        if (d == (x_displace + len(vector1))):
            list_y.append(list2[q])
        q += 1
    # 对所有y坐标取中位数
    p_1 = np.median(list_y)
    y_displace = p_1 - q_2
    x_displace += x_adjust
    y_displace += y_adjust
    result = pd.value_counts(list1)
    y_displace = diss()

    return x_displace, y_displace

#-------------------------------------------大图拼接--------------------------------------------------

#构造邻接矩阵/相邻的图像可以联通
def CompressTableMatrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        x = int(i/Image_width)#计算该图的XY序号
        y = i%Image_width
        XYaround = np.zeros((4,2))#存储四周的图序号
        XYaround[0][0] = x - 1
        XYaround[0][1] = y
        XYaround[1][0] = x
        XYaround[1][1] = y - 1
        XYaround[2][0] = x + 1
        XYaround[2][1] = y
        XYaround[3][0] = x
        XYaround[3][1] = y + 1
        ListAround = []
        ListAround.append(Image_width * XYaround[0][0] + XYaround[0][1])
        ListAround.append(Image_width * XYaround[1][0] + XYaround[1][1])
        ListAround.append(Image_width * XYaround[2][0] + XYaround[2][1])
        ListAround.append(Image_width * XYaround[3][0] + XYaround[3][1])

        for j in range(4):
            if (XYaround[j][0]>=0 and XYaround[j][0]<Image_length and
                XYaround[j][1]>=0 and XYaround[j][1]<Image_width):
                A[i][int(ListAround[j])] = 1
                A[int(ListAround[j])][i] = 1
    return A

#广度优先搜索
head = 0#队列头
tail = 0#队列尾
queue = []
queue.append(0)#向头中加入图的第一个节点
head = head + 1#队列扩展
flag = []#标记节点是否被访问
flag.append(0)
re = []#存入连接的边
XY_move = np.zeros((fileNum,2))#偏移量
A = CompressTableMatrix(fileNum)#构造邻接矩阵

while tail!=head:
    i = queue[tail]
    for j in range(fileNum):
        if (A[i][j]==1 and not(j in flag)):#判断是否邻接且未入队
            """
            拼接代码
            如果无法拼接返回stitch=-1,-1
            拼接能否成功的判断标准：1、是否有连通区域 2、有连通区域，拼接效果评价
            """
            x1 = int(i / Image_width)
            y1 = i % Image_width
            x2 = int(j / Image_width)
            y2 = j % Image_width
            if x1==x2: # 横向拼接
                if y1<y2:
                    x_temp, y_temp = lateral_stitch(i, j)
                else:
                    x_temp, y_temp = lateral_stitch(j, i)
                    x_temp = -x_temp
                    y_temp = -y_temp
            if y1==y2:# 纵向拼接
                if x1<x2:
                    x_temp, y_temp = straight_stitch(i, j)
                else:
                    x_temp, y_temp = straight_stitch(j, i)
                    x_temp = -x_temp
                    y_temp = -y_temp

            if (x_temp==404 and y_temp==404): continue#拼接失败

            queue.append(j)#如果成功拼接，节点入队
            head = head + 1
            flag.append(j)
            Edge = [i,j]
            re.append(Edge)#存入连接的边
            XY_move[j][0] = XY_move[i, 0] + x_temp
            XY_move[j][1] = XY_move[i, 1] + y_temp
            print("图像序号/累计偏移量")
            print(j, XY_move[j][0], XY_move[j][1])
    tail = tail + 1
queue.sort()
print("被搜过到的图像序列")
print(queue)

#保存偏移量，可配合读取渲染程序生成拼接图像
a = open(MovingDistance_stitch, 'w')
for i in XY_move:
    a.write(str(i[0]) +'\t')
    a.write(str(i[1]) +'\n')
a.close

plt.xlim(xmax=5000, xmin=-100)
plt.ylim(ymax=5000, ymin=-100)
for num in queue:

    Single_Image = codecs.open(path+'img'+str(num)+'.txt', mode='rb')
    line_temp = Single_Image.read(12 * 4)
    line = np.empty((1, 12))
    line = struct.unpack("12f", line_temp)
    list11 = []
    list22 = []

    while 1:
        x = line[1:2]
        y = line[2:3]
        x = math.ceil(float(x[0])) + XY_move[num, 0]
        y = math.ceil(float(y[0])) + XY_move[num, 1]
        list11.append(x)
        list22.append(y)
        line_temp = Single_Image.read(12 * 4)
        if len(line_temp) < 48:
            break
        line = struct.unpack("12f", line_temp)

    Single_Image.close()
    # 作图，打点方式为像素点，600dpi，可以设置保存路径
    plt.plot(list11, list22, 'ro', c='black', alpha=0.1, marker=',')

plt.savefig(stitch, dpi=600)
plt.show()

"""
说明：1、如果头结点图像周围没有连接图像，容易陷入局部最优/是否有必要加入初始头结点位置选择
     2、广度优先搜索相对于深度优先搜索能够相对减少连接链路的累计误差
"""