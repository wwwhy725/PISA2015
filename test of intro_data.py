import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
# 导入数据
df = pd.read_csv('pica2015.csv', low_memory=False, na_values=' ')
# 删除str
data = df.drop(columns=['STRATUM', 'Option_Read', 'Option_Math'])
# 处理缺失值、异常值
data.fillna(data.mode(), inplace=True)  # 填充
# 数据类型转换
data = data.astype(float)

# 遍历data找和repeat相关的特征
char_list = []  # 用来装挑出来的特征
user_col = list(data.columns.values)
for cols in user_col:
    if abs(data[[cols, 'REPEAT']].corr(method='pearson').iloc[0, 1]) >= 0.5:
        char_list.append(cols)

# 画repeat的饼图
a1 = pd.to_numeric(data['REPEAT'])
Repeat = a1.tolist()
sigma = 0
for i in Repeat:
    if i == 1.0:
        sigma += 1
p1 = np.array([sigma, 32120-sigma])
plt.pie(p1, labels=[1, 0], autopct='%.2f%%')
plt.title('REPEAT pie chart')
plt.show()

# 画非pv值的分布图

# ST001D01T
# 箱图
a2 = pd.to_numeric(data[char_list[0]])
p2 = a2.tolist()
plt.boxplot(p2, showmeans=True)
plt.title('ST001D01T box plot')
plt.grid(True)
plt.show()
# repeat_rate的关系
counter = [0, 0, 0, 0, 0]
repeat = [0, 0, 0, 0, 0]
for index, row in data.iterrows():
    if data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 7.0 and data.iloc[index, 426] == 1:
        counter[0] += 1
        repeat[0] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 7.0 and data.iloc[index, 426] == 0.0:
        counter[0] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 8.0 and data.iloc[index, 426] == 1.0:
        counter[1] += 1
        repeat[1] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 8.0 and data.iloc[index, 426] == 0.0:
        counter[1] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 9.0 and data.iloc[index, 426] == 1.0:
        counter[2] += 1
        repeat[2] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 9.0 and data.iloc[index, 426] == 0.0:
        counter[2] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 10.0 and data.iloc[index, 426] == 1.0:
        counter[3] += 1
        repeat[3] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 10.0 and data.iloc[index, 426] == 0.0:
        counter[3] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 11.0 and data.iloc[index, 426] == 1.0:
        counter[4] += 1
        repeat[4] += 1
    elif data.iloc[index, df.columns.get_loc('ST001D01T')-3] == 11.0 and data.iloc[index, 426] == 0.0:
        counter[4] += 1
rate = []
for i in range(len(counter)):
    rate.append(repeat[i]/counter[i])
plt.plot([7, 8, 9, 10, 11], rate)
plt.grid(True)
plt.title('ST001D01T-Repeat_rate')
plt.xlabel('Grade')
plt.ylabel('Rate of Repeat')
for i in range(5):
    plt.text(i+7, rate[i], round(rate[i], 3))
plt.show()

# ST127Q01TA
# 统计缺失值(9)
index_row = []  # 用于存放空值的行号
for index, row in data.iterrows():
    if data.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 9.0:
        index_row.append(index)
num_of_9 = len(index_row)  # 缺失值数量
data_drop = data.drop(index_row, axis=0)  # 删掉缺失值之后的数据集
# 饼图
a3 = pd.to_numeric(data_drop[char_list[1]])
a3 = a3.tolist()
p3 = [0, 0, 0]
for i in a3:
    if i == 1.0:
        p3[0] += 1
    elif i == 2.0:
        p3[1] += 1
    elif i == 3.0:
        p3[2] += 1
plt.pie(p3, labels=[1, 2, 3], autopct='%.2f%%')
plt.title('ST127Q01TA pie chart')
plt.show()
# 与repeat_rate的关系
counter2 = [0, 0, 0]
repeat2 = [0, 0, 0]
rate2 = []
for index in range(32120-num_of_9):
    if data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 1.0 and data_drop.iloc[index, 426] == 1.0:
        counter2[0] += 1
        repeat2[0] += 1
    elif data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 1.0 and data_drop.iloc[index, 426] == 0.0:
        counter2[0] += 1
    elif data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 2.0 and data_drop.iloc[index, 426] == 1.0:
        counter2[1] += 1
        repeat2[1] += 1
    elif data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 2.0 and data_drop.iloc[index, 426] == 0.0:
        counter2[1] += 1
    elif data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 3.0 and data_drop.iloc[index, 426] == 1.0:
        counter2[2] += 1
        repeat2[2] += 1
    elif data_drop.iloc[index, df.columns.get_loc('ST127Q01TA')-3] == 3.0 and data_drop.iloc[index, 426] == 0.0:
        counter2[2] += 1
for i in range(len(counter2)):
    rate2.append(repeat2[i]/counter2[i])
plt.plot([1, 2, 3], rate2)
plt.grid(True)
plt.title('ST127Q01TA-Repeat_rate')
plt.xlabel('whether repeated')
plt.ylabel('Rate of Repeat')
for i in range(3):
    plt.text(i+1, rate2[i], round(rate2[i], 3))
plt.show()

# ST063Q02NB
# 统计缺失值（之前那个遍历找空值的方法太愚蠢了！！但还是决定留在那展现真实的实验过程）
num_of_nan = df['ST063Q02NB'].isnull().sum()
data_drop2 = df[['ST063Q02NB', 'REPEAT']].dropna()
# 饼图
a4 = pd.to_numeric(data_drop2[char_list[2]])
a4 = a4.tolist()
p4 = [0, 0]
for i in a4:
    if i == 0.0:
        p4[0] += 1
    elif i == 1.0:
        p4[1] += 1
plt.pie(p4, labels=[0, 1], autopct='%.2f%%')
plt.title('ST063Q02NB pie chart')
plt.show()
# 与repeat_rate的关系
counter3 = [0, 0]
repeat3 = [0, 0]
rate3 = []
for i in range(32120-num_of_nan):
    if data_drop2.iloc[i, 0] == 0.0 and data_drop2.iloc[i, 1] == 1.0:
        counter3[0] += 1
        repeat3[0] += 1
    elif data_drop2.iloc[i, 0] == 0.0 and data_drop2.iloc[i, 1] == 0.0:
        counter3[0] += 1
    elif data_drop2.iloc[i, 0] == 1.0 and data_drop2.iloc[i, 1] == 1.0:
        counter3[1] += 1
        repeat3[1] += 1
    elif data_drop2.iloc[i, 0] == 1.0 and data_drop2.iloc[i, 1] == 0.0:
        counter3[1] += 1
for i in range(len(counter3)):
    rate3.append(repeat3[i]/counter3[i])
plt.plot([0, 1], rate3)
plt.grid(True)
plt.title('ST063Q02NB-Repeat_rate')
plt.xlabel('Checked or Not')
plt.ylabel('Rate of Repeat')
for i in range(2):
    plt.text(i, rate3[i], round(rate3[i], 3))
plt.show()

# PV
char_list.remove('ST001D01T')
char_list.remove('ST127Q01TA')
char_list.remove('ST063Q02NB')
data_ = data[char_list]
data_va = data_.values
std = MinMaxScaler()
data_std = std.fit_transform(data_va)
# 离散化
nr, nc = data_std.shape
interval = [0, 0, 0, 0, 0]
count = [0, 0, 0, 0, 0]
for i in range(nr):
    for j in range(nc-1):
        if data_std[i][j] <= 0.2:
            if data_std[i][nc-1] == 1.0:
                interval[0] += 1
                count[0] += 1
            else:
                interval[0] += 1
        elif data_std[i][j] <= 0.4:
            if data_std[i][nc - 1] == 1.0:
                interval[1] += 1
                count[1] += 1
            else:
                interval[1] += 1
        elif data_std[i][j] <= 0.6:
            if data_std[i][nc - 1] == 1.0:
                interval[2] += 1
                count[2] += 1
            else:
                interval[2] += 1
        elif data_std[i][j] <= 0.8:
            if data_std[i][nc - 1] == 1.0:
                interval[3] += 1
                count[3] += 1
            else:
                interval[3] += 1
        else:
            if data_std[i][nc-1] == 1.0:
                interval[4] += 1
                count[4] += 1
            else:
                interval[4] += 1
for i in range(len(interval)):
    interval[i] = count[i]/interval[i]
# 绘制折线图
plt.plot([0.1, 0.3, 0.5, 0.7, 0.9], interval)
plt.grid(True)
plt.title('PV-Repeat_rate')
plt.xlabel('PV')
plt.ylabel('Rate of Repeat')
for i in range(5):
    plt.text(i/5+0.1, interval[i], round(interval[i], 3))
plt.show()

# PVSCIE与ST131Q01NA
# 处理缺失值
index_row1 = []  # 用于存放空值的行号
character = ['ST131Q01NA', 'PV1SCIE', 'PV2SCIE', 'PV3SCIE', 'PV4SCIE', 'PV5SCIE', 'PV6SCIE', 'PV7SCIE', 'PV8SCIE', 'PV9SCIE',
             'PV10SCIE']
for index, row in data.iterrows():
    if data.iloc[index, df.columns.get_loc('ST131Q01NA')-3] == 9.0:
        index_row1.append(index)
num_of_nan1 = len(index_row1) + data['ST131Q01NA'].isnull().sum()  # 缺失值数量
data_drop3 = data.drop(index_row1, axis=0)[character].dropna()  # 删掉缺失值之后的数据集
# ST131Q01NA的箱图
a5 = pd.to_numeric(data_drop3['ST131Q01NA'])
p5 = a5.tolist()
plt.boxplot(p5, showmeans=True)
plt.title('ST131Q01NA box plot')
plt.grid(True)
plt.show()

# 规范化
data_va2 = data_drop3.values
std = MinMaxScaler()
data_std2 = std.fit_transform(data_va2)
ST_count = [0, 0, 0, 0, 0]
PV_count = [0, 0, 0, 0, 0]
nr2, nc2 = data_std2.shape
for i in range(nr2):
    for j in range(1, nc2):
        if data_std2[i][j] <= 0.2:
            ST_count[0] += data_std2[i][0]
            PV_count[0] += 1
        elif data_std2[i][j] <= 0.4:
            ST_count[1] += data_std2[i][0]
            PV_count[1] += 1
        elif data_std2[i][j] <= 0.6:
            ST_count[2] += data_std2[i][0]
            PV_count[2] += 1
        elif data_std2[i][j] <= 0.8:
            ST_count[3] += data_std2[i][0]
            PV_count[3] += 1
        else:
            ST_count[4] += data_std2[i][0]
            PV_count[4] += 1
for i in range(5):
    ST_count[i] = ST_count[i]/PV_count[i]
# 绘制折线图
plt.plot([0.1, 0.3, 0.5, 0.7, 0.9], ST_count)
plt.grid(True)
plt.title('PVSCIE-ST131Q01NA')
plt.xlabel('PV')
plt.ylabel('ST131Q01NA')
for i in range(5):
    plt.text(i/5+0.1, ST_count[i], round(ST_count[i], 3))
plt.show()


# 组合特征
# 四个特征和repeat放一起并扔掉空值
data_drop4 = data[['ST123Q01NA', 'ST123Q02NA', 'ST123Q03NA', 'ST123Q04NA', 'REPEAT']].\
    replace(9.0, np.NaN).replace(2.0, math.exp(1)).replace(3.0, math.exp(2)).replace(4.0, math.exp(3))
data_drop4 = data_drop4.dropna()
# 组合
corr1 = abs(data_drop4[['ST123Q01NA', 'REPEAT']].corr(method='pearson').iloc[0, 1])
corr2 = abs(data_drop4[['ST123Q02NA', 'REPEAT']].corr(method='pearson').iloc[0, 1])
corr3 = abs(data_drop4[['ST123Q03NA', 'REPEAT']].corr(method='pearson').iloc[0, 1])
corr4 = abs(data_drop4[['ST123Q04NA', 'REPEAT']].corr(method='pearson').iloc[0, 1])
cor = [corr1, corr2, corr3, corr4]
data_drop4['new_char'] = 0.4*data_drop4['ST123Q01NA'] + 0.3*data_drop4['ST123Q02NA'] +\
                         0.2*data_drop4['ST123Q03NA'] + 0.1*data_drop4['ST123Q04NA']
a6 = data_drop4[['new_char', 'REPEAT']].values
nr3, nc3 = a6.shape
new_count = [0, 0]
re_count = [0, 0]
for i in range(nr3):
    if a6[i][1] == 1.0:
        new_count[1] += a6[i][0]
        re_count[1] += 1
    else:
        new_count[0] += a6[i][0]
        re_count[0] += 1
total = re_count[0] + re_count[1]
new_mean = [new_count[0]/re_count[0], new_count[1]/re_count[1]]
# 新变量的均值关于复读的柱状图
plt.bar([0, 1], new_mean, width=0.6)
plt.xlabel('REPEAT')
plt.ylabel('Mean of new_R')
plt.show()
# 箱图
plt.boxplot(pd.to_numeric(data_drop4['new_char']).tolist(), showmeans=True)
plt.title('new_R box plot')
plt.grid(True)
plt.show()

# 以bound为界，看较大值与repeat的关系
bound = 13.0
new_count1 = [0, 0]  # 第一个位置放大且复读的，第二个放大且不复读的
re_count1 = [0, 0]  # 第一个放小且复读的，第二个放小且不复读的
for i in range(nr3):
    if a6[i][0] >= bound:
        if a6[i][1] == 1.0:
            new_count1[0] += 1
        else:
            new_count1[1] += 1
    else:
        if a6[i][1] == 1.0:
            re_count1[0] += 1
        else:
            re_count1[1] += 1
# 画饼图
plt.pie(new_count1, labels=[1, 0], autopct='%.2f%%')
plt.title('Higher new_R pie chart')
plt.show()
