import pandas as pd
import random

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
data = data.loc[:, char_list]  # 保留所有选中的特征和repeat
# 先只选前三个离散型的数据来操作
new_char = [char_list[0], char_list[1], char_list[2], char_list[-1]]
data = data.loc[:, new_char]
data.loc[data['ST127Q01TA'].isin([9]), 'ST127Q01TA'] = 0  # 把‘ST127Q01TA’的9换成0


'''
如果用两个复读的特征：
char_list = ['ST127Q01TA', 'ST127Q02TA', 'REPEAT']
data = data.loc[:, char_list]
data.loc[data['ST127Q01TA'].isin([9]), 'ST127Q01TA'] = 0
data.loc[data['ST127Q02TA'].isin([9]), 'ST127Q02TA'] = 0
'''


# 随机打乱原数据集
def rand_disruption(data):
    index = list(data.index)
    random.shuffle(index)
    data.index = index
    return data   # 返回打乱后的数据集


# 划分数据集并交叉验证
def cross_validation(data, k, t):  # data为数据集，k为划分的数量(本题中为5), t是第几次交叉验证
    n = data.shape[0]  # 总行数
    interval = []  # 存放数据集的index间隔
    for i in range(k+1):
        interval.append(int(n*i/k))
    if t == 1:
        test = data.loc[range(interval[t - 1], interval[t]), :]
        train = data.loc[range(interval[t], interval[k]), :]
    if t == 2:
        test = data.loc[range(interval[t - 1], interval[t]), :]
        train = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 3:
        test = data.loc[range(interval[t - 1], interval[t]), :]
        train = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 4:
        test = data.loc[range(interval[t - 1], interval[t]), :]
        train = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 5:
        test = data.loc[range(interval[t - 1], interval[t]), :]
        train = data.loc[range(interval[0], interval[t - 1]), :]
    return test, train


# 朴素贝叶斯
def naive_bayes(train, test):
    repeat = [0, 0]  # 该数组存放repeat的数量，第一个位置放不复读的，第二个放复读的
    ch1_re0 = [0, 0, 0, 0, 0]  # 放特征1的各值的数量。五个位置分别放ch1=7,8,9,10,11 在repeat=0的条件下
    ch2_re0 = [0, 0, 0, 0]  # 同上，ch2=0,1,2,3  0代表空值 即9
    ch3_re0 = [0, 0]  # ch3=0,1
    ch1_re1 = [0, 0, 0, 0, 0]  # 同上，但是是repeat=1的条件下
    ch2_re1 = [0, 0, 0, 0]
    ch3_re1 = [0, 0]
    n = train.shape[0]  # 训练集总数据量
    # 统计频数
    for index, row in train.iterrows():  # 按行遍历
        for i in range(7, 12):
            for j in range(4):
                for k in range(2):
                    if row['ST001D01T'] == i and row['ST127Q01TA'] == j and row['ST063Q02NB'] == k:
                        if row['REPEAT'] == 0:
                            repeat[0] += 1
                            ch1_re0[i - 7] += 1
                            ch2_re0[j] += 1
                            ch3_re0[k] += 1
                        elif row['REPEAT'] == 1:
                            repeat[1] += 1
                            ch1_re1[i - 7] += 1
                            ch2_re1[j] += 1
                            ch3_re1[k] += 1
    # 算概率
    p_re = [repeat[0] / (repeat[0] + repeat[1]), repeat[1] / (repeat[0] + repeat[1])]  # repeat = 0,1的概率
    p_ch1_re0 = [ch1_re0[0] / n / p_re[0], ch1_re0[1] / n / p_re[0], ch1_re0[2] / n / p_re[0], ch1_re0[3] / n / p_re[0],
                 ch1_re0[4] / n / p_re[0]]
    p_ch1_re1 = [ch1_re1[0] / n / p_re[1], ch1_re1[1] / n / p_re[1], ch1_re1[2] / n / p_re[1], ch1_re1[3] / n / p_re[1],
                 ch1_re1[4] / n / p_re[1]]
    p_ch2_re0 = [ch2_re0[0] / n / p_re[0], ch2_re0[1] / n / p_re[0], ch2_re0[2] / n / p_re[0], ch2_re0[3] / n / p_re[0]]
    p_ch2_re1 = [ch2_re1[0] / n / p_re[1], ch2_re1[1] / n / p_re[1], ch2_re1[2] / n / p_re[1], ch2_re1[3] / n / p_re[1]]
    p_ch3_re0 = [ch3_re0[0] / n / p_re[0], ch3_re0[1] / n / p_re[0]]
    p_ch3_re1 = [ch3_re1[0] / n / p_re[1], ch3_re1[1] / n / p_re[1]]
    # 测试
    correct = 0  # 预测正确的数量
    for index, row in test.iterrows():
        for i in range(7, 12):
            for j in range(4):
                for k in range(2):
                    if row['ST001D01T'] == i and row['ST127Q01TA'] == j and row['ST063Q02NB'] == k:
                        if p_re[0]*p_ch1_re0[i - 7]*p_ch2_re0[j]*p_ch3_re0[k] > p_re[1]*p_ch1_re1[i - 7]*p_ch2_re1[j]*\
                                p_ch3_re1[k]:
                            if row['REPEAT'] == 0:
                                correct += 1
                        else:
                            if row['REPEAT'] == 1:
                                correct += 1
    rate = correct / test.shape[0]
    return rate
'''
如果用两个复读的特征：
ch1_re0,ch1_re1,ch2_re0,ch2_re1都是1x4的列表
循环只需要两层for in range(4)
'''


# main
data = rand_disruption(data)
rate = 0.0
for t in range(1, 6):
    test, train = cross_validation(data, 5, t)
    print(naive_bayes(train, test))
    rate += naive_bayes(train, test)
print(rate/5)
