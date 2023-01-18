import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# 导入数据
df = pd.read_csv('pica2015.csv', low_memory=False, na_values=' ')
math = pd.read_csv('label_train.csv', low_memory=False, na_values=' ')
# 删除str
data = df.drop(columns=['STRATUM', 'Option_Read', 'Option_Math'])
# 处理缺失值、异常值
data.fillna(data.mode(), inplace=True)  # 填充
# 数据类型转换
data = data.astype(float)
math = math.astype(float)
# 合并原数据集与math
data_pre = data.loc[range(6426), :]  # 用于预测的数据集
data = pd.merge(data, math, how='inner', on='Unnamed: 0')
data = data.drop('index', axis=1)


# 遍历data找和math相关的特征
char_list = []  # 用来装挑出来的特征
user_col = list(data.columns.values)
for cols in user_col:
    if abs(data[[cols, 'MATH']].corr(method='pearson').iloc[0, 1]) >= 0.5:
        char_list.append(cols)
data = data.loc[:, char_list]  # 保留所有选中的特征和math
char_list.pop()
data_pre = data_pre.loc[:, char_list]  # 保留所有选中的特征
# print(data)
# print(data_pre)


# 交叉验证
def cross_validation(data, k, t):  # data为数据集，k为划分的数量, t是第几次交叉验证
    n = data.shape[0]  # 总行数
    interval = []  # 存放数据集的index间隔
    global test_
    global train_
    for i in range(k+1):
        interval.append(int(n*i/k))
    if t == 1:
        test_ = data.loc[range(interval[t - 1], interval[t]), :]
        train_ = data.loc[range(interval[t], interval[k]), :]
    if t == 2:
        test_ = data.loc[range(interval[t - 1], interval[t]), :]
        train_ = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 3:
        test_ = data.loc[range(interval[t - 1], interval[t]), :]
        train_ = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 4:
        test_ = data.loc[range(interval[t - 1], interval[t]), :]
        train_ = pd.concat([data.loc[range(interval[0], interval[t - 1]), :], data.loc[range(interval[t], interval[k]), :]])
    if t == 5:
        test_ = data.loc[range(interval[t - 1], interval[t]), :]
        train_ = data.loc[range(interval[0], interval[t - 1]), :]
    return test_, train_


# 设置参数
forest = RandomForestRegressor(n_estimators=500,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)
neighbors = KNeighborsClassifier(n_neighbors=1200)
clf = GaussianNB()
reg = LinearRegression()
# mse评估回归效果
k_rate = 0.0
f_rate = 0.0
b_rate = 0.0
l_rate = 0.0
for i in range(1, 7):
    test, train = cross_validation(data, 6, i)
    x_train = train.iloc[:, :-1].values
    y_train = train['MATH'].values
    x_test = test.iloc[:, :-1].values
    y_test = test['MATH'].values
    # KNN
    # 下面这加了astype是因为y值必须是int或者str 我这里懒得转str了就直接int 应该问题不大
    neighbors.fit(x_train.astype('int32'), y_train.astype('int32'))
    y_train_pre_k = neighbors.predict(x_train.astype('int32'))  # 这个是预测值
    y_test_pre_k = neighbors.predict(x_test.astype('int32'))
    k_mse = mean_squared_error(y_test, y_test_pre_k)
    k_rate += k_mse
    # forest
    forest.fit(x_train, y_train)
    y_train_pre_f = forest.predict(x_train)
    y_test_pre_f = forest.predict(x_test)
    f_mse = mean_squared_error(y_test, y_test_pre_f)
    f_rate += f_mse
    # bayes
    clf.fit(x_train.astype('int32'), y_train.astype('int32'))
    y_train_pre_b = clf.predict(x_train.astype('int32'))
    y_test_pre_b = clf.predict(x_test.astype('int32'))
    b_mse = mean_squared_error(y_test, y_test_pre_b)
    b_rate += b_mse
    # 线性回归
    reg.fit(x_train, y_train)
    y_train_pre_l = reg.predict(x_train)
    y_test_pre_l = reg.predict(x_test)
    l_mse = mean_squared_error(y_test, y_test_pre_l)
    l_rate += l_mse


print('KNN_mse:', k_rate/6)
print('forest_mse:', f_rate/6)
print('Bayes_mse:', b_rate/6)
print('LinearRegression_mse:', l_rate/6)


# 线性回归预测前6426个人的math
reg = LinearRegression()
reg.fit(data.iloc[:, :-1].values, data['MATH'].values)
result = reg.predict(data_pre)
index = [i for i in range(6426)]
dataframe = pd.DataFrame({'index': index, 'MATH': result})
dataframe.to_csv('pred_result.csv')
