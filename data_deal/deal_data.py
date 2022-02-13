import numpy as np
import pandas as pd
# from sklearn.impute import SimpleImputer
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


##
#  filePath               处理的文件目录
#  conditionPath          表，对应每个数据特征条件注明 0：删除该特征数据 1：连续数据  2：离散数据
#  train_prob             保留使用的训练数据占比
#  test_prob              保留使用的测试数据占比
#  test_len               原始数据划分，测试数据所占比率
##
def deal_data(filePath, conditionPath=None, train_prob=0.002, test_prob=0.003, test_len=0.25):
    data = pd.read_csv(filePath)
    nameKeys = data.keys()
    print(nameKeys)

    # 某个特征是离散 还是 连续
    identify_or_embedding = []
    # 离散数据的特征范围
    orgin_dims = []
    # Embedding后数据的特征范围
    target_dims = []

    if conditionPath != None:
        condition = pd.read_csv(conditionPath)
        for name in nameKeys[:-1]:
            c = condition[name].values[0]
            if c == 0:
                data = data.drop(name, 1)
            elif c == 2:
                identify_or_embedding.append(True)
            else:
                identify_or_embedding.append(False)

        nameKeys = data.keys()
        for name, cur in zip(nameKeys[:-1], identify_or_embedding):
            if cur == False:
                orgin_dims.append(1)
                target_dims.append(1)
            else:
                temp = len(set(data[name]))
                orgin_dims.append(temp)
                target_dims.append(temp)
    else:
        for name in nameKeys[:-1]:
            orgin_dims.append(1)
            target_dims.append(1)

    ## 处理非数字数据
    for name in nameKeys[:-1]:
        if str(data[name].dtype) == 'object':
            range_values = list(set(data[name].values))
            for i in range(len(range_values)):
                data.loc[data[name] == range_values[i], name] = i

    # imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    # hd = pd.DataFrame(imputer.fit_transform(data))
    # hd.columns = data.columns
    hd = data

    #####归一化函数#####
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    for name, cur in zip(nameKeys, identify_or_embedding):
        if cur == False:
            if max(hd[name]) > 10:
                hd[[name]] = hd[[name]].apply(max_min_scaler)

    # 特征
    feature_names = nameKeys[:-1]
    label = set(hd[nameKeys[-1]])
    print(label)

    X = hd.iloc[:, :-1].values.astype('float32')

    # split data
    featureX = []
    featureY = []
    for i in label:
        f = hd[hd[nameKeys[-1]] == i]
        featureX.append(f.iloc[:, :-1].values.astype('float32'))
        featureY.append(f.iloc[:, -1].values.astype('float32'))

    feature_trainX = []
    feature_trainY = []
    feature_testX = []
    feature_testY = []
    for i in range(len(featureX)):
        hdx_train, hdx_test, hdy_train, hdy_test = train_test_split(featureX[i], featureY[i], test_size=test_len, random_state=0)
        ### Train Data
        permutation = np.random.permutation(hdx_train.shape[0])
        hdx_len = int(len(permutation) * train_prob)
        hdx_train = hdx_train[permutation[:hdx_len], :]
        hdy_train = hdy_train[permutation[:hdx_len]]
        print(hdx_len)

        ### Test Data
        permutation = np.random.permutation(hdx_test.shape[0])
        hdx_len = int(len(permutation) * test_prob)
        hdx_test = hdx_test[permutation[:hdx_len], :]
        hdy_test = hdy_test[permutation[:hdx_len]]
        print(hdx_len)

        feature_trainX.append(hdx_train)
        feature_trainY.append(hdy_train)
        feature_testX.append(hdx_test)
        feature_testY.append(hdy_test)

        if i == 0:
            trainX = feature_trainX[i]
            trainY = feature_trainY[i]
            testX = feature_testX[i]
            testY = feature_testY[i]
        else:
            trainX = np.vstack((trainX, feature_trainX[i]))
            trainY = np.vstack((trainY.reshape([-1, 1]), feature_trainY[i].reshape([-1, 1]))).astype('int')
            testX = np.vstack((testX, feature_testX[i]))
            testY = np.vstack((testY.reshape([-1, 1]), feature_testY[i].reshape([-1, 1]))).astype('int')

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    print(feature_trainX[0].shape)
    print(feature_trainY[0].shape)
    print(feature_testX[0].shape)
    print(feature_testX[0].shape)

    return X, identify_or_embedding, orgin_dims, target_dims, feature_names, \
           trainX, trainY, testX, testY, \
           featureX, featureY, \
           feature_trainX, feature_testX, feature_trainY, feature_testY


def overSampling(trainX, trainY):
    ## oversampling
    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_sample(trainX, trainY)

    print(Counter(y_smo))
    y_smo = y_smo.reshape([-1, 1])

    return X_smo, y_smo


if __name__ == '__main__':
    deal_data(filePath='./data/mimic_cohort_modify.csv', conditionPath='./data/mimic_cohort_modify_if.csv')
