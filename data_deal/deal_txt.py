import numpy as np
import pandas as pd


def txt2csv():
    txt = np.loadtxt('data_for_dl.txt', dtype=str)
    txt = txt.T
    print(txt.shape)
    data = pd.DataFrame(txt[2:, :])
    data.columns = txt[1, :]
    # data.to_csv('file_T.csv', index=False)
    print(data.keys())

    print(data.shape)

def comm(space, data_age):
    label = []
    for j in data_age:
        cur = int(j)
        for l, s in zip(range(len(space)), space):
            if cur <= s:
                label.append(l)
                break
    return label

def csv_label_5():
    data = pd.read_csv('sample_feature_sort_1.csv')
    data2 = pd.read_csv('file_T.csv')
    data = data.join(data2)
    print(data.shape)
    data = data.drop('combine', 1)
    data = data.drop('genename', 1)
    print(data.shape)

    # # print(data['combine'].values == data2['genename'].values)
    # print()

    data_age = data['age'].values
    data_age[data_age > 94] = 94
    data['age'] = data_age

    label1 = comm([20, 40, 60, 80, 94], data_age)
    label2 = comm([1, 21, 41, 61, 81, 94], data_age)
    label3 = comm([2, 22, 42, 62, 82, 94], data_age)
    label14 = comm([13, 33, 53, 73, 93, 94], data_age)
    label20 = comm([19, 39, 59, 79, 94], data_age)

    print(label1)
    print(len(label1))
    print(data.loc[[0]].values.shape)
    data['label'] = label1
    data.to_csv('feature_1.csv', index=False)

    data['label'] = label2
    data.to_csv('feature_2.csv', index=False)

    data['label'] = label3
    data.to_csv('feature_3.csv', index=False)

    data['label'] = label14
    data.to_csv('feature_14.csv', index=False)

    data['label'] = label20
    data.to_csv('feature_20.csv', index=False)





def del_csv():
    data = pd.read_csv('../data/feature_1.csv')
    data = data.iloc[[0]]
    print(data.shape)
    data.iloc[:,:] = 1
    data.iloc[0, 0] = 0
    data.iloc[0, 1] = 0
    data.iloc[0, 2] = 0
    data.iloc[0, 3] = 0
    data.iloc[0, 4] = 0
    data.iloc[0, 5] = 0
    data.iloc[0, 6] = 0
    data.iloc[0, 7] = 0

    data.to_csv("feature_if.csv", index=False)


def create_csv():
    data = pd.read_csv('../data/feature_1.csv')
    temp = data.iloc[:, :20]
    temp = temp.join(data.iloc[:, -1])
    print(temp.shape)
    temp.to_csv("feature_1_1.csv", index=False)

    data_if = pd.read_csv('../data/feature_if.csv')
    temp_if = data_if.iloc[:, :20]
    temp_if = temp_if.join(data_if.iloc[:, -1])
    print(temp_if.shape)
    temp_if.to_csv("feature_if_if.csv", index=False)

def deal_equal():
    data = pd.read_csv('../data/feature_1.csv')
    data_if = pd.read_csv('../data/feature_if.csv')

txt2csv()