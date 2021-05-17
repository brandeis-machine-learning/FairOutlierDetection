"""
Obtain input, outlier labels, and sensitive attribute subgroups for all datasets
"""

import os
import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def mkdir(dset):
    print(f'Making {dset}')
    if not os.path.exists(dset):
        os.makedirs(dset)


def main():
    # -----cc-----#
    df_data = pd.read_excel('../datasets/cc.xls')
    data = df_data.to_numpy()
    X = data[1:, 1:24]
    categorical_data = X[:, 1:4]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)
    X = np.delete(X, 1, 1)
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[1:, 24]
    Y = []
    for i in y_temp:
        Y.append(i)
    Y = np.array(Y)
    np.save('../datasets/CC_X.npy', X_norm)
    np.save('../datasets/CC_Y.npy', Y)
    mkdir('cc')
    np.save('cc/attribute.npy', data[1:, 2])

    # -----adult-----#
    df_data = pd.read_csv('../datasets/adult.csv')
    mkdir('adult')
    np.save('adult/attribute.npy', df_data['race'].to_numpy())

    for i in range(len(df_data.columns)):
        most_frequent = df_data.iloc[:, i].value_counts()[:1].index.tolist()[0]
        for j in range(len(df_data.iloc[:, i])):
            if df_data.iloc[j, i] == '?':
                df_data.iloc[j, i] = most_frequent

    categorical_data = pd.concat(
        [df_data.iloc[:, 1], df_data.iloc[:, 3], df_data.iloc[:, 5:10], df_data.iloc[:, 13]], axis=1).to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    df_data = df_data.drop(
        ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
         'native-country'], axis=1)
    data = df_data.to_numpy()

    X = data[:, 0:6]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 6]
    Y = []
    for i in y_temp:
        if i == '<=50K':
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)

    np.save('../datasets/ADULT_X.npy', X_norm)
    np.save('../datasets/ADULT_Y.npy', Y)

    # -----kdd-----#
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz', header=None)
    df_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz', header=None)
    df_data = pd.concat([df, df_test], axis=0)
    df_data = df_data.drop(24,axis=1)
    mkdir('kdd')
    np.save('kdd/attribute.npy', df_data[10].to_numpy())

    categorical_list = [df_data.iloc[:, 1:5], df_data.iloc[:, 6:16], df_data.iloc[:, 19:29], df_data.iloc[:, 30:35], df_data.iloc[:, 36]]
    categorical_data = pd.concat(categorical_list, axis=1).to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    df_data = df_data.drop([1,2,3,4,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,25,26,27,28,29,31,32,33,34,35,37], axis=1)
    data = df_data.to_numpy()

    X = data[:, 0:10]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 10]
    Y = []
    for i in y_temp:
        if i == ' - 50000.':
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)
    np.save('../datasets/KDD_X.npy', X_norm)
    np.save('../datasets/KDD_Y.npy', Y)

    # -----student-----#
    resp = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    data = zipfile.open('student-mat.csv')
    data2 = zipfile.open('student-por.csv')
    df1 = pd.read_csv(data, sep=';')
    df2 = pd.read_csv(data2, sep=';')
    df_data = pd.concat([df1, df2], axis=0)
    mkdir('student')
    np.save('student/attribute.npy', df_data['sex'].to_numpy())

    categorical_list = [df_data.iloc[:, 0:2], df_data.iloc[:, 3:12], df_data.iloc[:, 15:23]]
    categorical_data = pd.concat(categorical_list, axis=1).to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    df_data = df_data.drop(['school','sex', 'address', 'famsize','Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
        'schoolsup', 'famsup', 'paid', 'activities','nursery','higher','internet', 'romantic'], axis=1)
    data = df_data.to_numpy()

    X = data[:, 0:14]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 13]
    Y = []
    for i in y_temp:
        if i > 7:
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)
    np.save('../datasets/STUDENT_X.npy', X_norm)
    np.save('../datasets/STUDENT_Y.npy', Y)

    # -----drug-----#
    df_data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data',
                          header=None)
    mkdir('drug')
    np.save('drug/attribute.npy', df_data[2].to_numpy())

    categorical_list = [2, 3, 4, 5, 13]
    categorical_data = df_data.iloc[:, categorical_list].to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()
    df_data = df_data.drop(
        [0, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], axis=1)
    data = df_data.to_numpy()
    X = data[:, 0:8]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 8]
    Y = []
    for i in y_temp:
        if i == 'CL0' or i == 'CL1' or i == 'CL2' or i == 'CL3':
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)

    np.save('../datasets/DRUG_X.npy', X_norm)
    np.save('../datasets/DRUG_Y.npy', Y)

    # -----german-----#
    resp = urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00573/SouthGermanCredit.zip")
    zipfile = ZipFile(BytesIO(resp.read()))
    data = zipfile.open('SouthGermanCredit.asc')
    df_data = np.loadtxt(data, skiprows=1)
    mkdir('german')
    np.save('german/attribute.npy', df_data[:, 8])

    categorical_list = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19]
    categorical_data = df_data[:, categorical_list]
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()
    data = np.delete(df_data, categorical_list, axis=1)

    X = data[:, 0:3]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 3]
    Y = []
    for i in y_temp:
        if i == 1:
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)

    np.save('../datasets/GERMAN_X.npy', X_norm)
    np.save('../datasets/GERMAN_Y.npy', Y)

    # -----asd-----#
    data = arff.loadarff('../datasets/Autism-Adult-Data.arff')
    df_data = pd.DataFrame(data[0])
    mkdir('asd')
    np.save('asd/attribute.npy', df_data['gender'].str.decode('utf-8').to_numpy())

    for i in df_data.columns:
        if i != 'age' and i != 'result':
            df_data[i] = df_data[i].str.decode('utf-8')

    for i in range(len(df_data.columns)):
        if i != 10 and i != 17:
            most_frequent = df_data.iloc[:, i].value_counts()[:1].index.tolist()[0]
            for j in range(len(df_data.iloc[:, i])):
                if df_data.iloc[j, i] == '?':
                    df_data.iloc[j, i] = most_frequent

    categorical_data = pd.concat(
        [df_data.iloc[:, 0:10], df_data.iloc[:, 11:17], df_data.iloc[:, 18:20]], axis=1)
    df_data = df_data.drop(categorical_data.columns, axis=1)
    data = df_data.to_numpy()
    categorical_data = categorical_data.to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    X = data[:, 0:2]
    for i in range(2):
        for j in range(X.shape[0]):
            if np.isnan(X[j, i]):
                temp = j
                while np.isnan(X[temp, i]):
                    temp -= 1
                X[j, i] = X[temp, i]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 2]
    Y = []
    for i in y_temp:
        if i == 'NO':
            Y.append(0)
        else:
            Y.append(1)
    Y = np.array(Y)

    np.save('../datasets/ASD_X.npy', X_norm)
    np.save('../datasets/ASD_Y.npy', Y)

    # -----obesity-----#
    df_data = pd.read_csv('../datasets/ObesityDataSet_raw_and_data_sinthetic.csv')
    mkdir('obesity')
    np.save('obesity/attribute.npy', df_data['Gender'].to_numpy())

    categorical_data = pd.concat(
        [df_data.iloc[:, 0], df_data.iloc[:, 4:6], df_data.iloc[:, 8:10], df_data.iloc[:, 11], df_data.iloc[:, 14:16]], axis=1)
    df_data = df_data.drop(categorical_data.columns, axis=1)
    data = df_data.to_numpy()
    categorical_data = categorical_data.to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(categorical_data)
    one_hot = enc.transform(categorical_data).toarray()

    X = data[:, 0:8]
    X_norm = scale(X)
    X_norm = np.append(X_norm, one_hot, axis=1)
    y_temp = data[:, 8]
    Y = []
    for i in y_temp:
        if i == 'Insufficient_Weight':
            Y.append(1)
        else:
            Y.append(0)
    Y = np.array(Y)

    np.save('../datasets/OBESITY_X.npy', X_norm)
    np.save('../datasets/OBESITY_Y.npy', Y)


if __name__ == '__main__':
    main()