import csv
from unittest import result
import numpy as np
import random
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from methods.PLS1 import PLS1Regression
import numpy.linalg as LA
import networkx.algorithms.community as nx_comm
from networkx.algorithms import approximation
from methods.utils import Utils as ls_ut
import scipy
import scipy.linalg
from graph import GraphStructure



def random_value():
    value = random.randint(0, 71)
    return value





def error(Y ,y_oz):
    dif = 0
    for i in range(len(y_oz)):
        dif += (y_oz[i] - Y[i]) ** 2
    err = np.sqrt(dif) / 72
    return err

def pls_prediction_lib(X, Y, comp):
    regression = PLSRegression(n_components=comp)  # defined pls, default stand nipals
    regression.fit(X, Y)  # Fit model to data.
    y_oz = regression.predict(X)
    R = regression.score(X, Y)
    return y_oz, R

def analysis_pVal(est, X, Y):
    sigLevel = 0.05
    max = 0
    pVals = est.pvalues
    delete_index = 0
    delete_column = 0
    columns = X.columns
    while True:   
        for i in range(len(pVals)):
            if pVals[i] > max:
                max = pVals[i]
                delete_index = i
                delete_column = columns[i]
        if pVals[delete_index] > sigLevel:
            # print(pVals[delete_index])
            print(delete_column)
            X = X.drop(delete_column, axis=1)
            est = sm.OLS(Y, X.values).fit()
            y_oz = est.predict(X.values)
            print(est.summary())
            pVals = est.pvalues
            columns = X.columns
            max = 0
            for column in columns:
                print(column)
        else:
            print(f"det: {LA.det(np.dot(X.values.T, X.values))}")
            # write_x(X.values, columns, "final_ols.csv")
            break

def ols_prediction(X,Y):
    est = sm.OLS(Y, X.values).fit()
    y_oz = est.predict(X.values)
    print(est.summary())
    analysis_pVal(est, X, Y)
    return y_oz

def rlm_prediction(X,Y):
    est = sm.RLM(Y, X.values, M=sm.robust.norms.HuberT()).fit()
    y_oz = est.predict(X.values)
    print(est.summary())
    return y_oz

def pls_prediction(X, Y, comp, method='classic'):
    regress = PLS1Regression(X, Y, comp, method)
    y_oz = regress.Predict(X)
    R = r2_score(Y, y_oz)
    return y_oz, R

def write_x(X, column, file='result_graph_X.csv'):
    f = open(file, 'w')
    f.write('\t'.join([str(a) for a in column]) + '\n')
    for row in X:
        row = np.asarray(row).reshape(-1)
        f.write('\t'.join([str(a) for a in row]) + '\n')

def write_y(X, file='result_y.csv'):
    f = open(file, 'w')
    for row in X:
        row = np.asarray(row).reshape(-1)
        f.write('\t'.join([str(a) for a in row]) + '\n')

def norm_X(X):
    sd = []
    for i in range(len(X[0])):
        arr = X[:,i]
        d = np.var(arr)
        sd.append(np.sqrt(d))
    return sd

def centr(X):
    m = []
    for i in range(len(X[0])):
        arr = X[:,i]
        sum = arr.sum()
        m.append(arr.sum() / len(arr))
    return m

def centr_norm(X):
    m = centr(X.values)
    sd = norm_X(X.values)
    for i in range(len(X.values)):
        for j in range(len(X.values[0])):
            X.values[i][j] = (X.values[i][j] - m[j])/ sd[j]
    return X

def uniq(X):
    X = np.unique(X, axis=1)
    b = X == X[0,:]
    c = b.all(axis=0)
    X = X[:,~c]
    write_x(X)
    return X

def uniq(X, pred):
    X = np.unique(X, axis=1)
    b = X == X[0,:]
    c = b.all(axis=0)
    X = X[:,~c]
    pred = pred[:, ~c]
    return X, pred

def cross_validation_ols(X, Y):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        # beginX, predictX = uniq(beginX, predictX)
        beginY = np.delete(beginY, [i], 0)
        est = sm.OLS(beginY, beginX).fit()
        predictYpred = est.predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

def cross_validation_ols_n(X, Y, n=9):
    resultCV = np.zeros(X.shape[0])
    i = 0
    k = 0
    j = 0
    while True:
        if i >= X.shape[0]:
            break
        i += n
        X_test = X[k:i]
        Y_test = Y[k:i]
        X_train = X
        Y_train = Y
        for _ in range(n-1):
            X_train = np.delete(X_train, [j], 0)
            Y_train = np.delete(Y_train, [j], 0)
        est = sm.OLS(Y_train, X_train).fit()
        predictYpred = est.predict(X_test)
        f = 0
        while j < i:
            resultCV[j] = predictYpred[f]
            j += 1
            k += 1
            f += 1
        if resultCV[71] != 0:
            break
        
    return resultCV

def cross_validation_rlm_n(X, Y, n=9):
    resultCV = np.zeros(X.shape[0])
    i = 0
    k = 0
    j = 0
    while True:
        if i >= X.shape[0]:
            break
        i += n
        X_test = X[k:i]
        Y_test = Y[k:i]
        X_train = X
        Y_train = Y
        for _ in range(n-1):
            X_train = np.delete(X_train, [j], 0)
            Y_train = np.delete(Y_train, [j], 0)
        est = sm.RLM(Y_train, X_train, M=sm.robust.norms.HuberT()).fit()
        predictYpred = est.predict(X_test)
        f = 0
        while j < i:
            resultCV[j] = predictYpred[f]
            j += 1
            k += 1
            f += 1
        if resultCV[71] != 0:
            break
        
    return resultCV


def cross_validation_rlm(X, Y):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        # beginX, predictX = uniq(beginX, predictX)
        est = sm.RLM(beginY, beginX, M=sm.robust.norms.HuberT()).fit()
        predictYpred = est.predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

def cross_validation_pls(X, Y):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        regress = PLS1Regression(beginX, beginY, 10, 'classic')
        predictYpred = regress.Predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

from sklearn.ensemble import RandomForestRegressor

def cross_validation_forest(X, Y):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        m = RandomForestRegressor()
        m.fit(beginX, beginY)
        predictYpred = m.predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

def error(y, y_oz):
    y = np.array(y)
    y_oz = np.array(y_oz)
    dif = (y - y_oz) ** 2
    scal = np.sum(dif)
    err = np.sqrt(scal) / 72
    return err
