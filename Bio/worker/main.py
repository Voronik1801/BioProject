import csv
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

components = [5]
df = pd.DataFrame()
res_df = pd.DataFrame()

def random_value():
    value = random.randint(0, 71)
    return value


def draw_graph(G):
    options = {
        'node_color': 'blue',
        'node_size': 15,
        'width': 0.5,
    }
    # nx.draw(G, cmap = plt.get_cmap('jet'),node_color='red',with_lables=True) 
    nx.draw(G, **options) 
    plt.show()

def calulate_property(G):
    # print(nx.transitivity(G))
    print(nx.density(G))
    print(G.edges())    



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
        else:
            print(LA.det(np.dot(X.values.T, X.values)))
            for column in columns:
                print(column)
            write_x(X.values, "result_for_pls_mod.txt")
            break

def ols_prediction(X,Y):
    est = sm.OLS(Y, X.values).fit()
    y_oz = est.predict(X.values)
    # est = sm.OLS(Y, X).fit()
    # y_oz = est.predict(X)
    print(est.summary())
    analysis_pVal(est, X, Y)
    return y_oz

def pls_prediction(X, Y, comp, method='classic'):
    regress = PLS1Regression(X, Y, comp, method)
    y_oz = regress.Predict(X)
    R = r2_score(Y, y_oz)
    return y_oz, R

def write_x(X, file='result_graph_X.txt'):
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

def cross_validation(X, Y):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        est = sm.OLS(Y, X).fit()
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
        regress = PLS1Regression(X, Y, 10, 'classic')
        predictYpred = regress.Predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

def error(y, y_oz):
    dif = (y - y_oz) ** 2
    scal = np.sum(dif)
    err = np.sqrt(scal) / 72
    return err

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('D:\Diplom\BioProject\Bio\graph_value.csv')
    write_x(structure.surv_time, 'result_y.txt')


    X = structure.X
    X = X.dropna(axis=1,how='all')
    # X = X.loc[:, (X != 0).any(axis=0)]
    # X = X.T.drop_duplicates().T

    write_x(X.values, 'sum')
    c = X.columns
    for column in c:
        print(column)

    # # with open('input') as f:
    #     # X = np.array([list(map(float, row.split())) for row in f.readlines()])
    # # X = uniq(X)
    # write_x(X.values, 'pre_x.txt')
    # # write_x(structure.X.values, file='pre_x.txt')
    print(LA.det(np.dot(X.values.T, X.values)))
    X = centr_norm(X)
    # ones = np.ones(len(structure.Graphs_full))
    # X['const'] = ones
    # # # X = np.hstack((X.values, np.atleast_2d(ones).T))
    # # write_x(X.values)

    # # # X = np.linalg.qr(X)[0]
    print(LA.det(np.dot(X.values.T, X.values)))
    # print(LA.det(np.dot(L.T, L)))
    ols_prediction(X, Y)


    # components = [4, 8, 20]


    # # # print(LA.eig(np.dot(X.T, X)))
    # for k in components:
        # y_oz, R = pls_prediction(X, Y, k)
    #     print(R)
    #     print('---')
    # utils = ls_ut(X, Y)
    # # cv = cross_validation(X, Y)
    # cv = cross_validation_pls(X, Y)
    # utils.CreateTwoPlot(Y, cv)
    # print(error(Y, cv))
 
main_graph()