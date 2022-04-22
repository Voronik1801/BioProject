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
import model

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('D:\Diplom\BioProject\Bio\graph_value.csv')
    model.write_y(structure.surv_time, 'result_y.txt')

    X = structure.X
    X = X.loc[:, (X != 0).any(axis=0)]

    c = X.columns
    for column in c:
        print(column)
    model.write_x(X.values, X.columns, 'estr.csv')

    print(f"det: {LA.det(np.dot(X.values.T, X.values))}")
    X = model.centr_norm(X)
    ones = np.ones(len(structure.Graphs))
    X["const"] = ones
    print(f"det: {LA.det(np.dot(X.values.T, X.values))}")

    model.ols_prediction(X, structure.surv_time)


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
 
# main_graph()

def main_input():
    df = pd.DataFrame()
    df_mod = pd.read_csv('data_for_ols/input_mod.csv', sep='\t')
    df_short = pd.read_csv('data_for_ols/input_short.csv', sep='\t')
    df_sum = pd.read_csv('data_for_ols/input_sum.csv', sep='\t')

    with open('data_for_ols/result_y.txt') as f:
        y = np.array([list(map(float, row.split())) for row in f.readlines()])
    
    # берем за основу т.к R = 0.502
    df = df_short
    # df = df.drop('short_124.B@OD1_46.B@NE2_71.B@NE2_124.B@OD2_126.B@N', 1)
    df['sum_79.A@NE_80.A@O_83.A@N'] = df_sum['sum_79.A@NE_80.A@O_83.A@N'] # коэф R = 0.515
    # при добавлении этого параметра определитель 0
    # df['sum_126.B@N_124.B@OD1_71.B@NE2_124.B@OD2_46.B@NH2'] = df_sum['sum_126.B@N_124.B@OD1_71.B@NE2_124.B@OD2_46.B@NH2'] 
    # данный параметр незначим в общей модели
    df['sum_125.B@OD2_71.B@NE2_124.B@OD1_126.B@N_124.B@OD2_46.B@NE2'] = df_sum['sum_125.B@OD2_71.B@NE2_124.B@OD1_126.B@N_124.B@OD2_46.B@NE2']
    
    # данный параметр незначим в общей модели
    df['modularity_52.A@OD2_54.A@OG1_54.A@N_52.A@OD1'] = df_mod['modularity_52.A@OD2_54.A@OG1_54.A@N_52.A@OD1']
    # данный параметр незначим в общей модели
    df['modularity_52.B@OD2_54.B@OG1_54.B@N_52.B@OD1'] = df_mod['modularity_52.B@OD2_54.B@OG1_54.B@N_52.B@OD1']
     # при добавлении этого параметра определитель 0
    # df['modularity_124.B@OD1_71.B@NE2_126.B@N_124.B@OD2_46.B@NH2'] = df_mod['modularity_124.B@OD1_71.B@NE2_126.B@N_124.B@OD2_46.B@NH2']
     # данный параметр незначим в общей модели
    df['modularity_124.B@OD1_71.B@NE2_126.B@N_124.B@OD2_125.B@OD2_46.B@NE2'] = df_mod['modularity_124.B@OD1_71.B@NE2_126.B@N_124.B@OD2_125.B@OD2_46.B@NE2']
     # данный параметр незначим в общей модели
    df['modularity_124.A@OD1_126.A@N_46.A@NE2_124.A@OD2_71.A@NE2'] = df_mod['modularity_124.A@OD1_126.A@N_46.A@NE2_124.A@OD2_71.A@NE2']


    # model.rlm_prediction(df, y)
    # cv = model.cross_validation_rlm(df.values, y)
    model.ols_prediction(df, y)
    cv = model.cross_validation_ols(df.values, y)
    utils = ls_ut(df.values, y)
    utils.CreateTwoPlot(data1=y, data2=cv)
    print(model.error(y, cv))

    # from sklearn.linear_model import ElasticNetCV
    # m = ElasticNetCV()
    # m.fit(df.values, y)
    # y_pred = m.predict(df.values)
    # # utils.CreateTwoPlot(data1=y, data2=y_pred)
    # print(model.error(y, y_pred))
    # print(m.score(df.values, y))

    # from sklearn.linear_model import RidgeCV
    # m = RidgeCV()
    # m.fit(df.values, y)
    # y_pred = m.predict(df.values)
    # # utils.CreateTwoPlot(data1=y, data2=y_pred)
    # print(model.error(y, y_pred))
    # print(m.score(df.values, y))

    # from sklearn.linear_model import LassoCV
    # m = LassoCV()
    # m.fit(df.values, y)
    # y_pred = m.predict(df.values)
    # # utils.CreateTwoPlot(data1=y, data2=y_pred)
    # print(model.error(y, y_pred))
    # print(m.score(df.values, y))

    # cv = model.cross_validation_forest(df.values, y)
    # from sklearn.ensemble import RandomForestRegressor
    # m = RandomForestRegressor()
    # m.fit(df.values, y)
    # y_pred = m.predict(df.values)
    # # utils.CreateTwoPlot(data1=y, data2=y_pred)
    # utils.CreateTwoPlot(data1=y, data2=cv)
    # # print(model.error(y, y_pred))
    # print(model.error(y, cv))
    # # print(m.score(df.values, y))

    
# main_input()

def final_model():
    df = pd.DataFrame()
    df = pd.read_csv('final_ols.csv', sep='\t')
    # df = pd.read_csv('final_ols_515.csv', sep='\t')

    with open('data_for_ols/result_y.txt') as f:
        y = np.array([list(map(float, row.split())) for row in f.readlines()])

    # model.rlm_prediction(df, y)
    # cv = model.cross_validation_rlm(df.values, y)

    model.ols_prediction(df, y)
    cv = model.cross_validation_ols(df.values, y)
    utils = ls_ut(df.values, y)
    utils.CreateTwoPlot(data1=y, data2=cv)
    print(model.error(y, cv))

final_model()