import csv
from ctypes import util
import re
from networkx.algorithms.isomorphism import isomorph
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import degree
from networkx.generators.trees import prefix_tree
from networkx.readwrite.edgelist import write_edgelist
import numpy as np
from methods.utils import Utils as ls_ut
import random
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from methods.PLS1 import PLS1Regression
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel
import numpy.linalg as LA
from itertools import groupby
from networkx.algorithms import approximation as approx

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
    # nx.draw(G, cmap = plt.get_cmap('jet'),node_color='red',with_labels=True) 
    nx.draw(G, **options) 
    plt.show()

def calulate_property(G):
    # print(nx.transitivity(G))
    print(nx.density(G))
    print(G.edges())    


class GraphStructure():
    def __init__(self):
        self.weights = []
        self.Graphs_full = []
        self.Graph_ost_wo = []
        self.Graph_ost = []
        self.surv_time = []
        self.property_kol = 0
        self.connection_nodes = []
        self.new = []
    
    def calculate_main_values(self, path):
        with open(path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            dataForAnalys = pd.DataFrame(csv_reader)

        self.stability_hidr_bond = dataForAnalys.loc[:0, 1:] # name of column is stability of the hydrogen bond
        donor_akceptor = dataForAnalys.loc[1:182, :0]
        self.surv_time = list(map(float,dataForAnalys.loc[183:, 1:].values[0]))
        for i in range(1, 73):
            self.weights.append(list(float(w) for w in dataForAnalys.loc[1:182, i:i].values))

        self.donor = [donor[0].split('-')[0] for donor in donor_akceptor.values]
        self.akceptor = [akceptor[0].split('-')[1] for akceptor in donor_akceptor.values]
        
    def load_values_in_graph(self, donor, akceptor, weights):
        Graph = nx.Graph()
        for i in range (len(donor)):
            if(weights[i] != 0.0):
                Graph.add_edge(donor[i], akceptor[i], weight=weights[i])
        return Graph

    def creat_full_value_graph(self):
        n = len(self.weights)
        for i in range(n):
            self.Graphs_full.append(self.load_values_in_graph(self.donor, self.akceptor, self.weights[i]))
        for i in range(n):
            self.Graphs_full[i] = self.uniq_subgraphs(self.Graphs_full[i])
        draw_graph(self.Graphs_full[0])

    # def uniq_subgraphs(self, G):
    #     exsisting_subgraph = []
    #     new_Graph = nx.Graph()
    #     for n in G.nodes:
    #         nodes_subgraph = list(nx.dfs_edges(G, n))
    #         sub = nx.classes.function.edge_subgraph(G, nodes_subgraph)
    #         subgraph = nx.Graph(sub)
    #         if subgraph not in exsisting_subgraph:
    #             isomorph = [nx.is_isomorphic(subgraph, exs_subgr) for exs_subgr in exsisting_subgraph]
    #             if not True in isomorph and len(nodes_subgraph) == 3:
    #                 for u, v, w in sub.edges(data=True):
    #                     new_Graph.add_edge(u, v, weght=w['weight'])
    #                     break
    #                 exsisting_subgraph.append(subgraph)
    #     return new_Graph
    def anslysis_graph(self, G):
        k = 0
        for n in G.nodes:
            a = len(list(nx.dfs_postorder_nodes(G, n)))
            if a == 3:
                conn = list(nx.dfs_postorder_nodes(G, n))
                break
        return conn

    def uniq_subgraphs(self, G):
        exsisting_subgraph = []
        new_Graph = nx.Graph()
        conn = self.anslysis_graph(G)
        for node in conn:
            a = list(G.edges(data=True))
            for n in a:
                if node in n:
                    n = list(n)
                    new_Graph.add_edge(n[0], n[1], weight=n[2]['weight'])
        return new_Graph

    def calculate_prop(self, G):
        property = []
        det_adj = LA.det(nx.adj_matrix(G))
        return property

    
    def create_x_matrix_full(self):
        X = np.zeros((len(self.Graphs_full), self.property_kol))
        for i in range(len(self.Graphs_full)):
            prop = self.calculate_prop(self.Graphs_full[i])
            for j in range(len(prop)):
                X[i][j] = prop[j]
        return X


    def full_graph_calc(self):
        self.creat_full_value_graph()
        X = np.array(self.create_x_matrix_full())
        Y = np.array(self.surv_time)
        return X, Y

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

def ols_prediction(X,Y):
    est = sm.OLS(Y, X).fit()
    y_oz = est.predict(X)
    print(est.summary())
    return y_oz

def pls_prediction(X, Y, comp, method='classic'):
    regress = PLS1Regression(X, Y, comp, method)
    y_oz = regress.Predict(X)
    R = r2_score(Y, y_oz)
    return y_oz, R

def write_x(X):
    f = open('result_graph_X.txt', 'w')
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         f.write(str(X[i][j]) + '\t')
    for row in X:
        row = np.asarray(row).reshape(-1)
        f.write('\t'.join([str(a) for a in row]) + '\n')

# исправить расчет нормирования и центрирования

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
    m = centr(X)
    sd = norm_X(X)
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = (X[i][j] - m[j])/ sd[j]
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

def error(y, y_oz):
    dif = (y - y_oz) ** 2
    scal = np.sum(dif)
    err = np.sqrt(scal) / 72
    return err

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('D:\Diplom\BioProject\Bio\graph_value.csv')
    structure.property_kol = 200
    X, Y = structure.full_graph_calc() #1
    X = uniq(X)
    write_x(X)
    ones = np.ones(len(structure.Graphs_full))
    X = centr_norm(X)
    X = np.hstack((X, np.atleast_2d(ones).T))
    write_x(X)
    # X = np.linalg.qr(X)[0]
    print(LA.det(np.dot(X.T, X)))
    # ols_prediction(X, Y)
    # components = [2, 3, 4]
    # # print(LA.eig(np.dot(X.T, X)))
    # for k in components:
    #     y_oz, R = pls_prediction(X, Y, k)
    #     print(R)
    #     print('---')
    # utils = ls_ut(X, Y)
    # cv = cross_validation(X, Y)
    # utils.CreateTwoPlot(Y, cv)
    # print(error(Y, cv))

main_graph()
