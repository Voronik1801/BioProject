import csv
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

components = [5]


def random_value():
    value = random.randint(0, 71)
    return value

def main_pls():
    # Open csv and save
    with open('BioProject/Bio/pls_value.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        dataForAnalys = list(csv_reader)

    # Defined X and Y for next PLS methos
    n = len(dataForAnalys) - 1
    p = len(dataForAnalys[0]) - 1

    X = np.zeros((n, p))
    Y = np.zeros(n)
    utils = ls_ut(X, Y)

    # Saving data for analysis in main structure for pls
    utils.ImportToX(dataForAnalys)
    utils.ImportToY(dataForAnalys)
    err = utils.ErrorLib(X, Y)
    
    # print(LA.det(np.dot(X.T, X)))
    # for i in err:
        # print(err[i])
    # components = [5]
    # for i in components:
        # regress = PLS1Regression(X, Y, i, 'classic')
        # y_oz = regress.Predict(X)
    #     R = r2_score(Y, y_oz)
    #     print(R)


def draw_graph(G):
    options = {
        'node_color': 'blue',
        'node_size': 15,
        'width': 0.5,
    }
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

        self.donor_ost = [d.split('@')[0] for d in self.donor]
        self.akceptor_ost = [a.split('@')[0] for a in self.akceptor]

        self.donor_ost_wo = [d.split('.')[0] for d in self.donor]
        self.akceptor_ost_wo = [a.split('.')[0] for a in self.akceptor]
        
    def load_values_in_graph(self, donor, akceptor, weights):
        Graph = nx.Graph()
        for i in range (len(donor)):
            if(weights[i] != 0.0):
                Graph.add_edge(donor[i], akceptor[i], weight=weights[i])
        return Graph

    def creat_full_value_graph(self):
        for i in range(len(self.weights)):
            self.Graphs_full.append(self.load_values_in_graph(self.donor, self.akceptor, self.weights[i]))

    def create_ost_wo_graph(self):
        for i in range(len(self.weights)):
            self.Graph_ost_wo.append(self.load_values_in_graph(self.donor_ost_wo, self.akceptor_ost_wo, self.weights[i]))

    def creat_ost_value_graph(self):
        for i in range(len(self.weights)):
            self.Graph_ost.append(self.load_values_in_graph(self.donor_ost, self.akceptor_ost, self.weights[i]))

    def calculate_prop(self, G):
        property = []   
        X = []
        # last_enter = []
        v = nx.adjacency_matrix(G)
        # res = np.array([list(item.values()) for item in v.values()])
        res = v.data
        f = open('result_graph_X.txt', 'w')
        for i in range(len(res)):
            for j in range(len(res[0])):
                f.write(str(res[i][j]) + '\t')
            f.write('\n')
        verh_treug = np.triu(res)
        for i in range(len(verh_treug)):
            for j in range(len(verh_treug[0])):
                if verh_treug[i][j]!=0:
                    property.append(verh_treug[i][j])
        # for n in G.nodes:
        #     # property.append(nx.degree(G, n))
        #     for k in v[n]:
        #         el = v[n][k]
        #         # if el != 0 and el not in last_enter:
        #         if el != 0:
        #             # last_enter.append(el)
        #             property.append(el)
        return property

    def create_x_matrix_ost_wo(self):
        X = np.zeros((len(self.weights), self.property_kol))
        for i in range(len(self.weights)):
            prop = self.calculate_prop(self.Graph_ost_wo[i])
            for j in range(len(prop)):
                X[i][j] = prop[j]
        return X
    
    def create_x_matrix_full(self):
        X = np.zeros((len(self.weights), self.property_kol))
        for i in range(len(self.weights)):
            prop = self.calculate_prop(self.Graphs_full[i])
            for j in range(len(prop)):
                X[i][j] = prop[j]
        return X

    def create_x_matrix_ost(self):
        X = np.zeros((len(self.weights), self.property_kol))
        for i in range(len(self.weights)):
            prop = self.calculate_prop(self.Graph_ost[i])
            for j in range(len(prop)):
                X[i][j] = prop[j]
        return X

    def full_graph_calc(self):
        self.creat_full_value_graph()
        X = np.array(self.create_x_matrix_full())
        Y = np.array(self.surv_time)
        return X, Y

    def ost_without_sub_graph_calc(self):
        self.create_ost_wo_graph()
        X = np.array(self.create_x_matrix_ost_wo())
        Y = np.array(self.surv_time)
        return X, Y
    
    def ost_graph_calc(self):
        self.creat_ost_value_graph()
        X = np.array(self.create_x_matrix_ost())
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
    for i in range(len(X)):
        for j in range(len(X[0])):
            f.write(str(X[i][j]) + '\t')
        f.write('\n')



def calc_pvalue_for_coef(X, Y, k):
    y_oz, R, ts_b = pls_prediction(X, Y, k)
    t_stat = [stats.t.cdf(np.abs(i),(len(X)-1)) for i in ts_b]
    p_values = [2*(1-i) for i in t_stat]
    f = open('p_value.txt', 'w')
    for i in range(len(p_values)):
        f.write(str(p_values[i]) + '\n')
    f.close()
    f = open('betta.txt', 'w')
    for i in range(len(ts_b)):
        f.write(str(ts_b[i]) + '\n')
    f.close()

    f = open('t_stat.txt', 'w')
    for i in range(len(t_stat)):
        f.write(str(t_stat[i]) + '\n')
    f.close()

def calc_stat_for_model(X, Y, structure):
    F = []
    S = []
    P = []
    for k in components:
        y_oz, R = pls_prediction(X, Y, k)
        n = len(Y)
        m = structure.property_kol
        F.append((R / (1 - R)) * ((n - m - 1)/m))
        stat, p = ttest_rel(Y, y_oz)
        S.append(stat)
        P.append(p)
    print(F)
    print(S)
    print(P)

def norm_X(X):
    for i in range(len(X[0])):
        arr = X[:,i]
        d = np.var(arr)
        a = np.var(X, axis = 0)
        for j in range(len(arr)):
            arr[j] = arr[j] / np.sqrt(d)
        X[:, i] = arr
    return X


def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('BioProject/Bio/graph_value.csv')
   
    # structure.property_kol = 1068
    # X, Y = structure.ost_graph_calc() #2
    # structure.property_kol = 698
    # X, Y = structure.ost_without_sub_graph_calc() #3
    
    structure.property_kol = 60000
    X, Y = structure.full_graph_calc() #1
    ######### удаляем столбцы где все элементы одинаковые
    X = np.round(X, 7)
    b = X == X[0,:]
    c = b.all(axis=0)
    X = X[:,~c]
    X = np.unique(X, axis=1)
    X = norm_X(X)
    X -= np.amin(X, axis=(0, 1))
    X /= np.amax(X, axis=(0, 1))
    ######### удаляем все столбцы где много 0
    # k = len(X[0])
    # i = 0
    # while True:
    #     if(X[0][i] == 0):
    #         X = np.delete(X, i, axis=1)
    #         k = len(X[0])
    #     else:
    #         i += 1
    #     if i + 1 == k:
    #         break
    ######### удаляем линейно зависимые строки
    # q,r = np.linalg.qr(X.T)
    # a = np.abs(np.diag(r))>=1e-10
    # X = X[a]
    # Y =Y[a]
    #####
    # for i in range(len(X)):
    #     for j in range(len(X[0])):
    #         if X[i][j] == 0.0:
    #             X[i][j] = 0.0001
    # write_x(np.dot(X.T, X))
    write_x(X)
    ut = ls_ut(X, Y)
    components = [5, 7, 10, 12]
    print(LA.det(np.dot(X.T, X)))
    # print(LA.eig(np.dot(X.T, X)))
    for k in components:
        y_oz, R = pls_prediction(X, Y, k)
        print(R)
        print('---')
    # ec = ut.ErrorPLS1Classic(X, Y)
    # for k in ec:
    #     print(ec[k])
    # print('---')
    # ecr = ut.ErrorPLS1Robust(X, eY)
    # for k in ecr:
    #     print(ecr[k])
    # print('---')
    # ecv = ut.ErrorCVClassic(X, Y)
    # for k in ecv:
    #     print(ecv[k])
    # print('---')
    # ecvr = ut.ErrorCVRobust(X, Y)
    # for k in ecvr:
    #     print(ecvr[k])

    # calc_pvalue_for_coef(X, Y, 10)

    # ut.CreateTwoPlot(Y, y_oz)
main_graph()
# main_pls()

# G1 = nx.Graph()
# G1.add_edge(0, 1, weight=0.1)
# G1.add_edge(0, 2, weight=1.1)
# G1.add_edge(1, 4, weight=0.4)
# G1.add_edge(4, 5, weight=0.3)

# glist = [ nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)]), G1]

# X = nx.communicability_exp(glist[0])
# res = np.array([list(item.values()) for item in X.values()])
# def prop(G):
#     X = nx.communicability_exp(G)
#     res = np.array([list(item.values()) for item in X.values()])
#     property = []
#     for i in range(len(res)):
#         for j in range(len(res[0])):
#             if res[i][j]!=0:
#                 property.append(res[i][j])
#     return property

# X = np.zeros((2, 49))
# for i in range(len(glist)):
#     property = prop(glist[i])
#     for j in range(len(property)):
#         X[i][j] = property[j]

# X = np.round(X, 5)
# b = X == X[0,:]
# c = b.all(axis=0)
# X = X[:,~c]
# X = np.unique(X, axis=1)
# q,r = np.linalg.qr(X.T)
# a = np.abs(np.diag(r))>=1e-10
# X = X[a]

# X = nx.communicability(glist[1])
# X = np.array([list(item.values()) for item in X.values()])
# X = np.triu(X)
# X = np.round(X, 3)
# f = open('result_graph_X.txt', 'w')
# X = np.dot(X.T, X)
# for i in range(len(X)):
#     for j in range(len(X[0])):
#         f.write(str(X[i][j]) + '\t')
#     f.write('\n')
# print(LA.det(np.dot(X.T, X)))
# print(LA.det(np.dot(res.T, res)))

# X = nx.communicability_exp(glist[0])
# res = np.array([list(item.values()) for item in X.values()])
# res = np.triu(res)
# property = []
# for i in range(len(res)):
#     for j in range(len(res[0])):
#         if res[i][j]!=0:
#             property.append(res[i][j])
# print(np.triu(res))