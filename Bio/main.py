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


class GraphStructure():
    def __init__(self):
        self.weights = []
        self.Graphs_full = []
        self.Graph_ost_wo = []
        self.Graph_ost = []
        self.surv_time = []
        self.property_kol = 200
        self.connection_nodes = []
        self.new = []
        self.property = [0] * 72
        for i in range(72):
            self.property[i] = [0] * 1
        self.X = pd.DataFrame()
    
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
        # for i in range(len(self.Graphs_full[0])):
            # draw_graph(self.Graphs_full[0][i])

    def uniq_subgraphs(self, G):
        exsisting_subgraph = []
        graphs = []
        for n in G.nodes:
            nodes_subgraph = list(nx.dfs_edges(G, n))
            sub = nx.classes.function.edge_subgraph(G, nodes_subgraph)
            subgraph = nx.Graph(sub)
            if subgraph not in exsisting_subgraph:
                isomorph = [nx.is_isomorphic(subgraph, exs_subgr) for exs_subgr in exsisting_subgraph]
                if not True in isomorph:
                    new_Graph = nx.Graph()
                    for u, v, w in sub.edges(data=True):
                        new_Graph.add_edge(u, v, weight=w['weight'])
                    graphs.append(new_Graph)
                    exsisting_subgraph.append(subgraph)
        return graphs

    # def anslysis_graph(self, G):
    #     k = 0
    #     conn = None
    #     for n in G.nodes:
    #         k = list(nx.dfs_postorder_nodes(G, n))
    #         a = len(k)
    #         if a == 7:
    #             conn = list(nx.dfs_postorder_nodes(G, n))
    #             break
    #     return conn

    # def uniq_subgraphs(self, G):
    #     exsisting_subgraph = []
    #     new_Graph = nx.Graph()
    #     conn = self.anslysis_graph(G)
    #     if conn == None:
    #         new_Graph.add_edge(0,0, weight=0)
    #     else:
    #         for node in conn:
    #             a = list(G.edges(data=True))
    #             for n in a:
    #                 if node in n:
    #                     n = list(n)
    #                     new_Graph.add_edge(n[0], n[1], weight=n[2]['weight'])
    #     return new_Graph

    def shortest_path(self, G, source, target):
        def path_cost(G, path):
            return sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
        try:
            x = nx.shortest_simple_paths(G, source,target,weight='weight')
            par = sorted([(path_cost(G,p), p) for p in x])
        except: 
            return 0
        return par[0][0]

    def longest_path(self, G):
        def path_cost(G, path):
            x = [G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]
            return sum(x)
        try:
            par = (path_cost(G,nx.dag_longest_path(G)), nx.dag_longest_path(G))
        except:
            return 0
        return par[0]

    def calculate_prop(self, G, i):
        name = ''
        nodes = G.nodes
        for n in nodes:
            name += f'_{n}'
        count_nodes = len(G.nodes)
        columns = self.X.columns

        lable1 = f'det{name}'
        while lable1 in columns and self.X[lable1][i] != 0:
            lable1 = lable1 + '_an'

        lable2 = f'sum{name}'
        while lable2 in columns and self.X[lable2][i] != 0:
            lable2 += '_an'

        lable3 = f'short{name}'
        while lable3 in columns and self.X[lable3][i] != 0:
            lable3 += '_an'

        lable4 = f'modularity{name}'
        while lable4 in columns and self.X[lable4][i] != 0:
            lable4 += '_an'
        
        lable5 = f'cluster{name}'
        while lable5 in columns and self.X[lable5][i] != 0:
            lable5 += '_an'

        # Кластеризация
        # cluster = approximation.average_clustering(G, trials=1000, seed=10)
        # if lable5 not in self.X.columns:
        #     self.X[lable5] = np.zeros(72)
        # self.X[lable5][i] = cluster



        # Модулярность
        modularity = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
        if lable4 not in self.X.columns:
            self.X[lable4] = np.zeros(72)
        self.X[lable4][i] = modularity


        # Определитель матрицы смежности с весами
        # adj = nx.adjacency_matrix(G)
        # det_adj = LA.det(adj.todense())
        # self.property[i].append(det_adj)
        # if lable1 not in self.X.columns:
        #     self.X[lable1] = np.zeros(72)
        # self.X[lable1][i] = det_adj

        # Определитель матрицы Лапласа L=D-A
        # l = nx.laplacian_matrix(G)
        # property.append(LA.det(l.todense()))

        # Сумма всех путей в графе
        # sum = 0
        # a = list(G.edges(data=True))
        # for j in a:
        #     sum += j[2]['weight']
        # self.property[i].append(sum)

        # if lable2 not in self.X.columns:
        #     self.X[lable2] = np.zeros(72)
        # self.X[lable2][i] = sum
                


        # Длина самого короткого пути от первой до последней вершины в графе
        # dfs = list(nx.dfs_preorder_nodes(G))
        # path = 20
        # for d in dfs:
        #     short = self.shortest_path(G, dfs[0], d)
        #     if short < path and short != 0:
        #         path = short
        # if path == 20:
        #     path = 0
        # self.property[i].append(path)
        # if lable3 not in self.X.columns:
        #     self.X[lable3] = np.zeros(72)
        # self.X[lable3][i] = path

        # Longest path (critical)
        # self.property.append(self.longest_path(G))

    
    def create_x_matrix_full(self):
        X = np.zeros((len(self.Graphs_full), self.property_kol))
        for i in range(len(self.Graphs_full)):
            for k in range(len(self.Graphs_full[i])):
                self.calculate_prop(self.Graphs_full[i][k], i)
            for j in range(len(self.property[i])):
                X[i][j] = self.property[i][j]
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
    m = centr(X.values)
    sd = norm_X(X.values)
    for i in range(len(X.values)):
        for j in range(len(X.values[0])):
            X.values[i][j] = (X.values[i][j] - m[j])/ sd[j]
    return X

# def centr_norm(X):
#     m = centr(X)
#     sd = norm_X(X)
#     for i in range(len(X)):
#         for j in range(len(X[0])):
#             X[i][j] = (X[i][j] - m[j])/ sd[j]
#     return X

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
    structure.property_kol = 40
    F, Y = structure.full_graph_calc() #1
    write_x(Y, 'result_y.txt')

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