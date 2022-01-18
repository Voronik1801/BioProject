import csv
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
        n = len(self.weights)
        for i in range(n):
            self.Graphs_full.append(self.load_values_in_graph(self.donor, self.akceptor, self.weights[i]))
        for i in range(n):
            self.Graphs_full[i] = self.uniq_subgraphs(self.Graphs_full[i])
        # draw_graph(self.Graphs_full[5])
        # draw_graph(self.Graphs_full[0])

    def filter(self):
        i = 0
        n = len(self.weights)
        while i < n:
            if self.connection_nodes[i] == None:
                self.connection_nodes.remove(self.connection_nodes[i])
                self.Graphs_full.remove(self.Graphs_full[i])
                self.surv_time.remove(self.surv_time[i])
                n -= 1
                continue
            i += 1                  
    
    def uniq_subgraphs(self, G):
        exsisting_subgraph = []
        new_Graph = nx.Graph()
        for n in G.nodes:
            nodes_subgraph = list(nx.dfs_edges(G, n))
            sub = nx.classes.function.edge_subgraph(G, nodes_subgraph)
            subgraph = nx.Graph(sub)
            if subgraph not in exsisting_subgraph:
                isomorph = [nx.is_isomorphic(subgraph, exs_subgr) for exs_subgr in exsisting_subgraph]
                if not True in isomorph and len(nodes_subgraph) != 4:
                    for u, v, w in sub.edges(data=True):
                        new_Graph.add_edge(u, v, weght=w['weight'])
                    exsisting_subgraph.append(subgraph)
        return new_Graph

    def calculate_prop(self, G):
        property = []
        if G == None: 
            return [0, 0 , 0] 
        nodes = G.nodes
        degree_sequence = [d for n, d in G.degree()]
        property=degree_sequence
        for n in nodes:
            # property.append(len(list(nx.dfs_postorder_nodes(G, n))))
            cycles = nx.cycle_basis(G, n)
            property.append(len(cycles))
            cycle = sorted([len(c) for c in nx.cycle_basis(G, n)])
            if cycle != []:
                property.append(max(cycle))
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


def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('BioProject/Bio/graph_value.csv')
    structure.property_kol = 200
    X, Y = structure.full_graph_calc() #1
    ones = np.ones(len(structure.Graphs_full))
    X = uniq(X)
    X = centr_norm(X)
    X = np.hstack((X, np.atleast_2d(ones).T))
    # write_x(X)
    print(LA.det(np.dot(X.T, X)))
    ols_prediction(X, Y)
    components = [10, 15, 24]
    # print(LA.eig(np.dot(X.T, X)))
    for k in components:
        y_oz, R = pls_prediction(X, Y, k)
        print(R)
        print('---')
main_graph()
