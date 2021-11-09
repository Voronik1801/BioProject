import csv
from networkx.algorithms.shortest_paths import weighted
from networkx.classes.function import degree
from networkx.generators.trees import prefix_tree
import numpy as np
from methods.utils import Utils as ls_ut
import random
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression
from methods.PLS1 import PLS1Regression
from sklearn.metrics import r2_score


components = [4, 6, 7, 10]

def random_value():
    value = random.randint(0, 71)
    return value

def main_pls():
    # Open csv and save
    with open('pls_value.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        dataForAnalys = list(csv_reader)

    # Defined X and Y for next PLS methos
    n = len(dataForAnalys) - 1
    p = len(dataForAnalys[0]) - 1

    X = np.zeros((n, p))
    Y = np.zeros(n)
    utils = Utils(X, Y)

    # Saving data for analysis in main structure for pls
    utils.ImportToX(dataForAnalys)
    utils.ImportToY(dataForAnalys)
    err = {}
    for i in components:
        err[i] = 0

    for i in range(100):
        print(i)
        checkY = copy.copy(Y)
        val = random_value()
        checkY[val] = 100
        ret = utils.ErrorPLS1Robust(X, Y)
        for k in components:
            err[k] += ret[k]

    for i in components:
        print (err[i]/100)

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
        self.Graph_atom = []
        self.Graph_ost = []
        self.Graph_ost_atom = []
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

        self.donor_atom = [d.split('.')[1] for d in self.donor]
        self.akceptor_atom = [a.split('.')[1] for a in self.akceptor]

        self.donor_ost = [d.split('@')[0] for d in self.donor]
        self.akceptor_ost = [a.split('@')[0] for a in self.akceptor]

        self.donor_ost_atom = [d.split('@')[1] for d in self.donor]
        self.akceptor_ost_atom = [a.split('@')[1] for a in self.akceptor]
        
    def load_values_in_graph(self, donor, akceptor, weights):
        Graph = nx.Graph()
        for i in range (len(donor)):
            if(weights[i] != 0.0):
                Graph.add_edge(donor[i], akceptor[i], weight=weights[i])
        return Graph

    def creat_full_value_graph(self):
        for i in range(len(self.weights)):
            self.Graphs_full.append(self.load_values_in_graph(self.donor, self.akceptor, self.weights[i]))
        # draw_graph(self.Graphs_full[70])

    def creat_atom_value_graph(self):
        for i in range(len(self.weights)):
            self.Graph_atom.append(self.load_values_in_graph(self.donor_atom, self.akceptor_atom, self.weights[i]))
        # draw_graph(self.Graph_atom[70])

    def creat_ost_value_graph(self):
        for i in range(len(self.weights)):
            self.Graph_ost.append(self.load_values_in_graph(self.donor_ost, self.akceptor_ost, self.weights[i]))
        # draw_graph(self.Graph_ost[70])

    def creat_ost_atom_value_graph(self):
        for i in range(len(self.weights)):
            self.Graph_ost_atom.append(self.load_values_in_graph(self.donor_ost_atom, self.akceptor_ost_atom, self.weights[i]))
        # draw_graph(self.Graph_ost_atom[70])
    
    # def calculate_prop(self, G):
    #     property = []
    #     property.append(G.number_of_nodes()) 
    #     property.append(G.number_of_edges()) 
    #     property.append(nx.density(G)) 
    #     # property.append(nx.radius(G)) 
    #     # property.append(nx.diameter(G)) 
    #     property.append(nx.transitivity(G)) 
    #     property.append(nx.average_clustering(G)) 
    #     property.append(nx.edge_connectivity(G)) 
    #     property.append(nx.degree_assortativity_coefficient(G)) 
    #     property.append(nx.algorithms.centrality.estrada_index(G)) 
    #     property.append(nx.algorithms.approximation.clique.large_clique_size(G)) 
    #     return property

    def calculate_prop(self, G):
        property = []   
        X = []
        v = nx.communicability_exp(G)
        for n in G.nodes:
            property.append(nx.degree(G, n))
            for k in v[n]:
                el = v[n][k]
                if(el != 0):
                    property.append(el)
        return property

    def create_x_matrix_atom(self):
        X = np.zeros((len(self.weights), self.property_kol))
        for i in range(len(self.weights)):
            prop = self.calculate_prop(self.Graph_atom[i])
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

    def create_x_matrix_ost_atom(self):
        X = np.zeros((len(self.weights), self.property_kol))
        for i in range(len(self.weights)):
            prop = self.calculate_prop(self.Graph_ost_atom[i])
            for j in range(len(prop)):
                X[i][j] = prop[j]
        return X

    def full_graph_calc(self):
        self.creat_full_value_graph()
        X = np.array(self.create_x_matrix_full())
        Y = np.array(self.surv_time)
        return X, Y

    def atom_graph_calc(self):
        self.creat_atom_value_graph()
        X = np.array(self.create_x_matrix_atom())
        Y = np.array(self.surv_time)
        return X, Y
    
    def ost_graph_calc(self):
        self.creat_ost_value_graph()
        X = np.array(self.create_x_matrix_ost())
        Y = np.array(self.surv_time)
        return X, Y

    def ost_atom_graph_calc(self):
        self.creat_ost_atom_value_graph()
        X = np.array(self.create_x_matrix_ost_atom())
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

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('BioProject/Bio/graph_value.csv')
    
    # structure.property_kol = 1062
    # X, Y = structure.full_graph_calc() #1
    # structure.property_kol = 1068
    # X, Y = structure.ost_graph_calc() #2

    structure.property_kol = 650
    X, Y = structure.atom_graph_calc() #3 650
    # structure.property_kol = 182
    # X, Y = structure.ost_atom_graph_calc() #4 182
    
    ut = ls_ut(X, Y)
    components = [5, 7, 10, 12]

    # for k in components:
    #     y_oz, R = pls_prediction(X, Y, k)
    #     print(R)
    # print('---')
    # ec = ut.ErrorPLS1Classic(X, Y)
    # for k in ec:
    #     print(ec[k])
    print('---')
    ecr = ut.ErrorPLS1Robust(X, Y)
    for k in ecr:
        print(ecr[k])
    # print('---')
    # ecv = ut.ErrorCVClassic(X, Y)
    # for k in ecv:
    #     print(ecv[k])
    # print('---')
    # ecvr = ut.ErrorCVRobust(X, Y)
    # for k in ecvr:
    #     print(ecvr[k])


    # y_oz, R = pls_prediction(X, Y, 12)
    # ut.CreateTwoPlot(Y, y_oz)

main_graph()