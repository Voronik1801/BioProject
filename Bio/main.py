import csv
from networkx.algorithms.shortest_paths import weighted
from networkx.generators.trees import prefix_tree
import numpy as np
from PLS.Utils.utils import Utils as ls_ut
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


components = [4, 6, 7, 10]

def calc_value(err):
    checkY = copy.copy(Y)
    val = random_value()
    checkY[val] = 100
    ret = Utils.ErrorCVRobust(X, Y)
    for k in components:
        err[k] += ret[k]

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
        self.prop_kol = 11
    
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
    
    def calculate_prop(self, G):
        property = []
        property.append(G.number_of_nodes()) 
        property.append(G.number_of_edges()) 
        property.append(nx.density(G)) 
        property.append(nx.radius(G)) 
        property.append(nx.diameter(G)) 
        property.append(nx.transitivity(G)) 
        property.append(nx.average_clustering(G)) 
        property.append(nx.edge_connectivity(G)) 
        property.append(nx.degree_assortativity_coefficient(G)) 
        property.append(nx.algorithms.centrality.estrada_index(G)) 
        property.append(nx.algorithms.approximation.clique.large_clique_size(G)) 
        return property

    def create_x_matrix_atom(self):
        X = np.zeros((len(self.weights), self.prop_kol))
        for i in range(X.shape[0]):
            prop = self.calculate_prop(self.Graph_atom[i])
            X[i] = prop
        return X
    
    def create_x_matrix_full(self):
        X = np.zeros((len(self.weights), self.prop_kol))
        for i in range(X.shape[0]):
            prop = self.calculate_prop(self.Graphs_full[i])
            X[i] = prop
        return X

    def create_x_matrix_ost(self):
        X = np.zeros((len(self.weights), self.prop_kol))
        for i in range(X.shape[0]):
            prop = self.calculate_prop(self.Graph_ost[i])
            X[i] = prop
        return X

    def create_x_matrix_ost_atom(self):
        X = np.zeros((len(self.weights), self.prop_kol))
        for i in range(X.shape[0]):
            prop = self.calculate_prop(self.Graph_ost_atom[i])
            X[i] = prop
        return X

    def full_graph_calc(self):
        self.creat_full_value_graph()
        X = self.create_x_matrix_full()
        Y = self.surv_time
        return X, Y

    def atom_graph_calc(self):
        self.creat_atom_value_graph()
        X = self.create_x_matrix_atom()
        Y = self.surv_time
        return X, Y
    
    def ost_graph_calc(self):
        self.creat_ost_value_graph()
        X = self.create_x_matrix_ost()
        Y = self.surv_time
        return X, Y

    def ost_atom_graph_calc(self):
        self.creat_ost_atom_value_graph()
        X = self.create_x_matrix_ost_atom()
        Y = self.surv_time
        return X, Y

class LeastSquareMethod():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.teta = np.zeros(len(Y))
        self.y_oz = None

    def calc_teta(self):
        teta = (np.dot(self.X.transpose(), self.X))
        teta = LA.inv(teta)
        teta = np.dot(teta, self.X.transpose())
        # return teta
        self.teta = np.dot(teta, self.Y)

    def calc_y_oz(self):
        self.y_oz = np.dot(self.X, self.teta)
        # e = self.y_oz - self.Y
        # self.y_oz += e 

    def predict(self, X):
        self.calc_teta()
        y_oz = np.dot(X, self.teta)
        return y_oz
    


def cross_validation(X, Y):
    resultCV = []
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        lm = LinearRegression()
        lm.fit(beginX, beginY)
        params = np.append(lm.intercept_,lm.coef_)
        predictYpred = lm.predict(predictX.reshape(1, -1))
        # regression = LeastSquareMethod(beginX, beginY)  # defined pls, default stand nipals
        # predictYpred = regression.predict(predictX.reshape(1, -1))
        resultCV.append(float(predictYpred))
    return resultCV

def error(Y ,y_oz):
    dif = 0
    for i in range(len(y_oz)):
        dif += (y_oz[i] - Y[i]) ** 2
    err = np.sqrt(dif) / 72
    return err
    

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('BioProject/Bio/graph_value.csv')
    # X, Y = structure.ost_atom_graph_calc()
    X, Y = structure.atom_graph_calc()
    # X, Y = structure.full_graph_calc()
    # X, Y = structure.ost_graph_calc()
    # lsm = LeastSquareMethod(X, Y)
    # lsm.calc_teta()
    # lsm.calc_y_oz()
    # y_oz = lsm.y_oz
    ut = ls_ut(X, Y)
    
    # lm = LinearRegression()
    # lm.fit(X, Y)
    # params = np.append(lm.intercept_,lm.coef_)
    # y_oz = lm.predict(X)
    # err = error(Y, y_oz)

    # cv = cross_validation(X, Y)
    # err_cv = error(Y, cv)

    est = sm.OLS(Y, X).fit()
    y_oz = est.predict(X)
    print(est.summary())

    ut.CreateTwoPlot(Y, y_oz)

main_graph()