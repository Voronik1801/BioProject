import csv
from networkx.algorithms.shortest_paths import weighted
from networkx.generators.trees import prefix_tree
import numpy as np
from PLS.Utils.utils import Utils
import random
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression
from concurrent.futures import ThreadPoolExecutor
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot

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
        property.append(0)
        return property

    def create_x_matrix(self):
        X = np.zeros((len(self.weights), 11))
        for i in range(X.shape[0]):
            prop = self.calculate_prop(self.Graph_atom[i])
            X[i] = prop
        return X

def main_graph():
    structure = GraphStructure()
    structure.calculate_main_values('BioProject/Bio/graph_value.csv')
    # structure.creat_full_value_graph()
    # for i in range (len(structure.Graphs_full)):
    #     structure.calculate_prop(structure.Graphs_full[i])
    structure.creat_atom_value_graph()
    structure.create_x_matrix()
    # for i in range (len(structure.Graph_atom)):
    #     structure.calculate_prop(structure.Graph_atom[i])
    # structure.creat_ost_value_graph()
    # structure.creat_ost_atom_value_graph()
    

main_graph()