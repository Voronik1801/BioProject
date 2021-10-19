import csv
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

def load_values_in_graph(donor, akceptor, weights):
    Graph = nx.Graph()
    for i in range (len(weights)):
        for j in range (len(weights[0])):
            Graph.add_edge(donor[i], akceptor[i], weight=weights[i][j])
    return Graph

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
    


def main_graph():
    df = pd.DataFrame()
    # Open csv and save
    with open('BioProject/Bio/graph_value.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        dataForAnalys = pd.DataFrame(csv_reader)

    stability_hidr_bond = dataForAnalys.loc[:0, 1:] # name of column is stability of the hydrogen bond
    donor_akceptor = dataForAnalys.loc[1:182, :0]
    weights = dataForAnalys.loc[1:182, 1:1]
    print(weights)
    
    donor = [donor[0].split('-')[0] for donor in donor_akceptor.values]
    akceptor = [akceptor[0].split('-')[1] for akceptor in donor_akceptor.values]
    weights = [list(map(float, w)) for w in weights.values]

    donor_atom = [d.split('.')[1] for d in donor]
    akceptor_atom = [a.split('.')[1] for a in akceptor]

    donor_ost = [d.split('@')[0] for d in donor]
    akceptor_ost = [a.split('@')[0] for a in akceptor]

    donor_ost_atom = [d.split('@')[1] for d in donor]
    akceptor_ost_atom = [a.split('@')[1] for a in akceptor]

    Graph_full = load_values_in_graph(donor, akceptor, weights)
    # draw_graph(Graph_full)

    Graph_atom = load_values_in_graph(donor_atom, akceptor_atom, weights)
    draw_graph(Graph_atom)

    Graph_ost = load_values_in_graph(donor_ost, akceptor_ost, weights)
    # draw_graph(Graph_ost)

    Graph_ost_atom = load_values_in_graph(donor_ost_atom, akceptor_ost_atom, weights)
    # draw_graph(Graph_ost_atom)
    calulate_property(Graph_ost_atom)

main_graph()