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

def load_nodes_in_graph(donor, akceptor):
    Graph = nx.Graph()
    Graph.add_nodes_from(donor)
    Graph.add_nodes_from(akceptor)
    return Graph

def load_edges_in_graph(Graph, weights):
    for i in range(len(weights.values)):
        Graph.add_edges_from(weights.values[i])


def main_graph():
    df = pd.DataFrame()
    # Open csv and save
    with open('BioProject/Bio/graph_value.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        dataForAnalys = pd.DataFrame(csv_reader)

    stability_hidr_bond = dataForAnalys.loc[:0, 1:] # name of column is stability of the hydrogen bond
    donor_akceptor = dataForAnalys.loc[1:182, :0]
    weights = dataForAnalys.loc[1:182, 1:]
    
    donor = [donor[0].split('-')[0] for donor in donor_akceptor.values]
    akceptor = [akceptor[0].split('-')[1] for akceptor in donor_akceptor.values]

    Graph = load_nodes_in_graph(donor, akceptor)
    load_edges_in_graph(Graph, weights)


main_graph()