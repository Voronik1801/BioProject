import numpy as np
import numpy.linalg as LA
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
    model.write_x(X.values, X.columns, 'data_for_ols.csv')

    print(f"det: {LA.det(np.dot(X.values.T, X.values))}")
    X = model.centr_norm(X)
    ones = np.ones(len(structure.Graphs))
    X["const"] = ones
    print(f"det: {LA.det(np.dot(X.values.T, X.values))}")

    model.ols_prediction(X, structure.surv_time)