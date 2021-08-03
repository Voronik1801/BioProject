import csv
import numpy as np
from PLS.Utils.utils import Utils
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression


# Open csv and save
with open('table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)

# Defined X and Y for next PLS methosS
n = len(dataForAnalys) - 1
p = len(dataForAnalys[0]) - 1

X = np.zeros((n, p))
Y = np.zeros(n)
utils = Utils(X, Y)

#Saving data for analysis in main structure for pls
utils.ImportToX(dataForAnalys)
utils.ImportToY(dataForAnalys)

Y[5] = 50
Y[56] = 100
# utils.PrintErrorCVRobust(X,Y)
utils.PrintErrorPLS1Robust(X, Y)

# utils.PrintErrorPLS1Classic(X, Y)