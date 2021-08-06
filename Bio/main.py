import csv
import numpy as np
from PLS.Utils.utils import Utils
import random
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression
from concurrent.futures import ThreadPoolExecutor
import copy

components = [4, 6, 7, 10]

def calc_value(err):
    checkY = copy.copy(Y)
    val = random_value()
    checkY[val] = 100
    ret = utils.ErrorCVRobust(X, Y)
    for k in components:
        err[k] += ret[k]

def random_value():
    value = random.randint(0, 71)
    return value

# Open csv and save
with open('table_aMD_th_0.80_00.csv', 'r') as csv_file:
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

