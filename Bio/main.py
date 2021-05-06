import csv
import numpy as np
from PLS.PLS1 import PLS1
from PLS.Utils import utils

def CrossValidation(X, Y, comp):
    resultCV = np.zeros(X.shape[0])
    for i in range(n):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        regression = PLS1(beginX, beginY, comp, "robust")
        predictYpred = regression.Predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

c = 2

# Open csv and save
with open('table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа
# Defined X and Y for next PLS methosS
n = len(dataForAnalys) - 1
p = len(dataForAnalys[0]) - 1

X = np.zeros((n, p))
Y = np.zeros(n)
Utils = utils(X, Y)
#Saving data for analysis in main structure for pls
Utils.ImportToX(dataForAnalys)
Utils.ImportToY(dataForAnalys)
#
# plsNipals = PLSRegression(n_components=c)  # defined pls, default stand nipals
# plsNipals.fit(X, Y)  # Fit model to data.
# predNipals = plsNipals.predict(X)  # create answer PLS

regress = PLS1(X, Y, c, "c")
other = regress.Predict(X)
# for i in range (len(other)):
#     print (other[i])

# utils.CratePlot(Y, predNipals, other)
#print(CrossValidation(X, Y, 2))