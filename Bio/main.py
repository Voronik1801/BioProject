import csv
import numpy as np
from PLS.Utils.utils import Utils
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression
def CrossValidationRobust(X, Y, comp):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        regression = PLS1Regression(beginX, beginY, comp, "robust")
        predictYpred = regression.Predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

# def CrossValidationRobust(X, Y, comp):
#     #esultCV = np.zeros(X.shape[0])
#    # for i in range(X.shape[0]):
#     beginX = X
#     predictX = X[56]
#     beginY = Y
#     beginX = np.delete(beginX, [56], 0)
#     beginY = np.delete(beginY, [56], 0)
#     #regression = PLS1Regression(beginX, beginY, comp, "robust")
#     #predictYpred = regression.Predict(predictX.reshape(1, -1))
#     plsNipals = PLSRegression(n_components=components)  # defined pls, default stand nipals
#     plsNipals.fit(beginX, beginY)  # Fit model to data.
#     predictYpred = plsNipals.predict(predictX.reshape(1,-1))  # create answer PLS
#     resultCV = predictYpred
#     return resultCV


components = 30
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
#print(Y[56])
Y[56] = Y[56] * 100

# plsNipals = PLSRegression(n_components=components)  # defined pls, default stand nipals
# plsNipals.fit(X, Y)  # Fit model to data.
# predNipals = plsNipals.predict(X)  # create answer PLS

#regress = PLS1Regression(X, Y, components, "robust")
plsPredict = CrossValidationRobust(X, Y, components)
#plsPredict = regress.Predict(X)
#print (plsPredict)
#print (predNipals[56])
#utils.CrateThreePlot(Y, predNipals, plsPredict)
#print(CrossValidation(X, Y, 2))
#print(CrossValidationRobust(X, Y, components))
utils.PrintErrorCVClassic(X,Y)