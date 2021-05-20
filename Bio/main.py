import csv
import numpy as np
from PLS.Utils.utils import Utils
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression

components = 3
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
Y[56] = Y[56] * 100

# predNipals = np.zeros(72)
# plsNipals = PLSRegression(n_components=components)  # defined pls, default stand nipals
# plsNipals.fit(X, Y)  # Fit model to data.
# predNipals = plsNipals.predict(X)  # create answer PLS

# regress1 = PLS1Regression(X, Y, components, "c")
# plsPredict1 = regress1.Predict(X)

# plsPredict = np.zeros(72)
# regress = PLS1Regression(X, Y, components, "robust")
# plsPredict = regress.Predict(X)
# plsPredict = CrossValidationRobust(X, Y, components)
#print (plsPredict)
#print (predNipals[56])
# utils.CrateThreePlot(Y, plsPredict1, plsPredict)
#utils.PrintErrorPLS1Robust(X,Y, components)
# utils.PrintErrorPLS1Classic(X,Y, components)
# utils.PrintErrorLib(X,Y, components)
# print("Robust")
# utils.PrintErrorCVRobust(X,Y)
# print("Classic")
# utils.PrintErrorCVClassic(X,Y)
utils.PrintErrorCVRobust(X,Y)