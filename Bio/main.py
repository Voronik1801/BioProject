import csv
import numpy as np
from PLS.Utils.utils import Utils
from PLS.PLS1.PLS1 import PLS1Regression
from sklearn.cross_decomposition import PLSRegression


components = 22
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

plsNipals = PLSRegression(n_components=components)  # defined pls, default stand nipals
plsNipals.fit(X, Y)  # Fit model to data.
predNipals = plsNipals.predict(X)  # create answer PLS

regress = PLS1Regression(X, Y, components, "robust")
plsPredict = regress.Predict(X)

utils.CrateThreePlot(Y, predNipals, plsPredict)
#print(CrossValidation(X, Y, 2))