import csv
#from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np

def PLS1(X, Y, components):
    X0 = np.asarray(X)
    Xt = X0.transpose()

    y0 = np.asarray(Y)
    weight = Xt.dot(y0)/np.linalg.norm(Xt.dot(y0))
    Xk = X0
    wk = weight
    for k in range (components):
        tk = Xk.dot(wk)
        tk_subscript = tk.transpose().dot(tk)
        
class Utils:

    # Print matrix
    def PrintMatrix(self,matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                print(matrix[i][j], " ", end="")
            print()


    # Saving to matrix X in list for Analysis, reading in csv
    def ImportToX(self,matrix):
        for i in range(1, len(matrix)):
            for j in range(0, len(matrix[i]) - 1):
                X[i - 1][j] = float(matrix[i][j])


    # Saving to vector Y in list for Analysis, reading in csv
    def ImportToY(self, matrix):
        for i in range(1, len(matrix)):
            Y[i - 1] = float(matrix[i][len(matrix[i])-1])


# Open csv and save
with open('File/table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа
utils = Utils()
# Defined X and Y for next PLS methosS
X = [0] * (len(dataForAnalys) - 1)
for i in range(0, len(dataForAnalys) - 1):
    X[i] = [0] * (len(dataForAnalys[i]) - 1)
Y = [0] * (len(dataForAnalys) - 1)

# Saving data for analysis in main structure for pls
utils.ImportToX(dataForAnalys)
utils.ImportToY(dataForAnalys)
# print(X)
# print(Y)

plsNipals = PLSRegression(n_components=4)  # defined pls, default stand nipals
plsNipals.fit(X, Y)  # Fit model to data.
predNipals = plsNipals.predict(X)  # create answer PLS

#print(predNipals)
# R = plsNipals.score(X,Y)
# print(R)
#print("P for X")
#print(plsNipals.x_loadings_)# Gamma -  в нашем случае это Р для Х
# print("T for X")
# print(plsNipals._x_scores)#Xi - в нашем случае это Т при разложении Х
#print("U for Y")
#print(plsNipals._y_scores)#Omega - в нашем случае это U при разложении Y
# print("Q for Y")
#print(plsNipals.y_loadings_)# значение компонент
#print(plsNipals.coef_)
# for i in range (len(Y)):
#     print(Y[i]-predNipals[i])
    # err = [0] * len(Y)
    # scal = 0
    # for i in range(0, len(Y)):
    #     err[i] = (predNipals[i]-Y[i])**2
    #     scal += err[i]
    #print(sqrt(scal))
    # for j in range(0, len(Y)):
    #     print(predNipals[j])
    # print ("-------------")
    # print(sqrt(Summ))
    # print(predNipals)

#plt.plot(Y)
#plt.plot(predNipals)
#plt.show()

PLS1(X, Y, 1)