import csv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import numpy as np


class PLS1Regression:

    def PLS1(self, X, Y, components):
        Xk = X
        y = Y

        n = Xk.shape[0]
        p = Xk.shape[1]
        q_scalar = 1

        W = np.zeros((p, components))
        P = np.zeros((p, components))
        q = np.zeros(components)

        pk = np.zeros((1,p))

        weight = X.T.dot(y)/np.linalg.norm(X.T.dot(y))
        W[:,0]= weight

        for k in range (0,components):
            tk = Xk.dot(W[:,k])
            tk_scalar = tk.T.dot(tk)
            tk = tk/tk_scalar
            pk=Xk.T.dot(tk)
            P[:,k] = pk.T
            q[k] = y.T.dot(tk)
            if q[k] == 0:
                components = k
                break
            if k < components-1:
                help1 = np.zeros((1,n))
                help1[k,:] = tk*tk_scalar
                help2 =(help1[k]) * pk.T
                Xk = Xk - help2.T
                W[:,k] = Xk.transpose().dot(y)

        helpPW = P.transpose().dot(W)
        B = (W.dot(np.linalg.inv(helpPW))).dot(q)
        B0 = q[0] - P[:,0].transpose().dot(B)
        return B, B0

    def Predict(self,components, X, Y):
        B, B0 = self.PLS1(X, Y, components)
        return X.dot(B) + B0


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


utils = Utils()
regression = PLS1Regression()
# Open csv and save
with open('File/table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа
# Defined X and Y for next PLS methosS
n = len(dataForAnalys)- 1
p = len(dataForAnalys[0]) - 1

X = np.zeros((n, p))
Y = np.zeros(n)

# Saving data for analysis in main structure for pls
utils.ImportToX(dataForAnalys)
utils.ImportToY(dataForAnalys)

plsNipals = PLSRegression(n_components=1)  # defined pls, default stand nipals
plsNipals.fit(X, Y)  # Fit model to data.
predNipals = plsNipals.predict(X)  # create answer PLS

#plt.plot(Y)
#plt.plot(predNipals)

other = regression.Predict(1,X,Y)
plt.plot(other)
plt.show()
