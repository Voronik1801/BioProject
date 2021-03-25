import csv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd




class PLS1Regression:
    def __init__(self, _X, _Y, _components):
        self.X = _X
        self.Y = _Y
        self.components = _components

    def PLS1(self):
        #init X0 & y0
        Xk = self.X
        y = self.Y

        # n - size of testee
        # p - size of properties
        n = Xk.shape[0]
        p = Xk.shape[1]

        # W - help matrix of weights
        # P, q - Load matrix
        W = np.zeros((p, self.components))
        P = np.zeros((p, self.components))
        b = np.zeros(self.components)
        t = np.zeros((n, self.components))  # x_scores
        p_loading = np.zeros((p, self.components)) # x_loading

        W[:,0] = X.T.dot(y)/np.linalg.norm(X.T.dot(y))

        for k in range (0,self.components):
            t[:,k] = np.dot(Xk, W[:,k]) #x_scores
            tk_scalar = np.dot(t[:,k].T, t[:,k])
            t[:,k] = t[:,k]/tk_scalar

            P[:,k] = np.dot(Xk.T, t[:,k])
            b[k] = np.dot (y.T, t[:,k])

            if b[k] == 0:
                components = k
                break

            if k < self.components-1:
                help1 = tk_scalar * t[:,k]
                help2 = np.outer(help1, P[:,k].T)

                Xk = Xk - help2
                W[:,k+1] = Xk.transpose().dot(y)


        helpPW = P.transpose().dot(W)
        B = (W.dot(np.linalg.inv(helpPW))).dot(b)
        B0 = b[0] - P[:,0].transpose().dot(B)
        #print(P)
        return B, B0

    def Predict(self):
        B, B0 = self.PLS1()
        # for i in range (len(B)):
        #     print(B[i])
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
    def CratePlot (self, data1, data2, data3):

        # plt.title("Title")
        # plt.subplots.plot(data1, legend = 'data1')
        # plt.plot(data2)
        # plt.plot(data3)

        x = np.arange(len(data1))
        fig, ax = plt.subplots()
        ax.set_title("Предсказания времени дожития пациентов с БАС, n=10")
        ax.plot(x, data1, label='Исходные данные')
        ax.plot(x, data2, label='Библиотечные прогнозы')
        ax.plot(x, data3, label='Прогнозы собственной реализации')
        ax.set_ylabel("Время дожития, (лет)")
        ax.set_xlabel("Номер пациента")
        ax.legend()
        ax.grid()

        fig.set_figheight(5)
        fig.set_figwidth(16)
        plt.show()
        plt.show()


utils = Utils()
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

plsNipals = PLSRegression(n_components=10)  # defined pls, default stand nipals
plsNipals.fit(X, Y)  # Fit model to data.
predNipals = plsNipals.predict(X)  # create answer PLS

regress = PLS1Regression(X,Y, 2)
other = regress.Predict()
utils.CratePlot(Y,predNipals,other)

# for i in range (len(other)):
#     print(other[i])

#Print err
# for k in range (1,11):
#     other = regression.Predict(k, X, Y)
#     for i in range (len(Y)):
#         err = [0] * len(Y)
#         scal = 0
#         for j in range(0, len(Y)):
#             err[j] = (other[j]-Y[j])**2
#             scal+=err[j]
#     print(np.sqrt(scal),"\t")


