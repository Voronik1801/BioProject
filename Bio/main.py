import csv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd


def CrossValidation(X, Y, comp):
    resultCV = np.zeros(X.shape[0])
    for i in range(n):
        beginX = X
        predictX = X[i]
        beginX = np.delete(beginX, [i], 0)
        regression = PLS1Regression(X,Y,comp)
        predictY = regression.Predict(predictX)
        resultCV[i] = predictY
    return resultCV


class PLS1Regression:
    def __init__(self, _X, _Y, _components):
        #Кол-во компонени должно быть меньше количества признаков, иначе нельзя раскладывать
        if _X.shape[1] < _components:
            print("error")
        else:
            self.X = _X # matrix of predictors
            self.Y = _Y # vector of real data
            self.components = _components # number of latent variable
            self.n = _X.shape[0] # number of testing people
            self.p = _X.shape[1] # number of properties
            self.B, self.B0 = self.PLS1() # regression coef and regression const

    def centerscale(self,scale=True):
        self.X -= np.amin(self.X, axis=(0, 1))
        self.X /= np.amax(self.X, axis=(0, 1))



    def PLS1(self):
        #init X0 & y0
        self.centerscale()
        Xk = self.X
        y = self.Y

        # n - size of testee
        # p - size of properties


        # W - help matrix of weights
        # P, q - Load matrix
        W = np.zeros((self.p, self.components))
        P = np.zeros((self.p, self.components))
        b = np.zeros(self.components)
        t = np.zeros((self.n, self.components))  # x_scores
        p_loading = np.zeros((self.p, self.components)) # x_loading

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

    def Predict(self, X):
        return X.dot(self.B) + self.B0

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
        ax.plot(x, data2, label='Библиотечная оценка')
        ax.plot(x, data3, label='Оценка собственной реализации')
        ax.set_ylabel("Время дожития, (лет)")
        ax.set_xlabel("Номер пациента")
        ax.legend()
        ax.grid()

        fig.set_figheight(5)
        fig.set_figwidth(16)
        plt.show()
        plt.show()

    def PrintError(self,X,Y):
        #Print err
        for k in range (15,31):
            # dataCV = CrossValidation(X,Y,k)
            #
            # plsNipals = PLSRegression(n_components=k)  # defined pls, default stand nipals
            # plsNipals.fit(X, Y)  # Fit model to data.
            # predNipals = plsNipals.predict(X)  # create answer PLS

            regress = PLS1Regression(X,Y, k)
            other = regress.Predict(X)
            err = np.zeros(n)
            scal = 0
            for j in range(0, len(Y)):
                # err[j] = (dataCV[j]-Y[j])**2
                # err[j] = (predNipals[j]-Y[j])**2
                err[j] = (other[j]-Y[j])**2
                scal+=err[j]
            print(np.sqrt(scal),"\t")



utils = Utils()
# Open csv and save
with open('File/table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа
# Defined X and Y for next PLS methosS
n = len(dataForAnalys)- 1
p = len(dataForAnalys[0]) - 1

components = 30

X = np.zeros((n, p))
Y = np.zeros(n)

# Saving data for analysis in main structure for pls
utils.ImportToX(dataForAnalys)
utils.ImportToY(dataForAnalys)

# plsNipals = PLSRegression(n_components=components)  # defined pls, default stand nipals
# plsNipals.fit(X, Y)  # Fit model to data.
# predNipals = plsNipals.predict(X)  # create answer PLS

# regress = PLS1Regression(X,Y, components)
# other = regress.Predict(X)
dataCV = np.zeros(n)
#dataCV = CrossValidation(X,Y,components)

# utils.CratePlot(Y,predNipals,other)
utils.PrintError(X,Y)



