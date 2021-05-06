import csv
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import copy

d = 1.345


def CrossValidation(X, Y, comp):
    resultCV = np.zeros(X.shape[0])
    for i in range(n):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        regression = PLS1Regression(beginX, beginY, comp, "robust")
        predictYpred = regression.Predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV


class PLS1Regression:

    def __init__(self, _X, _Y, _components, mode):
        # Кол-во компонени должно быть меньше количества признаков, иначе нельзя раскладывать
        if _X.shape[1] < _components:
            print("error")
        else:
            self.X = _X  # matrix of predictors
            self.Y = _Y  # vector of real data
            self.components = _components  # number of latent variable
            self.n = _X.shape[0]  # number of testing people
            self.p = _X.shape[1]  # number of properties
            if (mode == "robust"):
                self.B, self.B0 = self.PLS1Robust()  # regression coef and regression const
            else:
                self.B, self.B0 = self.PLS1()  # regression coef and regression const

    def Hiuber(self, x):
        if abs(x) < d:
            return 1.0 / 2.0 * x ** 2
        else:
            return d * (abs(x) - 1.0 / 2.0 * d)

    def MinimazeFunc(self, b):
        f = 0
        Multiply = np.dot(self.X, b)
        for i in range(len(self.Y)):
            f += self.Hiuber(self.Y[i] - Multiply[i])
        return f

    def ExploratorySearch(self, startB, delta):
        trialStep = copy.copy(startB)

        isStepMade = 0
        for i in range(len(startB)):
            trialStep[i] = startB[i] + delta[i]
            if self.MinimazeFunc(trialStep) > self.MinimazeFunc(startB):
                trialStep[i] = startB[i] - delta[i]
                if self.MinimazeFunc(trialStep) > self.MinimazeFunc(startB):
                    isStepMade = 0
                    # keep
                else:
                    startB[i] = trialStep[i]
                    isStepMade = 1
            else:
                startB[i] = trialStep[i]
                isStepMade = 1

        return isStepMade, startB

    def HookaJivsa(self, startB, delta, step, error, maxiter):
        if step < 2:
            print("the step should be larger")
        else:
            x = copy.copy(startB)
            for i in range(maxiter):
                funcValue = self.MinimazeFunc(x)
                isStepMade, x = self.ExploratorySearch(startB, delta)
                while (not isStepMade):
                    delta /= step
                    isStepMade, x = self.ExploratorySearch(startB, delta)
                    if abs(self.MinimazeFunc(x) - funcValue) < error:
                        return x
            return x

    def centerscale(self, scale=True):
        self.X -= np.amin(self.X, axis=(0, 1))
        self.X /= np.amax(self.X, axis=(0, 1))

    def PLS1(self):
        # init X0 & y0
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
        p_loading = np.zeros((self.p, self.components))  # x_loading

        W[:, 0] = Xk.T.dot(y) / np.linalg.norm(Xk.T.dot(y))

        for k in range(0, self.components):
            t[:, k] = np.dot(Xk, W[:, k])  # x_scores
            tk_scalar = np.dot(t[:, k].T, t[:, k])
            t[:, k] = t[:, k] / tk_scalar

            P[:, k] = np.dot(Xk.T, t[:, k])
            b[k] = np.dot(y.T, t[:, k])

            if b[k] == 0:
                components = k
                break

            if k < self.components - 1:
                help1 = tk_scalar * t[:, k]
                help2 = np.outer(help1, P[:, k].T)

                Xk = Xk - help2
                W[:, k + 1] = Xk.transpose().dot(y)

        helpPW = P.transpose().dot(W)
        B = (W.dot(np.linalg.inv(helpPW))).dot(b)
        B0 = b[0] - P[:, 0].transpose().dot(B)
        return B, B0

    def PLS1Robust(self):
        # init X0 & y0
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
        p_loading = np.zeros((self.p, self.components))  # x_loading

        W[:, 0] = Xk.T.dot(y) / np.linalg.norm(Xk.T.dot(y))

        for k in range(0, self.components):
            t[:, k] = np.dot(Xk, W[:, k])  # x_scores
            tk_scalar = np.dot(t[:, k].T, t[:, k])
            t[:, k] = t[:, k] / tk_scalar

            P[:, k] = np.dot(Xk.T, t[:, k])
            b[k] = np.dot(y.T, t[:, k])

            if b[k] == 0:
                components = k
                break

            if k < self.components - 1:
                help1 = tk_scalar * t[:, k]
                help2 = np.outer(help1, P[:, k].T)

                Xk = Xk - help2
                W[:, k + 1] = Xk.transpose().dot(y)

        helpPW = P.transpose().dot(W)
        B = (W.dot(np.linalg.inv(helpPW))).dot(b)
        delta = np.zeros(len(B))
        delta += 0.00005
        B = self.HookaJivsa(B, delta, 2, 0.01, 50)
        B0 = b[0] - P[:, 0].transpose().dot(B)
        return B, B0

    def Predict(self, X):
        return X.dot(self.B) + self.B0


class Utils:
    # Print matrix
    def PrintMatrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                print(matrix[i][j], " ", end="")
            print()

    # Saving to matrix X in list for Analysis, reading in csv
    def ImportToX(self, matrix):
        for i in range(1, len(matrix)):
            for j in range(0, len(matrix[i]) - 1):
                X[i - 1][j] = float(matrix[i][j])

    # Saving to vector Y in list for Analysis, reading in csv
    def ImportToY(self, matrix):
        for i in range(1, len(matrix)):
            Y[i - 1] = float(matrix[i][len(matrix[i]) - 1])

    def CratePlot(self, data1, data2, data3):

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

    def PrintError(self, X, Y):
        # Print err
        for k in range(1, 31):
            #dataCV = CrossValidation(X, Y, k)

            # plsNipals = PLSRegression(n_components=k)  # defined pls, default stand nipals
            # plsNipals.fit(X, Y)  # Fit model to data.
            # predNipals = plsNipals.predict(X)  # create answer PLS

            regress = PLS1Regression(X, Y, k, "robust")
            other = regress.Predict(X)
            err = np.zeros(n)
            scal = 0
            for j in range(0, len(Y)):
                #err[j] = (dataCV[j] - Y[j]) ** 2

                #err[j] = (predNipals[j]-Y[j])**2

                err[j] = (other[j] - Y[j]) ** 2
                scal += err[j]
            print(np.sqrt(scal), "\t")


c = 2
utils = Utils()
# Open csv and save
with open('table_aMD_th_0.80_00.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа
# Defined X and Y for next PLS methosS
n = len(dataForAnalys) - 1
p = len(dataForAnalys[0]) - 1

X = np.zeros((n, p))
Y = np.zeros(n)

#Saving data for analysis in main structure for pls
utils.ImportToX(dataForAnalys)
utils.ImportToY(dataForAnalys)
#
# plsNipals = PLSRegression(n_components=c)  # defined pls, default stand nipals
# plsNipals.fit(X, Y)  # Fit model to data.
# predNipals = plsNipals.predict(X)  # create answer PLS

# regress = PLS1Regression(X, Y, c, "c")
# other = regress.Predict(X)
# for i in range (len(other)):
#     print (other[i])

# utils.CratePlot(Y, predNipals, other)
utils.PrintError(X,Y)
#print(CrossValidation(X, Y, 2))