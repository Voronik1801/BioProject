import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import time
from .PLS1 import PLS1Regression
import math

components = [2, 4, 6]

def CrossValidationLib(X, Y, comp):
    resultCV = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        beginX = X
        predictX = X[i]
        beginY = Y
        beginX = np.delete(beginX, [i], 0)
        beginY = np.delete(beginY, [i], 0)
        regression = PLSRegression(n_components=comp)  # defined pls, default stand nipals
        regression.fit(beginX, beginY)  # Fit model to data.
        predictYpred = regression.predict(predictX.reshape(1, -1))
        resultCV[i] = predictYpred
    return resultCV

class Utils():
    def __init__(self, _X, _Y):
            self.X = _X  # matrix of predictors
            self.Y = _Y  # vector of real data

    def CrossValidationRobust(self, comp):
        resultCV = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            beginX = self.X
            predictX = self.X[i]
            beginY = self.Y
            beginX = np.delete(beginX, [i], 0)
            beginY = np.delete(beginY, [i], 0)
            regression = PLS1Regression(beginX, beginY, comp, 'robust')
            resultCV[i] = regression.Predict(predictX.reshape(1, -1))
        return resultCV

    def CrossValidationClassic(self, comp):
        resultCV = np.zeros(self.X.shape[0])
        for i in range(self.X.shape[0]):
            beginX = self.X
            predictX = self.X[i]
            beginY = self.Y
            beginX = np.delete(beginX, [i], 0)
            beginY = np.delete(beginY, [i], 0)
            regression = PLS1Regression(beginX, beginY, comp, "classic")
            resultCV[i] = regression.Predict(predictX.reshape(1, -1))
        return resultCV

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
                self.X[i - 1][j] = float(matrix[i][j])

    # Saving to vector Y in list for Analysis, reading in csv
    def ImportToY(self, matrix):
        for i in range(1, len(matrix)):
            self.Y[i - 1] = float(matrix[i][len(matrix[i]) - 1])

    def CreateThreePlot(self, data1, data2, data3):
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

    def CreateTwoPlot(self, data1, data2):
        x = np.arange(len(data1))
        fig, ax = plt.subplots()
        ax.set_title("Предсказания времени дожития пациентов с БАС")
        ax.plot(x, data1, label='Исходные данные')
        ax.plot(x, data2, label='Оценка')
        ax.set_ylabel("Время дожития, (лет)")
        ax.set_xlabel("Номер пациента")
        ax.legend()
        ax.grid()

        fig.set_figheight(5)
        fig.set_figwidth(16)
        plt.show()
        plt.show()

    def ErrorCVClassic(self, X, Y):
        # Print err
        er = {}
        for k in components:
            CV = self.CrossValidationClassic(k)
            err = np.zeros(X.shape[0])
            scal = 0
            for j in range(0, len(Y)):
                err[j] = (CV[j] - Y[j]) ** 2
                scal += err[j]
            # print(np.sqrt(scal)/72, "\t")
            er[k] = np.sqrt(scal) / 72
        return er

    def ErrorCVLib(self, X, Y):
        # Print err
        err = {}
        for k in components:
            CV = self.CrossValidationLib(k)
            err = np.zeros(X.shape[0])
            scal = 0
            for j in range(0, len(Y)):
                err[j] = (CV[j] - Y[j]) ** 2
                scal += err[j]
            # print(np.sqrt(scal)/72, "\t")
            err[k] = np.sqrt(scal) / 72
        return err

    # def PrintErrorCVRobust(self, X, Y):
    #     # Print err
    #     for k in components:
    #         CV = self.CrossValidationRobust(k)
    #         err = np.zeros(X.shape[0])
    #         scal = 0
    #         for j in range(0, len(Y)):
    #             err[j] = (CV[j] - Y[j]) ** 2
    #             scal += err[j]
    #         print(np.sqrt(scal)/72, "\t")

    def ErrorCVRobust(self, X, Y):
        # Print err
        err = {}
        for k in components:
            CV = self.CrossValidationRobust(k)
            dif = (CV - Y) ** 2
            scal = np.sum(dif)
            # print(np.sqrt(scal)/72, "\t")
            err[k] = np.sqrt(scal) / 72
        return err

    def ErrorPLS1Robust(self, X, Y):
        # Print err
        err = {}
        for k in components:
            regress = PLS1Regression(X, Y, k)
            regress.PLS1Robust()
            plsPredict = regress.Predict(X)
            dif = (plsPredict - Y) ** 2
            scal = np.sum(dif)
            # print((scal)/72, "\t")
            err[k] = np.sqrt(scal) / 72
        return err

    def ErrorPLS1Classic(self, X, Y):
        # Print err
        err = {}
        for k in components:
            regress = PLS1Regression(X, Y, k)
            plsPredict = regress.Predict(X)
            dif = (plsPredict - Y) ** 2
            scal = np.sum(dif)
            err[k] = np.sqrt(scal) / 72
        return err

    def ErrorLib(self, X, Y):
        # Print err
        err = {}
        for k in components:
            plsNipals = PLSRegression(n_components=k)  # defined pls, default stand nipals
            plsNipals.fit(X, Y)  # Fit model to data.
            predNipals = plsNipals.predict(X)  # create answer PLS
            dif = (predNipals - Y) ** 2
            scal = np.sum(dif)
            err[k] = np.sqrt(scal) / 72
        return err