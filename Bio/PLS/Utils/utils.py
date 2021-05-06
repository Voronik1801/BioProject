import matplotlib.pyplot as plt
import numpy as np
from PLS.PLS1 import PLS1


class Utils:
    def __init__(self, _X, _Y):
            self.X = _X  # matrix of predictors
            self.Y = _Y  # vector of real data
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

            regress = PLS1(X, Y, k, "robust")
            other = regress.Predict(X)
            err = np.zeros(182)
            scal = 0
            for j in range(0, len(Y)):
                #err[j] = (dataCV[j] - Y[j]) ** 2

                #err[j] = (predNipals[j]-Y[j])**2

                err[j] = (other[j] - Y[j]) ** 2
                scal += err[j]
            print(np.sqrt(scal), "\t")

