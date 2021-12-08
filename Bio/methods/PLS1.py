import numpy as np
import copy
from scipy.optimize import minimize
import time
from .nelder_mead import nelder_mead
from numpy import linalg as LA

d = 3
class PLS1Regression():
    def __init__(self, _X, _Y, _components, mode='classic'):
        # Кол-во компонени должно быть меньше количества признаков, иначе нельзя раскладывать
        if _X.shape[1] < _components:
            print("error")
        else:
            self.X = _X  # matrix of predictors
            self.Y = _Y  # vector of real data
            self.B = 0
            self.B0 = 0
            self.components = _components  # number of latent variable
            self.n = _X.shape[0]  # number of testing people
            self.p = _X.shape[1]  # number of properties
            if(mode == 'robust'):
                self.PLS1Robust()
            else:
                self.PLS1()


    def Hiuber(self, x):
        if abs(x) < d:
            return 1.0 / 2.0 * x ** 2
        else:
            return d * (abs(x) - 1.0 / 2.0 * d)

    def MinimazeFunc(self, b):
        f = np.zeros(self.n)
        multiply = np.dot(self.X, b)
        dif = self.Y - multiply
        for i in range(len(self.Y)):
            f[i] = self.Hiuber(dif[i])
        f = np.sum(f)
        return f



    # def ExploratorySearch(self, startB, delta):
    #     trialStep = copy.copy(startB)
    #     isStepMade = 0
    #     for i in range(len(startB)):
    #         trialStep[i] = startB[i] + delta[i]
    #         if self.MinimazeFunc(trialStep) > self.MinimazeFunc(startB):
    #             trialStep[i] = startB[i] - delta[i]
    #             if self.MinimazeFunc(trialStep) > self.MinimazeFunc(startB):
    #                 isStepMade = 0
    #                 # keep
    #             else:
    #                 startB[i] = trialStep[i]
    #                 isStepMade = 1
    #         else:
    #             startB[i] = trialStep[i]
    #             isStepMade = 1
    #
    #     return isStepMade, startB
    #
    # def HookaJivsa(self, startB, delta, step, error, maxiter):
    #     if step < 2:
    #         print("the step should be larger")
    #     else:
    #         x = copy.copy(startB)
    #         for i in range(maxiter):
    #             funcValue = self.MinimazeFunc(x)
    #             isStepMade, x = self.ExploratorySearch(startB, delta)
    #             while (not isStepMade):
    #                 delta /= step
    #                 isStepMade, x = self.ExploratorySearch(startB, delta)
    #                 if abs(self.MinimazeFunc(x) - funcValue) < error:
    #                     return x
    #         return x

    def centerscale(self, scale=True):
        self.X -= np.amin(self.X, axis=(0, 1))
        self.X /= np.amax(self.X, axis=(0, 1))

    def PLS1(self):
        # self.centerscale()
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
        znam = np.linalg.norm(np.dot(Xk.T, y))
        W[:, 0] = np.divide(np.dot(Xk.T, y), znam)

        for k in range(0, self.components):
            t[:, k] = np.dot(Xk, W[:, k])  # x_scores
            tk_scalar = np.dot(t[:, k].T, t[:, k])
            # print(tk_scalar)
            t[:, k] = np.divide(t[:, k], tk_scalar)

            P[:, k] = np.dot(Xk.T, t[:, k])
            b[k] = np.dot(y.T, t[:, k])

            if b[k] == 0:
                break

            if k < self.components - 1:
                help1 = tk_scalar * t[:, k]
                help2 = np.outer(help1, P[:, k].T)

                Xk = Xk - help2
                W[:, k + 1] = np.dot(Xk.T, y)

        helpPW = np.dot(P.T, W)
        f = open('result_graph_P.txt', 'w')
        for i in range(len(P)):
            for j in range(len(P[0])):
                f.write(str(P[i][j]) + '\t')
            f.write('\n')
        print(b)
        self.B = np.dot((W.dot(np.linalg.inv(helpPW))), b)
        self.B0 = b[0] - np.dot(P[:, 0].T, self.B)



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
        znam = np.linalg.norm(np.dot(Xk.T, y))
        W[:, 0] = np.divide(np.dot(Xk.T, y), znam)

        for k in range(0, self.components):
            t[:, k] = np.dot(Xk, W[:, k])  # x_scores
            # t[:, k] = ne.evaluate("Xk*W[:, k]")  # x_scores
            tk_scalar = np.dot(t[:, k].T, t[:, k])
            t[:, k] = np.divide(t[:, k], tk_scalar)

            P[:, k] = np.dot(Xk.T, t[:, k])
            b[k] = np.dot(y.T, t[:, k])

            if b[k] == 0:
                break

            if k < self.components - 1:
                help1 = tk_scalar * t[:, k]
                help2 = np.outer(help1, P[:, k].T)

                Xk = Xk - help2
                W[:, k + 1] = np.dot(Xk.T, y)

        helpPW = np.dot(P.T, W)
        Bpls = np.dot((W.dot(np.linalg.inv(helpPW))), b)
        # start_time = time.time()
        # resMinimization = minimize (self.MinimazeFunc, Bpls, method="nelder-mead")
        rez = nelder_mead(self.MinimazeFunc, Bpls)
        # end_time = time.time()
        # print(end_time - start_time)
        # self.B = resMinimization.x
        self.B = rez[0]
        self.B0 = b[0] - np.dot(P[:, 0].T, self.B)

    def Predict(self, X):
        return np.dot(X, self.B) + self.B0

