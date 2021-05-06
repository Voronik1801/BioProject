import numpy as np
import copy

d = 1.345
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

