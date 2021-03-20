import numpy as np

def PLS1(X, Y, components):
    X0 = X
    Xt = X.transpose()
    y0 = Y
    weight = Xt.dot(y0)