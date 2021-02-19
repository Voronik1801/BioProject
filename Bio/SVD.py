import csv
from sklearn.cross_decomposition import PLSSVD
import matplotlib.pyplot as plt


# Print matrix
def PrintMatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], " ", end="")
        print()


# Saving to matrix X in list for Analysis, reading in csv
def ImportToX(matrix):
    for i in range(1, len(matrix)):
        for j in range(0, len(matrix[i]) - 1):
            X[i - 1][j] = float(matrix[i][j])


# Saving to vector Y in list for Analysis, reading in csv
def ImportToY(matrix):
    for i in range(1, len(matrix)):
        Y[i - 1] = float(matrix[i][len(matrix[i])-1])


# Open csv and save
with open('File/data2.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа

# Defined X and Y for next PLS methosS
X = [0] * (len(dataForAnalys) - 1)
for i in range(0, len(dataForAnalys) - 1):
    X[i] = [0] * (len(dataForAnalys[i]) - 1)
Y = [0] * (len(dataForAnalys[1]) - 1)
# Saving data for analysis in main structure for pls
ImportToX(dataForAnalys)
ImportToY(dataForAnalys)
# print(X)
# print(Y)

plsSVD = PLSSVD(n_components=1)  # defined pls, default stand nipals
plsSVD.fit(X, Y)  # Fit model to data.
PLSSVD()  # create PLS


