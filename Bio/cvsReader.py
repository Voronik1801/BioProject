import csv
from sklearn.cross_decomposition import PLSRegression
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
        for j in range(len(matrix[i]) - 1):
            X[i - 1][j] = int(matrix[i][j])


# Saving to vector Y in list for Analysis, reading in csv
def ImportToY(matrix):
    for i in range(1, len(matrix)):
        for j in range(len(matrix[i]) - 1, len(matrix[i])):
            Y[i - 1] = int(matrix[i][j])


# Open csv and save
with open('File/data2.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    dataForAnalys = list(csv_reader)  # рабочий и самый подходящий вариант для дальнейшего анализа

    # Defined X and Y for next PLS methosS
    X = [0] * (len(dataForAnalys) - 1)
    for i in range(len(dataForAnalys) - 1):
        X[i] = [0] * (len(dataForAnalys[i]) - 1)
        Y = [0] * (len(dataForAnalys[i]) - 1)
    # Saving data for analysis in main structure for pls
    ImportToX(dataForAnalys)
    ImportToY(dataForAnalys)

plsNipals = PLSRegression(n_components=1) #defined pls, default stand nipals
plsNipals.fit(X, Y) #Fit model to data.
PLSRegression() #create PLS
predNipals = plsNipals.predict(X) #create answer PLS
print(predNipals)
plt.plot(predNipals)
plt.show()
#this method dont work, why? i all do like in documentation
#plsSVD = PLSRegression(n_components=1, algorithm="svd")
