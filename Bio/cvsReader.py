import csv
def PrintMatrix (matrix):
    for i in range(len(dataForAnalys)):
        for j in range(len(dataForAnalys[i])):
            print(dataForAnalys[i][j], " ", end="")
        print()

with open('File/data.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    dataForAnalys=list(csv_reader) #рабочий и самый подходящий вариант для дальнейшего анализа
    PrintMatrix(dataForAnalys)



#попытки
    # for line in csv_reader:
    #     lengthLine = (len(line))  # как по мне не самый оптимизированный вариант нахождения длины строки
    #     lengthRow += 1
    # print(lengthLine)
    # print(lengthRow)
    # print(line)
    # data = list(line)
    # print(data)

