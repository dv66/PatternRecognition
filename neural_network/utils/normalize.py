import  numpy as np


def normalize(dataSet):
    mean = np.mean(dataSet[:,:-1], axis=0)
    maxElement = np.max(dataSet[:,:-1], axis=0)
    minElement = np.min(dataSet[:, :-1], axis=0)
    dataSet = dataSet.tolist()


    for i in range (len(dataSet)):
        for j in range (len(dataSet[i])-1):
            dataSet[i][j] = (dataSet[i][j]-minElement[j])/(maxElement[j]-minElement[j])

    return np.array(dataSet)