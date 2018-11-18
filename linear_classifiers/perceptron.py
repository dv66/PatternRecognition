import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

'''
    Parceptron, pocket and reward punishment only for binary classification
    Kesler for multiclass classification
'''
def perceptron(dataSet):
    w = np.array( [1 for i in range(len(dataSet[0])-1)] + [ np.random.uniform(-1, 1)])
    Y = [-1]
    while Y:
        Y = []
        # rho = np.random.uniform(0, 1)
        rho = 0.7
        for X in dataSet:
            actualLabel = X[-1]
            x = list(X[:-1]) + [1]
            predictedLabel = np.mat(w) * np.transpose(np.mat(x))
            if (actualLabel == 1 and predictedLabel < 0) or \
                    (actualLabel == 2 and predictedLabel > 0):
                Y.append((x, actualLabel))
        val = np.array([0 for i in range(len(dataSet[0]))])
        for y in Y:
            if y[1] == 1:
                val = val - (np.array(y[0]))
            else:
                val = val + (np.array(y[0]))
        w = w - val
    return w



def rewardPunishment(dataSet):
    w = np.array([1 for i in range(len(dataSet[0])-1)] + [np.random.uniform(-1, 1)])
    nMisclassified = -1
    while nMisclassified != 0:
        nMisclassified = 0
        # rho = np.random.uniform(0, 1)
        rho = 0.7
        for X in dataSet:
            actualLabel = X[-1]
            x = list(X[:-1]) + [1]
            predictedLabel = np.mat(w) * np.transpose(np.mat(x))
            if (actualLabel == 1 and predictedLabel < 0):
                w = w + rho* np.array(x)
                nMisclassified+=1
            elif (actualLabel == 2 and predictedLabel > 0):
                w = w - rho * np.array(x)
                nMisclassified+=1
    return w



def pocket(dataSet):
    w = np.array([1 for i in range(len(dataSet[0]) - 1)] + [np.random.uniform(-1, 1)])
    Y = [-1]
    history = 0
    pocket = w
    nTotal = len(dataSet)
    while Y:
        Y = []
        # rho = np.random.uniform(0, 1)
        rho = 0.7
        for X in dataSet:
            actualLabel = X[-1]
            x = list(X[:-1]) + [1]
            predictedLabel = np.mat(w) * np.transpose(np.mat(x))
            if (actualLabel == 1 and predictedLabel < 0) or \
                    (actualLabel == 2 and predictedLabel > 0):
                Y.append((x, actualLabel))
        val = np.array([0 for i in range(len(dataSet[0]))])
        if nTotal-len(Y) > history:
            history = nTotal-len(Y)
            pocket = w
        for y in Y:
            if y[1] == 1:
                val = val - (np.array(y[0]))
            else:
                val = val + (np.array(y[0]))
        w = w - val
    return pocket

















'''
    keslers construction
'''

def keslerPerceptron(dataSet, w):
    Y = [-1]
    while Y:
        Y = []
        rho = 0.5
        for X in dataSet:
            product = np.array(np.mat(w) * np.transpose(np.mat(X)))
            if product[0][0] <= 0:
                Y.append(X)
        val = np.array([0 for i in range(len(dataSet[0]))])
        for y in Y:
            val = val - (np.array(y))
        w = w - val
    return w


def fit(pos,dimension, ex, negTx):
    k = 0
    ex = deepcopy(ex)
    for i in range(pos, pos+dimension):
        ex[i] = negTx[k]
        k+=1
    return ex


### label 1 theke shuru dhorlam
def getExtendedVectors(label, tx, dimension, numClasses):
    base = [0 for i in range(dimension * numClasses)]
    start = (label-1)*dimension
    for i in range(len(tx)):
        base[start+i] = tx[i]
    negTx = [-x for x in tx]
    extendedVectors = [base for i in range(numClasses-1)]
    indices = [i*dimension for i in range(numClasses)]
    k = 0
    for ind in indices:
        if k == len(extendedVectors): break
        if extendedVectors[k][ind] == 0:
            extendedVectors[k] = fit(ind, dimension, extendedVectors[k], negTx)
            k+=1
    return extendedVectors


def keslerConstruction(dataSet):
    dimension = len(dataSet[0])
    numClasses = len(list(set(dataSet[:, -1])))
    Xvectors = []
    for x in dataSet:
        tx = list(x[:-1]) + [1]
        Xvectors += getExtendedVectors(int(x[-1]), tx , dimension, numClasses)
    Xvectors = np.array(Xvectors)
    baseWeight = [1 for i in range(dimension-1)] + [np.random.uniform(-1, 1)]
    Wvectors = np.array([baseWeight for i in range(numClasses)]).flatten()
    return keslerPerceptron(Xvectors, Wvectors)




FILE = '../datasets/synthetic_data_for_bayesian_classifier/Train.txt'
# FILE = '../datasets/synthetic_data_for_bayesian_classifier/Train (1).txt'
# FILE = '../datasets/synthetic_data_for_bayesian_classifier/synthetic_real_valued_data.txt'
fp = open(FILE)
lines = fp.readlines()[1:]
dataSet = []
for line in lines:
    line = line.strip().split()
    line = list(map(float, line))
    dataSet.append(line)
dataSet = np.array(dataSet)


'''
    1 == +ve
    2 == -ve
'''




print(perceptron(dataSet))
# print(rewardPunishment(dataSet))
# print(pocket(dataSet))
# print(keslerConstruction(dataSet))




