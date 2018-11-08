import numpy as np


class CategoricalClassifier:

    def __init__(self):
        self.__classProbabilities = None
        self.__probabilities = None
        self.__classes = None
        self.__totalClasses = None
        self.__numFeatures = None

    def __getClassProbabilities(self, X):
        classes = list(set(X[:, -1]))
        self.__classes = classes
        self.__totalClasses = len(classes)
        self.__classProbabilities = {}
        for c in classes: self.__classProbabilities[c]=0
        for x in X:
            self.__classProbabilities[x[-1]] += 1
        nTotal = len(X)
        for k in self.__classProbabilities:
            self.__classProbabilities[k] /= nTotal


    def __getProbabilites(self, X):
        classes = list(set(X[:, -1]))
        nFeatures = len(X[0])-1
        nSamples = len(X)
        self.__numFeatures = nFeatures
        self.__probabilities = {}
        for i in range(nFeatures): self.__probabilities[i] = {}
        for col in range(nFeatures):
            colData = list(set(X[:, col]))
            for c in colData:
                tempProb = {}
                for cl in classes: tempProb[cl] = 0
                cnt = 0
                for i in range(nSamples):
                    if X[i][col] == c:
                        cnt += 1
                        tempProb[X[i][-1]] += 1
                for t in tempProb: tempProb[t] /= cnt
                self.__probabilities[col][c] = tempProb


    def train(self, dataset):
        self.__getClassProbabilities(dataset)
        self.__getProbabilites(dataset)


    # 3 1 0 1 0 1
    def predict(self, sample):
        findClass = {}
        for c in self.__classes: findClass[c] = self.__classProbabilities[c]
        for i in range(self.__numFeatures):
            p = self.__probabilities[i][sample[i]]
            for c in self.__classes: findClass[c] *= p[c]
        pred = []
        for x in findClass: pred.append((findClass[x] , x))
        pred = sorted(pred)
        return pred[-1][1]



