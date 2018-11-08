import numpy as np


class GaussianNaiveBayesClassifier:

    def __init__(self):
        self.__classProbabilities = None
        self.__parameters = None
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
            

    def __getMean(self, X):
        return sum(X) / len(X)


    def __getVariance(self, X):
        mean = self.__getMean(X)
        s = 0.0
        n = len(X)
        for x in X:
            s += ((x - mean) ** 2)
        return s/(n-1)
        # return np.std(X)

    def __getParameters(self, X):
        mean = self.__getMean(X)
        variance = self.__getVariance(X)
        return mean, variance


    def __getSubsetData(self, cls, X):
        nX = []
        for x in X:
            if x[-1] == cls: nX.append(x)
        return np.array(nX)[:, :-1]


    def __estimateProbabilities(self, X):
        self.__parameters = {}
        self.__numFeatures = len(X[0])-1
        for i in range(self.__numFeatures): self.__parameters[i] = {}
        for c in self.__classes:
            subset = self.__getSubsetData(c, X)
            for i in range(self.__numFeatures):
                ''' calculating parameters(mu, sigma) for all points of feature i under class c'''
                self.__parameters[i][c] = self.__getParameters(subset[:, i])


    
    def __calculateProbaFromDensity(self, x, mean, variance):
        prob = 1 / (np.sqrt(2 * np.pi * variance)) \
               * np.exp(-((x-mean)**2)/(2*variance))
        return prob




    def train(self, dataset):
        self.__getClassProbabilities(dataset)
        self.__estimateProbabilities(dataset)



    def predict(self, sample):
        findClass = {}
        for c in self.__classes: findClass[c] = self.__classProbabilities[c]
        for i in range(self.__numFeatures):
            params = self.__parameters[i]
            for c in self.__classes:
                cParams = params[c]
                mean, variance = cParams[0], cParams[1]
                findClass[c] *= self.__calculateProbaFromDensity(sample[i], mean, variance)
        pred = []
        for x in findClass: pred.append((findClass[x], x))
        pred = sorted(pred)
        return pred[-1][1]


