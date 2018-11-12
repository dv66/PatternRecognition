import numpy as np


class BayesianClassifier:

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
        mean = np.mean(X,axis=0)
        return mean


    def __getCovariance(self, X):
        trans = np.transpose(X)
        covmat = np.cov(trans)
        return covmat


    def __getParameters(self, X):
        mean = self.__getMean(X)
        variance = self.__getCovariance(X)
        return (mean, variance)


    def __getSubsetData(self, cls, X):
        nX = []
        for x in X:
            if x[-1] == cls: nX.append(x)
        return np.array(nX)[:, :-1]


    def __estimateProbabilities(self, X):
        self.__parameters = {}
        self.__numFeatures = len(X[0])-1
        # for i in range(self.__numFeatures): self.__parameters[i] = {}
        for c in self.__classes:
            subset = self.__getSubsetData(c, X)
            self.__parameters[c] = self.__getParameters(subset)


    
    def __calculateProbaFromDensity(self, x, mean, variance):
        d = float(len(mean))

        prob = 1/((pow(2*np.pi, 1/d)) * np.sqrt(np.linalg.det(variance)))

        # temp = np.mat(x-mean) * np.mat(np.linalg.inv(variance))
        temp = np.mat(np.linalg.inv(variance)) * np.transpose(np.mat(x - mean))
        val = np.mat(x-mean) * temp
        return prob * np.exp(-0.5 * np.array(val)[0][0])





    def train(self, dataset):
        self.__getClassProbabilities(dataset)
        self.__estimateProbabilities(dataset)




    def predict(self, sample):
        findClass = {}
        for c in self.__classes: findClass[c] = self.__classProbabilities[c]
        for c in self.__classes:
            cParams = self.__parameters[c]
            mean, variance = cParams[0], cParams[1]
            findClass[c] *= self.__calculateProbaFromDensity(sample, mean, variance)
        pred = []
        for x in findClass: pred.append((findClass[x], x))
        pred = sorted(pred)
        return pred[-1][1]


