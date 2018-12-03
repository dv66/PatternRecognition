import numpy as np





class Neuron:

    def __init__(self, numInputs):
        self.__numInputs = numInputs
        """ weight-vector includes bias value as the last item """
        self.__weightVector = [np.random.uniform(-1,1) for i in range(numInputs+1)]
        self.__bias = self.__weightVector[-1]
        self.__outPut = None
        self.__netValue = None
        pass


    def __sigmoid(self,x, a=1):
        return 1/(1+np.exp(-a*x))


    def processInput(self, x):
        x = np.array(x.tolist() + [1])
        self.__netValue = np.dot(self.__weightVector,x)
        self.__outPut = self.__sigmoid(self.__netValue)

    def getNetvalue(self):
        return self.__netValue

    def getOutput(self):
        return self.__outPut

    def updateWeights(self, newWeights):
        self.__weightVector = newWeights

    def getCurrentWeightVector(self):
        return self.__weightVector

    def __repr__(self):
        return "Neuron = " + str(self.__weightVector ) + \
            ' --------> Output = ' + str(self.__outPut)

