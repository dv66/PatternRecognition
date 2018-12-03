import numpy as np
from neural_network.neuron import Neuron



class MultiLayerPerceptron:
    """
    self.__layers : contains neurons by layers
                    last layer is the output layer

    numNeuronsInLayers : list containing number of neurons in each layers
    """

    def __init__(self, numNeuronsInHiddenLayers, learningRate = 0.5):
        self.__numNeuronsInHiddenLayers = numNeuronsInHiddenLayers
        self.__network = []
        self.__dataSet = None
        self.__learningRate = learningRate



    def __printNeuralNetwork(self):
        k = 1
        for l in self.__network:
            print('Layer -> ', k)
            for n in l:
                print(n)
            print('-------------------------------------------------------------------------------------------------------------')
            k+=1



    def __getError(self, given, found):
        err = 0
        for i in range(len(given)):
            err += ((given[i]-found[i])*(given[i]-found[i]))
        return 0.5 * err


    def __update(self, sample):
        """update weights of the network for a training sample"""

        """processing outputs of first layer neurons"""
        for neuron in self.__network[0]:
            neuron.processInput(sample[:-1])

        """processing outputs of next layers neurons"""
        for i in range (1, len(self.__network)):
            inputForThisLayer = np.array([n.getOutput() for n in self.__network[i-1]])
            for neuron in self.__network[i]:
                neuron.processInput(inputForThisLayer)

        finalOutputs = [n.getOutput() for n in self.__network[-1]]
        given = [1 if i == sample[-1] else 0 for i in range(1, len(finalOutputs)+1)]

        Y = []
        for r in range(len(self.__network)):
            tempY = []
            for neuron in self.__network[r]:
                tempY.append(neuron.getOutput())
            Y.append(tempY)


        delta_values = []
        for y in Y:
            t = []
            for i in range(len(y)): t.append(0)
            delta_values.append(t)


        """ when r = L"""
        for j in range(len(delta_values[-1])):
            out = finalOutputs[j]
            target = given[j]
            delta_values[-1][j] = (out - target)*self.__network[-1][j].getOutput()*(1-self.__network[-1][j].getOutput())

        """ when r < L"""
        for r in range(len(delta_values)-1,0,-1):
            for j in range(len(delta_values[r-1])):
                e = 0
                for k in range(len(delta_values[r])):
                    e += (delta_values[r][k]*self.__network[r][k].getCurrentWeightVector()[j])
                out = self.__network[r-1][j].getOutput()
                delta_values[r - 1][j] = e * out * (1-out)

        """update weights"""
        for r in range(1,len(delta_values)):
            for j in range(len(delta_values[r])):
                delta_w_r_j = [-self.__learningRate * delta_values[r][j] * y for y in Y[r-1]]
                w_old = self.__network[r][j].getCurrentWeightVector()
                w_new = [w_old[i]+delta_w_r_j[i] for i in range(len(delta_w_r_j))] + [w_old[-1]]
                self.__network[r][j].updateWeights(w_new)


        return np.argmax(finalOutputs) == np.argmax(given)





    def predict(self, sample):
        '''
        predicts the class label of a datapoint by forward computation
        :param sample: sample datapoint with label at the end
        :return: predicted class label
        '''

        """processing outputs of first layer neurons"""
        for neuron in self.__network[0]:
            neuron.processInput(sample[:-1])

        """processing outputs of next layers neurons"""
        for i in range(1, len(self.__network)):
            inputForThisLayer = np.array([n.getOutput() for n in self.__network[i - 1]])
            for neuron in self.__network[i]:
                neuron.processInput(inputForThisLayer)

        finalOutputs = [n.getOutput() for n in self.__network[-1]]
        return np.argmax(finalOutputs)+1



    def __trainNetwork(self):
        """
        take each sample from trainset and iteratively update the network
        until all the samples in the training set are classified correctly
        :return:
        nothing
        """

        """ run until all the samples in the training set are classified correctly"""
        nSteps = 0
        while True:
            misClassified = 0
            for d in self.__dataSet:
                pred = self.__update(sample=d)
                if pred == False: misClassified+=1
            if misClassified == 0:
                break
            nSteps+=1
        print('Total Epochs = ' , nSteps)



    def __initializeNeuralNetwork(self):
        for i in range (1, len(self.__numNeuronsInHiddenLayers)):
            layer = []
            for j in range (self.__numNeuronsInHiddenLayers[i]):
                layer.append(Neuron(self.__numNeuronsInHiddenLayers[i-1]))
            self.__network.append(layer)
        pass



    def train(self, dataSet):
        self.__dataSet = dataSet
        self.__numFeatures = len(dataSet[0])-1
        self.__numClasses = len(set(dataSet[:, -1]))
        self.__numNeuronsInHiddenLayers = [self.__numFeatures] +self.__numNeuronsInHiddenLayers + [self.__numClasses]
        self.__initializeNeuralNetwork()
        self.__trainNetwork()


    def DEBUG(self):
        pass



