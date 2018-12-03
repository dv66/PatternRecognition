import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize, StandardScaler
from neural_network.multilayer_perceptron import MultiLayerPerceptron
from utils.data_cleaner import getCleanedDataset
from utils.normalize import normalize

TRAIN_FILE = '../dataset/trainNN.txt'
TEST_FILE = '../dataset/testNN.txt'


trainData = getCleanedDataset(TRAIN_FILE)
testData = getCleanedDataset(TEST_FILE)


''' normalizing data '''
trainData = normalize(trainData)
testData = normalize(testData)

model = MultiLayerPerceptron([3,4,5], learningRate=0.5)
model.train(trainData)



data = testData
error = 0
k = 0
for d in data:
    pred = model.predict(d)
    if pred != d[-1]: error+=1
    k+=1

print('accuracy = ', 100* (1- error/len(data)))
