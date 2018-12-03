import numpy as np


def getCleanedDataset(fileLocation):
    fp = open(fileLocation)
    lines = fp.readlines()
    data = []
    for line in lines:
        line = line.strip().split()
        data.append(line)
    data = np.array(data).astype(float)
    return data