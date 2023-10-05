import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM,Dense
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras import regularizers
import numpy.ma as ma
import time
from tensorflow.keras import activations
import random
# function that splits the df into training and testing
def splitTrain(df):
    length = len(df)
    trainingRows = []
    testingRows = []
    for i in range(int(length * 0.20)):
        trainingRows.append(random.randint(length))

    for i in range(length):
        if i not in trainingRows:
            testingRows.append(i)


"""
For step 1, we implement a global model. All data from all sites in one model.
"""
model = (train)

