
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, LSTM,Dense
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn.linear_model import LinearRegression

# Import data
O3J = pd.read_csv(r"D:\Will_Git\Ozone_ML\Year2\Merged_Data\merge2.csv")


'''configure the model'''
timesize = 6 ##e.g., past 6 hours
input_var_cnt = 9 ##the number of variables used to perform prediction e.g., NO2, Ozone ... from the previous x tine steps
input_lstm = Input(shape=(timesize, input_var_cnt)) ##what is the input for every sample, sample count is not included every sample should be a 2D matrix
##prepare a LSTM layer
unit_lstm = 32 ##hidden dimensions transfer data, larger more complex model
lstmlayer = LSTM(unit_lstm) (input_lstm) ##this outputs a matrix of 1*unit_lstm, the format is the layer (input), the output of the layer stores the time series info and the interaction of variables..
denselayer = Dense(1)(lstmlayer) ## reduce the hidden dimension to 1 ==== output data ,1 value for 1 input sample --- predicted ozone
model = Model(inputs = input_lstm, outputs = denselayer)
model.compile(loss='mean_squared_error', optimizer='adam') ##how to measure the accuracy  compute mean squared error using all y_pred, y_true

# Loop through stations, run LSTM on data from each one, run regression on the results comparing actual vs expected value
stations = []
outputs = []
trains = []
dtownDen = [float(39.751184)]

#for station in O3J['latitude'].unique():
for station in dtownDen:
    oneStation = O3J[O3J['latitude'] == station]
    stations.append(station)
    oneStation.dropna(inplace=True)
    oneStation.reset_index(inplace=True)

    inputArray = np.zeros(shape=((len(oneStation)),6,9))
    outputArray = np.zeros(shape=((len(oneStation))))
    for index, row in oneStation.iterrows():
        if index < (len(oneStation) - 6):
            input = oneStation[['sample_measurement','t2m', 'r2', 'sp', 'dswrf', 'MAXUVV', 'MAXDVV', 'u10', 'v10']][index:index+6].to_numpy()
            inputArray[index] = input
            outputArray[index] = oneStation['sample_measurement'][index+6]

    model.fit(inputArray, outputArray, epochs=100, batch_size=32, verbose=2)
    trainPredict = model.predict(inputArray)

    outputs.append(outputArray)
    trains.append(trainPredict)

print('why is the model predicting all the outputs to be the same value?')
print(trainPredict[0:10])


def runRegression(xvars, y):
    X = xvars.reshape(-1,1)
    y = y
    model1 = LinearRegression()
    model1.fit(X, y)

    r_sq = model1.score(X, y)
    print(f"coefficient of determination: {r_sq}")
    #print('coefficients '+ str(model1.coef_))
    #print('intercept '+ str(model1.intercept_))
#
# for i in range(len(outputs)):
#     print(stations[i])
#     runRegression(xvars=outputs[i], y=trains[i])

