import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import MinMaxScaler

from data_processing import *
from utils import *

###### hyper parameter ######
epoch=40
input_len=20
num_of_prediction_data = 60
num_of_training_data = 120
how_deep = 3

###### variables ######
traffic_file_path = "traffic.pickle"
file_name = "Electric_Production.csv"
#file_name = "weatherHistory.csv"

first_traning_data_idex = 0
last_traning_data_idex = first_traning_data_idex + num_of_training_data - 1
first_prediction_data_idex = last_traning_data_idex + 1
last_prediction_data_idex = first_prediction_data_idex + num_of_prediction_data - 1

################## fetch data ##################
#traffic = get_sin_data(num_of_training_data, num_of_prediction_data)
#traffic = load_traffic_file()
traffic = get_data(file_name, 1, 1)

################## data preprocessing ##################
traffic = smooth_curve(traffic, 2)
traffic = np.array(traffic)

###### normalization ######
traffic = traffic.reshape(-1, 1)
scaler = MinMaxScaler()
traffic_scaled = scaler.fit_transform(traffic)
traffic = traffic_scaled.flatten()
traffic_predict = copy.deepcopy(traffic)

###### create sliding window data ######
buf=[]
x_train=[]
y_train=[]

# get x_train data
for i in range(first_traning_data_idex, last_traning_data_idex - input_len + 1):
    for j in range(i, i + input_len ):
        buf.append(traffic[j])
    x_train.append(buf)
    buf = []

# get y_train data
for i in range(first_traning_data_idex, last_traning_data_idex - input_len + 1):
    y_train.append(traffic[i + input_len])
    

x_train = np.array(x_train, dtype='f')
y_train = np.array(y_train, dtype='f')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

################## model build-up ##################
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(64, input_shape = (input_len, 1), activation ='relu')) # tanh relu
for i in range(0, how_deep):
    model.add(Dense(32))
model.add(Dense(1))
model.summary()

################## training ##################
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=epoch, batch_size=1)

################## prediction ##################
x_predict = []

for i in range(first_prediction_data_idex - input_len, last_prediction_data_idex - input_len + 1):
    for j in range(i, i + input_len ):
        buf.append(traffic_predict[j])
    x_predict = np.array([buf], dtype='f')
    buf = []
    x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
    y_predict = model.predict(x_predict)    
    traffic_predict[i + input_len] = y_predict[0][0]
   
################## data post-processing ##################
traffic_predict_debug = traffic_predict
traffic_predict = traffic_predict[first_prediction_data_idex: last_prediction_data_idex + 1]
traffic_train = traffic[first_traning_data_idex: last_traning_data_idex + 1]
traffic_real = traffic[first_prediction_data_idex: last_prediction_data_idex + 1]
traffic_for_continous_plot = traffic[last_traning_data_idex : first_prediction_data_idex + 1]

###### inverse scaling data ######
traffic_predict = traffic_predict.reshape(-1, 1)
traffic_predict = scaler.inverse_transform(traffic_predict)
traffic_predict = traffic_predict.flatten()

traffic_train = traffic_train.reshape(-1, 1)
traffic_train = scaler.inverse_transform(traffic_train)
traffic_train = traffic_train.flatten()

traffic_real = traffic_real.reshape(-1, 1)
traffic_real = scaler.inverse_transform(traffic_real)
traffic_real = traffic_real.flatten()

traffic_for_continous_plot = traffic_for_continous_plot.reshape(-1, 1)
traffic_for_continous_plot = scaler.inverse_transform(traffic_for_continous_plot)
traffic_for_continous_plot = traffic_for_continous_plot.flatten()

traffic_predict_debug = traffic_predict_debug.reshape(-1, 1)
traffic_predict_debug = scaler.inverse_transform(traffic_predict_debug)
traffic_predict_debug = traffic_predict_debug.flatten()

###### plot ######
time = []
for i in range(0, len(traffic)):
    time.append(i)
time_predict = time[first_prediction_data_idex : last_prediction_data_idex + 1]
time_train = time[first_traning_data_idex : last_traning_data_idex + 1]
time_real = time[first_prediction_data_idex:last_prediction_data_idex+1]
time_for_continous_plot = time[last_traning_data_idex : first_prediction_data_idex + 1]

plt.plot(history.history["loss"])
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(time_predict, traffic_predict, marker='o', ms=3, color = 'red', label="prediction")
plt.plot(time_train, traffic_train, color = 'gray', marker='o', ms=3, label="training data")
plt.plot(time_for_continous_plot, traffic_for_continous_plot, color = 'gray')
plt.plot(time_real, traffic_real, color = 'orange', marker='o', ms=3, label="real value")   # linestyle = 'dotted', alpha=0.3,
#plt.plot(time, traffic_predict_debug, color = 'green', alpha=0.3, label="real value")   # linestyle = 'dotted'

'''
# vertical line
plt.axvline(x=last_prediction_data_idex, color="magenta", linestyle="--", alpha=0.3, label="last prediction")
'''

plt.title("Time Series Forcasting")
plt.xlabel("Time")
plt.ylabel("Electric Production")
plt.legend()
plt.show()
