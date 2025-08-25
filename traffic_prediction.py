import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

traffic_file_path = "traffic.pickle"

def load_traffic_file():
    if os.path.exists(traffic_file_path):
        with open(traffic_file_path, "rb") as f:
            traffic = pickle.load(f) 
        
    return np.array(traffic)

def get_sin_data():
    x = np.linspace(0, np.pi * 5, num_of_training_data + num_of_prediction_data)
    
    return np.sin(x) 


#### hyper parameter
epoch=10
input_len=5
num_of_prediction_data = 10
num_of_training_data = 30
how_deep = 2

'''
epoch=30
input_len=20
num_of_prediction_data = 30
num_of_training_data = 500
'''

first_traning_data_idex = 0
last_traning_data_idex = first_traning_data_idex + num_of_training_data - 1
first_prediction_data_idex = last_traning_data_idex + 1
last_prediction_data_idex = first_prediction_data_idex + num_of_prediction_data - 1

################## fetch data ##################

traffic = get_sin_data()
#traffic = load_traffic_file()

################## data preprocessing ##################

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
#model.add(LSTM(7, input_shape = (input_len, 1), activation ='relu'))
model.add(LSTM(64, input_shape = (input_len, 1), activation ='tanh'))
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
traffic_predict = traffic_predict[first_traning_data_idex:last_prediction_data_idex + 1]
traffic = traffic[first_traning_data_idex:last_prediction_data_idex + 1]

###### inverse scaling data ######
traffic_predict = traffic_predict.reshape(-1, 1)
traffic_predict = scaler.inverse_transform(traffic_predict)
traffic_predict = traffic_predict.flatten()

traffic = traffic.reshape(-1, 1)
traffic = scaler.inverse_transform(traffic)
traffic = traffic.flatten()

###### plot ######
time = []
for i in range(0,len(traffic)):
    time.append(i)

plt.plot(history.history["loss"])
plt.show()

plt.plot(time, traffic_predict, linestyle = 'dotted', color = 'red')
plt.plot(time, traffic, color = 'gray', alpha=0.3, label="real value")
plt.axvline(x=last_traning_data_idex, color="blue", linestyle="--", alpha=0.3)
plt.legend()
plt.show()
