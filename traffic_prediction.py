import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os

#### what we want to predict
def function(x):
    # return np.power(1.15,x)
    # return x*np.sin(x)
    # return x**3
    # return np.log(x)
    return x+0.1

def load_traffic_file():
    if os.path.exists(traffic_file_path):
        with open(traffic_file_path, "rb") as f:
            traffic = pickle.load(f) 
        
    return traffic

traffic_file_path = "traffic.pickle"

#### hyper parameter
epoch=7
input_len=3
num_of_prediction = 10

first_traning_data_idex = 10
last_traning_data_idex = 15

first_prediction_data_idex = last_traning_data_idex + 1
last_prediction_data_idex = first_prediction_data_idex + num_of_prediction

#### data preprocessing
buf=[]
x=[]
x_train=[]
x_test=[]
y_real=[]
y_train=[]
time = []

traffic = load_traffic_file()

# test
traffic = []
for i in range (0, 100):
    traffic.append(i)

# get x_train data
for i in range(first_traning_data_idex, last_traning_data_idex + 1):
    for j in range(i, i + input_len ):
        buf.append(traffic[j])
    x_train.append(buf)
    buf = []

# get y_train data
for i in range(first_traning_data_idex, last_traning_data_idex + 1):
    time.append(i)    
    y_train.append(traffic[i + input_len])
    y_real.append(traffic[i + input_len])

x_train = np.array(x_train, dtype='f')
y_train = np.array(y_train, dtype='f')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

#### model build-up
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (input_len, 1), activation ='tanh'))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#### training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=epoch, batch_size=1)

prediction_list = []

x_predict = []
buf = []
for i in range(first_prediction_data_idex, first_prediction_data_idex + input_len):
    buf.append(i)
x_predict = np.array([buf], dtype='f')
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)

prediction = model.predict(x_predict)
prediction_list.append(prediction[0][0])



plt.plot(history.history["loss"])
plt.show()


plt.plot(time, y_real, color = 'red')
plt.show()
