import numpy as np
import matplotlib.pyplot as plt
import copy
import pickle
import os

#### what we want to predict
def function(x):
    # return np.power(1.15,x)
    # return x*np.sin(x)
    # return x**
    # return np.log(x)
    return x+0.1

def load_traffic_file():
    if os.path.exists(traffic_file_path):
        with open(traffic_file_path, "rb") as f:
            traffic = pickle.load(f) 
        
    return traffic

traffic_file_path = "traffic.pickle"

#### hyper parameter
epoch=100
input_len=10
num_of_prediction_data = 10
num_of_training_data = 75

first_traning_data_idex = 200
last_traning_data_idex = first_traning_data_idex + num_of_training_data - 1

first_prediction_data_idex = last_traning_data_idex + 1
last_prediction_data_idex = first_prediction_data_idex + num_of_prediction_data - 1

#### data preprocessing
buf=[]
x=[]
x_train=[]
x_test=[]
y=[]
y_train=[]
time = []

traffic = load_traffic_file()
traffic_copy = copy.deepcopy(traffic)

# get x_train data
for i in range(first_traning_data_idex, last_traning_data_idex - input_len + 1):
    for j in range(i, i + input_len ):
        buf.append(traffic[j])
    x_train.append(buf)
    buf = []

# get y_train data
for i in range(first_traning_data_idex, last_traning_data_idex - input_len + 1):
    time.append(i)
    y.append(traffic[i])    
    y_train.append(traffic[i + input_len])
    

x_train = np.array(x_train, dtype='f')
y_train = np.array(y_train, dtype='f')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

#### model build-up
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
#model.add(LSTM(7, input_shape = (input_len, 1), activation ='relu'))
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

for i in range(first_prediction_data_idex - input_len, last_prediction_data_idex - input_len + 1):
    for j in range(i, i + input_len ):
        buf.append(traffic_copy[j])
    
    x_predict = np.array([buf], dtype='f')
    buf = []
    x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
    prediction = model.predict(x_predict)
    traffic_copy[i + input_len] = int(prediction[0][0])

    #prediction_list.append(prediction[0][0])

aa = traffic_copy[first_traning_data_idex:last_prediction_data_idex + 1]
cc = traffic[first_traning_data_idex:last_prediction_data_idex + 1]

bb = []
for i in range(0,len(aa)):
    bb.append(i)

plt.plot(history.history["loss"])
plt.show()


plt.plot(bb, aa, color = 'red')
plt.plot(bb, cc, color = 'blue')
plt.show()
