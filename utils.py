import numpy as np
import pickle
import os

def load_traffic_file(file):
    if os.path.exists(file):
        with open(file, "rb") as f:
            traffic = pickle.load(f) 
        
    return traffic

def get_sin_data(num_of_training_data, num_of_prediction_data):
    x = np.linspace(0, np.pi * 5, num_of_training_data + num_of_prediction_data)
    
    return np.sin(x) 