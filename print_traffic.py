import pickle
import os

traffic_file_path = "traffic.pickle"


def load_traffic_file():
    with open(traffic_file_path, "rb") as f:
        traffic = pickle.load(f) 
    
    return traffic

traffic = load_traffic_file()
        
print(traffic)