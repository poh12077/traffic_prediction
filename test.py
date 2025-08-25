import numpy as np
from sklearn.preprocessing import MinMaxScaler

import csv

with open("weatherHistory.csv", 'r', encoding='cp949' ) as f:
    csv_reader = csv.reader(f)
    data = []
    flag = 0
    for row in csv_reader: 
        if flag > 0: 
            data.append(float(row[7]))
        flag +=1    
    

    



# Original traffic data (1D)
traffic = np.array([1, 2, 3])
traffic_ = np.array(traffic)

# Reshape to 2D (3 samples, 1 feature)
traffic = traffic.reshape(-1, 1)

# Create scaler (default range = 0 to 1)
scaler = MinMaxScaler()

# Fit and transform
traffic_scaled = scaler.fit_transform(traffic)

print("Original:", traffic.flatten())
print("Normalized:", traffic_scaled.flatten())

# Convert back (inverse transform)
traffic_original = scaler.inverse_transform(traffic_scaled)
print("Back to original:", traffic_original.flatten())
