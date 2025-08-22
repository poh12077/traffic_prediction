import time
import pickle
import os

my_interface_name = "docker0"
proc_path = "/proc/net/dev"
traffic_file_path = "traffic.pickle"

MAX_NUM_OF_DATA = 1500
MOVING_AVG_LEN = 5

def get_inbps():
    bytes = get_byte()
    time.sleep(1)
    return abs(bytes - get_byte())

def get_byte():
    f = open(proc_path,"r")
    while True:
        line = f.readline()
        line_split = line.split() 
        interface = line_split[0]
        if interface == (my_interface_name  + ":"):
            inbps = line_split[1]
            break
    f.close()
    #return [int(time.time()), int(inbps)]
    return int(inbps)

def load_traffic_file():
    with open(traffic_file_path, "rb") as f:
        traffic = pickle.load(f) 
        
    return traffic

def get_moving_avg(traffic, inbps, num):
    if len(traffic) < num:
        return inbps
    if num < 2:
        return inbps
    
    sum = inbps
    sum += traffic[-1]
    if num > 2:
        for i in range( -(num-1), -1):
            sum += traffic[i]

    return int( sum / num )

def dump_traffic_file(traffic):
    inbps = get_inbps()
    inbps_avg = get_moving_avg(traffic, inbps, MOVING_AVG_LEN)
    traffic.append(inbps_avg)
    with open(traffic_file_path, "wb") as f:       
        pickle.dump(traffic, f)

def init_traffic_file():
    if not os.path.exists(traffic_file_path):
        with open(traffic_file_path, "wb") as f:
            pickle.dump([],f)

def init_traffic_file_():
    with open(traffic_file_path, "wb") as f:
        pickle.dump([],f)


init_traffic_file()
#init_traffic_file_()
while True:    
    traffic = load_traffic_file()
    if len(traffic) > MAX_NUM_OF_DATA:
            traffic.pop(0)
    dump_traffic_file(traffic)
