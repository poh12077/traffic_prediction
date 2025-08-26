import csv

def get_data(file, skip_row=0, column=0):
    with open(file, 'r', encoding='cp949') as f:
        csv_reader = csv.reader(f)
        data = []
        idx = 0
        for row in csv_reader: 
            if idx >= skip_row:
                data.append(float(row[column]))
            idx +=1
    return data

def split_data(data, n=1):
    k = int(len(data) / n)
    return data[:k]
  
def fill_data(data, how_many_times=1):
    for n in range(how_many_times):
        smoothed_data = []
        for i in data:
            smoothed_data.append(i)
            smoothed_data.append(0)
        smoothed_data.pop()

        for k, v in enumerate(data):
            if k+1 == len(data):
                break
            mean = (float(data[k]) + float(data[k+1])) / 2    
            smoothed_data[2*k+1] = mean
        data = smoothed_data

    return data

def get_sample(data, sampling=1):
    value_list = []
    for k, v in enumerate(data):
        if (k % sampling) == 0:
            value_list.append(v)
    return value_list

def shrink_min_max(data, iteration=1):
    for i in range(0, iteration):
        max_value = max(data)
        min_value = min(data)
        center_value = (max_value + min_value) / 2 
        max_limit = (max_value + center_value)/2
        min_limit = (min_value + center_value)/2            

        key_list = []
        for k, v in enumerate(data):
            if v > max_limit:
                key_list.append(k)
            elif v < min_limit:
                key_list.append(k)
        
        key_list.reverse()
        for k in key_list:
            data.pop(k)
    return data


