import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
file_path = 'seed_0_init.txt'
output_path = 'seed_0_scores.csv'
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from apex_oracle import apex_wrapper

def abaumannii_wrapper(seqs):
    scores = apex_wrapper(seqs)
    return -scores[:, 0]

def process_in_chunks(data, chunk_size, process_function):
    num_items = len(data)
    results = []
    for start in range(0, num_items, chunk_size):
        end = start + chunk_size
        chunk = data[start:end]
        result = process_function(chunk)
        results.append(result)
    return np.vstack(results)

def process_in_chunks_single_bacteria(data, chunk_size, process_function):
    num_items = len(data)
    results = []
    for start in tqdm(range(0, num_items, chunk_size)):
        end = start + chunk_size
        chunk = data[start:end]
        result = process_function(chunk)
        results.append(result)
    return np.hstack(results)
x_list = pd.read_csv(file_path, header=None, skip_blank_lines=False)[0].to_list()
x_list = ['' if x != x else x for x in x_list]
for i in range(1, len(x_list)):
    if i % 1000 != 0:
        if x_list[i] == '':
            x_list[i] = x_list[i - 1]
    elif x_list[i] == '':
        x_list[i] = x_list[i + 1]
for i in range(len(x_list)):
    if not isinstance(x_list[i], str):
        print(i, x_list[i])
chunk_size = 10000
scores = process_in_chunks_single_bacteria(x_list, chunk_size, abaumannii_wrapper)
np.savetxt(output_path, scores, delimiter=',', fmt='%.8f')
