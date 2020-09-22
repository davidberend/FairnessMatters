import numpy as np
import cv2
import random
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2, os, argparse
from PIL import Image
from collections import defaultdict
import copy
import random
import json
import time
import math
import sys
sys.path.append('../')
print(sys.path)
from get_raw_data import getUTKdata, getMORPHdata, getAPPAdata, getMegaasianData, getFGNETdata, getIMDB, getWIKI
from utils.data_utils import getMinMaxSample, update

def getBalancedData(data_folder):
    dataset_names = ['UTKdata','Megaasian','APPA','MORPH']
    # Get all four datasets
    alldatasets = {
        'UTKdata':getUTKdata(data_folder),
        'Megaasian': getMegaasianData(data_folder),
        'APPA': getAPPAdata(data_folder),
        'MORPH': getMORPHdata(data_folder)
    }
    single_num_samples = {
        'caucasian':{i:0 for i in dataset_names},
        'afroamerican':{i:0 for i in dataset_names},
        'asian':{i:0 for i in dataset_names}
    }
    dataset_samples = {
        i:copy.deepcopy(single_num_samples) for i in range(1,101)
    }
    # num = dataset_samples[2]['caucasian']['UTKdata']
    single_samples = {
        'caucasian':defaultdict(list),
        'afroamerican':defaultdict(list),
        'asian':defaultdict(list)
    }

    all_samples = {
        'UTKdata':copy.deepcopy(single_samples),
        'Megaasian': copy.deepcopy(single_samples),
        'APPA': copy.deepcopy(single_samples),
        'MORPH': copy.deepcopy(single_samples)
    }

    # Number of samples for each ethnicity in each age
    num_sample = {
        'caucasian':{i:0 for i in range(1,101)},
        'afroamerican':{i:0 for i in range(1,101)},
        'asian':{i:0 for i in range(1,101)}
    }

    for dataset in alldatasets:
        for samples in tqdm(alldatasets[dataset]):
            if 1<=samples['age']<=100 and samples['race'] in ['caucasian','afroamerican','asian']:
                all_samples[dataset][samples['race']][samples['age']].append([samples['image_path'],samples['race'],samples['age']-1])
                dataset_samples[samples['age']][samples['race']][dataset]+=1
                num_sample[samples['race']][samples['age']]+=1

    for key in num_sample:
        samples = copy.deepcopy(num_sample[key])
        num_sample[key] = dict(sorted(samples.items(), key=lambda samples:samples[1]))

    min_sample , max_sample = getMinMaxSample(num_sample)
    # max_sample = 1000
    balancedTrainData = []
    balancedTestData = []
    train_data_num = {
        'caucasian':{i:0 for i in range(1,101)},
        'afroamerican':{i:0 for i in range(1,101)},
        'asian':{i:0 for i in range(1,101)}
    }
    for age in range(1,101):
        threshold = np.inf
        for race in num_sample:
            threshold = min(threshold,num_sample[race][age])
        threshold = int(min(max_sample,max(min_sample,threshold)))
        ds_num = len(alldatasets)
        select_size = math.ceil(threshold*1.0/ds_num)
        print(threshold, select_size,max_sample,min_sample)
        for race in num_sample:
            race_threshold = threshold
            race_select_size = select_size
            ds_num = len(alldatasets)
            race_num_sample = copy.deepcopy(dataset_samples[age][race])
            dataset_samples[age][race] = dict(sorted(race_num_sample.items(), key=lambda race_num_sample:race_num_sample[1]))
            for dataset in dataset_samples[age][race]:
                num = dataset_samples[age][race][dataset]
                if num < race_select_size:
                    train_size = math.ceil(num*0.8)
                    for index in range(train_size) :
                        balancedTrainData.append(all_samples[dataset][race][age][index])
                        train_data_num[race][age]+=1
                    for index in range(train_size,num):
                        balancedTestData.append(all_samples[dataset][race][age][index])
                    race_select_size, race_threshold, ds_num = update(race_select_size, race_threshold, num, ds_num)
                else:
                    indices = np.random.choice(len(all_samples[dataset][race][age]),race_select_size,replace=False)
                    train_size = math.floor(race_select_size*0.8)
                    ds_num-=1
                    for index in range(train_size):
                        balancedTrainData.append(all_samples[dataset][race][age][indices[index]])
                        train_data_num[race][age]+=1
                    for index in range(train_size,len(indices)):
                        balancedTestData.append(all_samples[dataset][race][age][indices[index]])
    
    balancedTestData = pd.DataFrame(balancedTestData)
    balancedTrainData = pd.DataFrame(balancedTrainData)

    balancedTestData.to_csv('./data/test.tsv',header=None, index=None,sep='\t')
    balancedTrainData.to_csv('./data/train.tsv',header=None, index=None,sep='\t')

    return 


def main(data_folder):
    getBalancedData(data_folder)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='/mnt/nvme/aibias/OriDatasets/')
    parser.add_argument('-train_save_path', type=str)
    parser.add_argument('-test_save_path', type=str)
    args = parser.parse_args()
    data_folder = args.dir
    train_save_path = args.train_save_path
    test_save_path = args.test_save_path
    main(data_folder)
