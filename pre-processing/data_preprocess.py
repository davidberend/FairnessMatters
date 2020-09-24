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

def get_balanced_data(data_folder, train_save_path='../data/train.tsv' , test_save_path='../data/test.tsv'):
    dataset_names = ['UTKdata','Megaasian','APPA','MORPH']
    races = ['caucasian','afroamerican','asian']
    # Get all four datasets
    all_datasets = {
        'UTKdata':getUTKdata(data_folder),
        'Megaasian': getMegaasianData(data_folder),
        'APPA': getAPPAdata(data_folder),
        'MORPH': getMORPHdata(data_folder)
    }

    # Store the number of samples of each age of each race of each dataset
    # Organized as age->race->dataset
    # For sorting the datasets for choosing data
    num_samples_tmp = {
        race:{i:0 for i in dataset_names} for race in races
    }
    dataset_samples = {
        i:copy.deepcopy(num_samples_tmp) for i in range(1,101)
    }

    # Store the samples, organized by dataset->race->age
    # For sampling the balanced data
    all_samples_tmp = {
        race:defaultdict(list) for race in races
    }
    all_samples = {
        dataset:copy.deepcopy(all_samples_tmp) for dataset in dataset_names
    }

    # Number of samples for each ethnicity in each age
    # For getting the max, min and threshold
    num_sample = {
        race:{i:0 for i in range(1,101)} for race in races
    }

    # Store and organize the original data from the raw data
    for dataset in all_datasets:
        for samples in tqdm(all_datasets[dataset]):
            if 1<=samples['age']<=100 and samples['race'] in ['caucasian','afroamerican','asian']:
                all_samples[dataset][samples['race']][samples['age']].append([samples['image_path'],samples['race'],samples['age']-1])
                dataset_samples[samples['age']][samples['race']][dataset]+=1
                num_sample[samples['race']][samples['age']]+=1

    # Sort the number of samples of each race
    for key in num_sample:
        samples = copy.deepcopy(num_sample[key])
        num_sample[key] = dict(sorted(samples.items(), key=lambda samples:samples[1]))

    min_sample , max_sample = getMinMaxSample(num_sample)

    # Store the train data and test data
    balanced_train_data = []  
    balanced_test_data = []
    train_data_num = {
        race:{i:0 for i in range(1,101)} for race in races
    }

    for age in range(1,101):

        # Get threshold
        threshold = np.inf
        for race in num_sample:
            threshold = min(threshold,num_sample[race][age])
        threshold = int(min(max_sample,max(min_sample,threshold)))

        # Get select_size
        ds_num = len(all_datasets)
        select_size = math.ceil(threshold*1.0/ds_num)
        # print(threshold, select_size,max_sample,min_sample)


        for race in num_sample:
            # Copy threshold and threshold for update for this race
            race_threshold = threshold
            race_select_size = select_size
            ds_num = len(all_datasets)

            # Sort the dataset according to the number of samples about each race at this age 
            race_num_sample = copy.deepcopy(dataset_samples[age][race])
            dataset_samples[age][race] = dict(sorted(race_num_sample.items(), key=lambda race_num_sample:race_num_sample[1]))

            # Begin sampling data
            for dataset in dataset_samples[age][race]:

                # get the number of the samples of this dataset in this age and this race
                num = dataset_samples[age][race][dataset]
                if num < race_select_size:
                    train_size = math.ceil(num*0.8)

                    # Sampling train data
                    for index in range(train_size) :
                        balanced_train_data.append(all_samples[dataset][race][age][index])
                        train_data_num[race][age]+=1
                    # Sampling test data
                    for index in range(train_size,num):
                        balanced_test_data.append(all_samples[dataset][race][age][index])
                    race_select_size, race_threshold, ds_num = update(race_select_size, race_threshold, num, ds_num)
                else:
                    
                    # random sample from dataset
                    indices = np.random.choice(len(all_samples[dataset][race][age]),race_select_size,replace=False)
                    train_size = math.floor(race_select_size*0.8)
                    ds_num-=1
                    for index in range(train_size):
                        balanced_train_data.append(all_samples[dataset][race][age][indices[index]])
                        train_data_num[race][age]+=1
                    for index in range(train_size,len(indices)):
                        balanced_test_data.append(all_samples[dataset][race][age][indices[index]])
    
    balanced_test_data = pd.DataFrame(balanced_test_data)
    balanced_train_data = pd.DataFrame(balanced_train_data)

    balanced_test_data.to_csv(test_save_path,header=None, index=None,sep='\t')
    balanced_train_data.to_csv(train_save_path,header=None, index=None,sep='\t')
    print(train_data_num)
    return 
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='/mnt/nvme/aibias/OriDatasets/')
    parser.add_argument('-train_save_path', type=str, default='../data/train.tsv')
    parser.add_argument('-test_save_path', type=str, default='../data/test.tsv')
    args = parser.parse_args()
    data_folder = args.dir
    train_save_path = args.train_save_path
    test_save_path = args.test_save_path
    get_balanced_data(data_folder, train_save_path, test_save_path)
