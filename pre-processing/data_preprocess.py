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

from get_raw_data import getUTKdata, getMORPHdata, getAPPAdata, getMegaasianData, getFGNETdata, getIMDB, getWIKI
 
def getAlldata(data_folder, train_save_path, test_save_path):

    alldatasets = {
        'UTKdata':getUTKdata(data_folder),
        'Megaasian': getMegaasianData(data_folder),
        'APPA': getAPPAdata(data_folder),
        'MORPH': getMORPHdata(data_folder)
        }

    allraces={
        'caucasian':defaultdict(list),
        'afroamerican':defaultdict(list),
        'asian':defaultdict(list)
    }
    age_num=defaultdict(int)
    race_num={
        'caucasian':copy.deepcopy(age_num),
        'afroamerican':copy.deepcopy(age_num),
        'asian':copy.deepcopy(age_num)
    }
    for data in alldatasets:
        for samples in tqdm(alldatasets[data]):
            if 1<=samples['age']<=101 and samples['race']in ['caucasian','afroamerican','asian']:
                if race_num[samples['race']][samples['age']]>=1000:
                    continue
                allraces[samples['race']][samples['age']].append([samples['image_path'],samples['age'],samples['race'],samples['gender']])
                race_num[samples['race']][samples['age']]+=1

    train_race_num={
        'caucasian':copy.deepcopy(age_num),
        'afroamerican':copy.deepcopy(age_num),
        'asian':copy.deepcopy(age_num)
    }
    test_race_num={
        'caucasian':copy.deepcopy(age_num),
        'afroamerican':copy.deepcopy(age_num),
        'asian':copy.deepcopy(age_num)
    }
    train_info=open(train_save_path,'w')
    test_info=open(test_save_path,'w')
    train_size=int(750*0.8)
    
    for data in alldatasets:
        indices=[i for i in range(len(alldatasets[data]))]
        random.shuffle(indices)
        for index in tqdm(indices):
            if alldatasets[data][index]['race'] not in ['caucasian','afroamerican','asian']:
                continue
            if alldatasets[data][index]['age']<1 or alldatasets[data][index]['age']>101:
                continue
            if train_race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]<int(race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]*0.8):
                if alldatasets[data][index]['age']>=98:
                    train_info.write('{}\t{}\t{}\t{}\n'.format(alldatasets[data][index]['image_path'],alldatasets[data][index]['race'],alldatasets[data][index]['age']-2,alldatasets[data][index]['gender']))
                else:
                    train_info.write('{}\t{}\t{}\t{}\n'.format(alldatasets[data][index]['image_path'],alldatasets[data][index]['race'],alldatasets[data][index]['age']-1,alldatasets[data][index]['gender']))
                train_race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]+=1
            elif test_race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]<race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]-int(race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]*0.8):
                if alldatasets[data][index]['age']>=98:
                    test_info.write('{}\t{}\t{}\t{}\n'.format(alldatasets[data][index]['image_path'],alldatasets[data][index]['race'],alldatasets[data][index]['age']-2,alldatasets[data][index]['gender']))
                else:
                    test_info.write('{}\t{}\t{}\t{}\n'.format(alldatasets[data][index]['image_path'],alldatasets[data][index]['race'],alldatasets[data][index]['age']-1,alldatasets[data][index]['gender']))
                test_race_num[alldatasets[data][index]['race']][alldatasets[data][index]['age']]+=1
    print(train_race_num,test_race_num)

def getMinMaxSample(num_sample):
    max_sample = np.inf
    min_sample = 0
    for key in num_sample:
        max_sample = min(max_sample,np.quantile(list(num_sample[key].values()),0.8))
        min_sample = max(min_sample,np.quantile(list(num_sample[key].values()),0.2))
    return min_sample, max_sample

def update(select_size, threshold, num,ds_num):
    threshold -= num
    ds_num -= 1
    if ds_num!=0:
        select_size = math.ceil(threshold*1.0/ds_num)
    else:
        select_size = threshold
    return select_size, threshold, ds_num

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
                all_samples[dataset][samples['race']][samples['age']].append([samples['image_path'],samples['race'],samples['age']-1,samples['gender']])
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

    balancedTestData.to_csv('test.csv',header=None, index=None,sep='\t')
    balancedTrainData.to_csv('train.csv',header=None, index=None,sep='\t')

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
