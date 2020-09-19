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
    max_sample = 0
    min_sample = np.inf
    for key in num_sample:
        max_sample = max(max_sample,np.quantile(num_sample[key].values(),0.8))
        min_sample = min(min_sample,np.quantile(num_sample[key].values(),0.2))
    return min_sample, max_sample

def getBalancedData():
    
    # Get all four datasets
    alldatasets = {
    'UTKdata':getUTKdata(data_folder),
    'Megaasian': getMegaasianData(data_folder),
    'APPA': getAPPAdata(data_folder),
    'MORPH': getMORPHdata(data_folder)
    }

    datasets = {
        'caucasian':defaultdict(list),
        'afroamerican':defaultdict(list),
        'asian':defaultdict(list)
    }

    # Number of samples for each ethnicity in each age
    num_sample = {
        'caucasian':defaultdict(int),
        'afroamerican':defaultdict(int),
        'asian':defaultdict(int)
    }

    for data in alldatasets:
        for samples in tqdm(alldatasets[data]):
            if 1<=samples['age']<=101 and samples['race'] in ['caucasian','afroamerican','asian']:
                datasets[samples['race']][samples['age']].append([samples['image_path'],samples['age'],samples['race'],samples['gender']])
                num_sample[samples['race']][samples['age']]+=1

    for key in num_sample:
        samples = copy.deepcopy(num_sample[key])
        num_sample[key] = dict(sorted(samples.items(), key=lambda samples:samples[1]))

    min_threshold,max_threshold = getMinMaxSample(num_sample)

    # for age in 
    return 


def main(data_folder):
    getAlldata(data_folder)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='/mnt/nvme/aibias/OriDatasets/')
    parser.add_argument('-train_save_path', type=str)
    parser.add_argument('-test_save_path', type=str)
    args = parser.parse_args()
    data_folder = args.dir
    train_save_path = args.train_save_path
    test_save_path = args.test_save_path
    main(data_folder, train_save_path, test_save_path)
