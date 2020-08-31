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

def RaceSpecificData():
    f=open('/home/david/aibias/datasets/alldata.json','r')
    alldatasets=json.load(f)
    allraces={
        'caucasian':[],
        'afroamerican':[],
        'asian':[]
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
                allraces[samples['race']].append({'image_path':samples['image_path'],'age':int(samples['age']),'gender':samples['gender'],'race':samples['race']})
                race_num[samples['race']][samples['age']]+=1
    
    print(race_num)
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
    
    train_size=int(750*0.8)
    
    for data in allraces:
        if not os.path.exists('/home/david/aibias/datasets/1_101_1yr/{}'.format(data)):
            os.makedirs('/home/david/aibias/datasets/1_101_1yr/{}'.format(data))
        train_info=open(os.path.join('/home/david/aibias/datasets/1_101_1yr/{}'.format(data),('RaceFineTuneData'+'_train_info_'+str(1)+'yr.txt')),'w')
        test_info=open(os.path.join('/home/david/aibias/datasets/1_101_1yr/{}'.format(data),('RaceFineTuneData'+'_test_info_'+str(1)+'yr.txt')),'w')
        indices=[i for i in range(len(allraces[data]))]
        random.shuffle(indices)
        for index in tqdm(indices):
            if allraces[data][index]['race'] not in ['caucasian','afroamerican','asian']:
                continue
            if allraces[data][index]['age']<1 or allraces[data][index]['age']>100:
                continue
            if train_race_num[allraces[data][index]['race']][allraces[data][index]['age']]<\
                int(race_num[allraces[data][index]['race']][allraces[data][index]['age']]*0.8):
                train_info.write('{}\t{}\t{}\t{}\n'.format(allraces[data][index]['image_path'],allraces[data][index]['race'],allraces[data][index]['age']-1,allraces[data][index]['gender']))
                train_race_num[allraces[data][index]['race']][allraces[data][index]['age']]+=1
            elif test_race_num[allraces[data][index]['race']][allraces[data][index]['age']]<\
                race_num[allraces[data][index]['race']][allraces[data][index]['age']]-int(race_num[allraces[data][index]['race']][allraces[data][index]['age']]*0.8):
                test_info.write('{}\t{}\t{}\t{}\n'.format(allraces[data][index]['image_path'],allraces[data][index]['race'],allraces[data][index]['age']-1,allraces[data][index]['gender']))
                test_race_num[allraces[data][index]['race']][allraces[data][index]['age']]+=1
    print(train_race_num,test_race_num)

def getUnbalenceddata(opt):
    f=open('/home/david/aibias/datasets/alldata.json','r')
    alldatasets=json.load(f)
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
            if 1<=samples['age']<=101 and samples['race']in ['caucasian','asian']:
                if race_num[samples['race']][samples['age']]>=1000:
                    continue
                allraces[samples['race']][samples['age']].append([samples['image_path'],samples['age'],samples['race'],samples['gender']])
                race_num[samples['race']][samples['age']]+=1
            if 1<=samples['age']<=101 and samples['race']=='afroamerican':
                if race_num[samples['race']][samples['age']]>=200:
                    continue
                allraces[samples['race']][samples['age']].append([samples['image_path'],samples['age'],samples['race'],samples['gender']])
                race_num[samples['race']][samples['age']]+=1
    for key in race_num['caucasian']:
        race_num['asian'][key]=min(race_num['asian'][key],race_num['caucasian'][key])
        race_num['caucasian'][key]=race_num['asian'][key]
    for key in race_num['caucasian']:
        print(race_num['asian'][key],race_num['caucasian'][key])
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
    if not os.path.exists('/home/david/aibias/datasets/1_101_1yr/Unbalanced'):
        os.makedirs('/home/david/aibias/datasets/1_101_1yr/Unbalanced')
    train_info=open(os.path.join('/home/david/aibias/datasets/1_101_1yr/Unbalanced',('FineTuneData'+'_train_info_'+str(1)+'yr.txt')),'w')
    test_info=open(os.path.join('/home/david/aibias/datasets/1_101_1yr/Unbalanced',('FineTuneData'+'_test_info_'+str(1)+'yr.txt')),'w')
    
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
    print(train_race_num.items(),test_race_num.items())


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