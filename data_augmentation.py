from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import math
from utils import data_utils
import argparse
import sys
import copy
from ood.OOD_techniques.glod import retrieve_scores, ConvertToGlod, calc_gaussian_params
from models.Generalmodels import create_Resnet
from utils.OOD_utils import visualize_distribution, take_samples, Convert, ConvertAndGetscores, split_data
from utils.data_utils import getMinMaxSample, update
# from autoaugment import ImageNetPolicy

def random_augmentation(im, brightness=0, contrast=0, saturation=0, hue=0,
                        erase_p=0.5, erase_scale=(0.02, 0.33),
                        erase_ratio=(0.3, 3.3), erase_value=0,
                        erase_inplace=False, degrees=0, translate=None,
                        scale=None, shear=None, resample=False, fillcolor=0,
                        h_flip=0.0):

    color_jitter = transforms.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation,
        hue=hue)

    random_affine = transforms.RandomAffine(
        degrees, translate=translate, scale=scale, shear=shear,
        resample=resample, fillcolor=fillcolor)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(h_flip),
                             color_jitter,
                             random_affine,
                             ])
    return transform(im)

def batch_augmentation(race,age,aug_ratio, total_data, aug_data):

    # ---------------------
    # RANDOM SETTINGS
    # ---------------------
    fillcolor = 0  # 255
    brightness = (0.5, 2.0)
    contrast = (0.5, 2.0)
    saturation = (0.7, 1.8)
    hue = (-0.08, 0.08)

    degrees = 45
    translate = (0.25, 0.25)  # (0.2, 0.4)
    scale = (0.6, 1.8)  # (0.5, 2.0)
    shear = 30  # 30
    h_flip = 0.5

    for i in range(aug_ratio):
        for sample in total_data[race][age]:
            img = Image.open(sample[0])
            if img.mode=='L':
                img=img.convert("RGB")
            # policy = ImageNetPolicy()
            # img = policy(img)
            img=random_augmentation(img,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, erase_p=0.0,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            fillcolor=fillcolor,
            h_flip=h_flip)
            # img.save(sample[0]+'aug_{}.jpg'.format(i))
            # aug_datas.append([sample[0]+'aug_{}.jpg'.format(i),sample[1],sample[2],sample[3]])
            aug_data.write('{}\t{}\t{}\t{}\n'.format(sample[0]+'aug_{}.jpg'.format(i),sample[1],sample[2],sample[3]))

def data_augmentation(data_path=None):
    ## Loading data
    data_path = './data/train.tsv'
    f=open(data_path,'r')
    alldatasets=f.readlines()
    total_data={
            'caucasian':defaultdict(list),
            'afroamerican':defaultdict(list),
            'asian':defaultdict(list),
    }
    race_num={
            'caucasian':defaultdict(int),
            'afroamerican':defaultdict(int),
            'asian':defaultdict(int),
    }
    # Load data according to age and race
    for data in alldatasets:
        data=data.strip()
        try:
            total_data[data.split('\t')[1]][int(data.split('\t')[2])].append(data.split('\t'))
            race_num[data.split('\t')[1]][int(data.split('\t')[2])]+=1
        except:
            continue

    all_num=[]
    for key in race_num:
        all_num.extend(list(race_num[key].values()))
    all_num = np.array(all_num)

    median_num = np.median(all_num)
    mean_num = np.mean(all_num)
    max_num = np.max(all_num)
    max_ratio = math.ceil(median_num/mean_num)

    aug_data_file=open('./data/train_aug_ori.tsv','w')
    print('saving to {}'.format(aug_data_file))
    for race in total_data:
        for age in (range(100)):
            num = len(total_data[race][age])
            if num==0:continue
            aug_ratio = math.ceil(max_num/num)
            aug_ratio = min(aug_ratio,max_ratio)
            batch_augmentation(race,age, aug_ratio, total_data, aug_data_file)
    aug_data_file.close()
    return 

def getSplitData(aug_data_path):
    device = torch.device('cuda')
    img_pixels = (224,224)
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.550, 0.500, 0.483], [0.273, 0.265, 0.266])])
    # Load data
    aug_data_X, aug_data_age, _ ,aug_data_race= data_utils.process_data_ood(aug_data_path)
    batch_size = 256
    aug_loader = data_utils.make_dataloader_iter(aug_data_X, aug_data_age, img_size=img_pixels,
                                             batch_size=batch_size, transform_test=transform, shuffle=False)
    glod_k=20
    quantile = 0.05
    scores = ConvertAndGetscores(train_path, aug_loader, batch_size, weight_path, num_classes=100, glod_k = glod_k, transform=transform)

    splited_aug_data = split_data(scores, aug_data_X ,aug_data_age, aug_data_race, quantile=0.05)
    return splited_aug_data

def balancing_augmented_data():
    splited_data = getSplitData(aug_data_path='./data/train_aug_ori.tsv')
    print(len(splited_data))

    all_samples = {
        'caucasian':defaultdict(list),
        'afroamerican':defaultdict(list),
        'asian':defaultdict(list)
    }

    single_num_samples = {
        'caucasian':0,
        'afroamerican':0,
        'asian':0
    }

    age_num_samples = {
        i:copy.deepcopy(single_num_samples) for i in range(0,100)
    }

    race_num_samples = {
        'caucasian':{i:0 for i in range(0,100)},
        'afroamerican':{i:0 for i in range(0,100)},
        'asian':{i:0 for i in range(0,100)}
    }

    for samples in splited_data:
        all_samples[samples[1]][samples[2]].append(samples)
        race_num_samples[samples[1]][samples[2]]+=1
        age_num_samples[samples[2]][samples[1]]+=1

    balanced_splited_aug_data =[]

    train_data_num = {
        'caucasian':{i:0 for i in range(0,100)},
        'afroamerican':{i:0 for i in range(0,100)},
        'asian':{i:0 for i in range(0,100)}
    }

    min_sample, max_sample = getMinMaxSample(race_num_samples)
    print(min_sample, max_sample)
    for age in range(0,100):
        threshold = np.inf
        for race in race_num_samples:
            threshold = min(threshold,age_num_samples[age][race])
        threshold = int(min(max_sample,max(min_sample,threshold)))

        race_num = len(race_num_samples)

        select_size = math.ceil(threshold/race_num)
        samples = copy.deepcopy(age_num_samples[age])
        age_num_samples[age] = dict(sorted(samples.items(), key=lambda samples:samples[1]))

        for race in age_num_samples[age]:
            num = age_num_samples[age][race]
            print(age,race,num,len(all_samples[race][age]),select_size)
            if num <select_size :
                for index in range(num):
                    balanced_splited_aug_data.append(all_samples[race][age][index])
                    train_data_num[race][age]+=1
                select_size, threshold, race_num = update(select_size, threshold, num, race_num)
            else:
                indices = np.random.choice(len(all_samples[race][age]),select_size,replace=False)
                race_num-=1
                for index in indices:
                    balanced_splited_aug_data.append(all_samples[race][age][index])
                    train_data_num[race][age]+=1
    ori_train_data = pd.read_csv('./data/train.tsv',header=None,sep='\t')
    balanced_splited_aug_data = pd.DataFrame(balanced_splited_aug_data)
    new = pd.concat([ori_train_data,balanced_splited_aug_data])
    new.to_csv('./data/balanced_aug_data.tsv',header=None, index=None,sep='\t')


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-train_path', help='The dataset which used to train the original age prediction model',
    default='/home/david/bias_ai_glod/SweetGAN/data/Balanced/FineTuneData_train_info_1yr.txt')
    parser.add_argument('-model_path', help='The path of the trained age prediction model',
    default = '/home/david/aibias/model_weights/regression/both_resnet50_FineTuneData_adam_0.0001_100/resnet50_FineTuneData_epoch_122_0.3150045821877864.pt')
    parser.add_argument('-in_path', help='In distribution dataset',
    default='/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr.txt')
    parser.add_argument('-out_path', help='Out of distribution dataset',
    default='/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr_augextreme.txt')
    parser.add_argument('-batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('-glod_k', type=int, help='glod_k value', default=100)
    parser.add_argument('-quantile', type=int, help='quantile', default=0.05)
    parser.add_argument('-num_classes', type=int, help='number of calss', default=100)

    
    args = parser.parse_args()

    batch_size = args.batch_size
    train_path = args.train_path
    weight_path = args.model_path
    in_path = args.in_path
    out_path = args.out_path
    glod_k = args.glod_k
    quantile = args.quantile
    num_classes = args.num_classes
    
    # data_augmentation()

    # ConvertAndGetscores(train_path, batch_size, weight_path, num_classes, glod_k = glod_k, quantile=quantile)
    balancing_augmented_data()