from PIL import Image
import torchvision.transforms as trn
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
import json
import numpy as np
import math
# from autoaugment import ImageNetPolicy

# Define random_augmentation function
def random_augmentation(im, brightness=0, contrast=0, saturation=0, hue=0,
                        erase_p=0.5, erase_scale=(0.02, 0.33),
                        erase_ratio=(0.3, 3.3), erase_value=0,
                        erase_inplace=False, degrees=0, translate=None,
                        scale=None, shear=None, resample=False, fillcolor=0,
                        h_flip=0.0):

    color_jitter = trn.ColorJitter(
        brightness=brightness, contrast=contrast, saturation=saturation,
        hue=hue)

    random_affine = trn.RandomAffine(
        degrees, translate=translate, scale=scale, shear=shear,
        resample=resample, fillcolor=fillcolor)
    transform = trn.Compose([trn.RandomHorizontalFlip(h_flip),
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
            aug_data.write('{}\t{}\t{}\t{}\n'.format(sample[0]+'aug_{}.jpg'.format(i),sample[1],sample[2],sample[3]))

def data_augmentation(data_path=None):
    ## Loading data
    data_path = 'train.csv'
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

    aug_data=open('train_aug_ori.txt','w')
    print('saving to {}'.format(aug_data))
    for race in total_data:
        for age in (range(100)):
            num = len(total_data[race][age])
            if num==0:continue
            aug_ratio = math.ceil(max_num/num)
            aug_ratio = min(aug_ratio,max_ratio)
            batch_augmentation(race,age, aug_ratio, total_data, aug_data)
    aug_data.close()

def balancing_augmented_data():
    

data_augmentation()