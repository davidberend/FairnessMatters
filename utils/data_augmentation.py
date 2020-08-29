from PIL import Image
import torchvision.transforms as trn
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from tqdm import tqdm
import json
from autoaugment import ImageNetPolicy

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


## Loading data
data_path = 'datasets/1_101_1yr/all/FineTuneData_train_info_1yr.txt'

f=open(data_path,'r')
alldatasets=f.readlines()
total_data={
        'caucasian':defaultdict(list),
        'afroamerican':defaultdict(list),
        'asian':defaultdict(list),
        'hispanic':defaultdict(list)
}
race_num={
        'caucasian':defaultdict(int),
        'afroamerican':defaultdict(int),
        'asian':defaultdict(int),
        'hispanic':defaultdict(int)
}

# Load data according to age and race
for data in alldatasets:
    data=data.strip()
    try:
        total_data[data.split('\t')[1]][int(data.split('\t')[2])].append(data.split('\t'))
        race_num[data.split('\t')[1]][int(data.split('\t')[2])]+=1
    except:
        continue
    
# check for max_samples_of_age in data
max_samples_of_age = 800

max_augmentation_of_sample = 10

min_augmentation_of_sample = 1

max = 0
for race in total_data:
    for age in range(100):
        age_data = total_data[race][age]
        if len(age_data) > max: max == len(age_data)
if max_samples_of_age < max_samples_of_age: max_samples_of_age = max


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

# GENERATE
aug_data=open('/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr_augextreme.txt','w')
print('saving to {}'.format(aug_data))
for race in total_data:
    for age in (range(100)):
        augmentation_ratio = 0
        no_samples_of_age = len(total_data[race][age])
        if no_samples_of_age==0:continue
        augmentation_ratio = max(int(max_samples_of_age/no_samples_of_age),1)
        if augmentation_ratio > max_augmentation_of_sample: augmentation_ratio=max_augmentation_of_sample
        aug_num=0
        augmentation_ratio*=3
        for i in range(augmentation_ratio):
            for sample in total_data[race][age]:
                aug_num+=1
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
                img.save(sample[0]+'aug_{}.jpg'.format(i))
                aug_data.write('{}\t{}\t{}\t{}\n'.format(sample[0]+'aug_{}.jpg'.format(i),sample[1],sample[2],sample[3]))
        print(race,age,no_samples_of_age,aug_num)
