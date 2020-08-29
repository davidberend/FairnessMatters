import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt
from PIL import Image

DarkBlue= '#051c2c'
LightBlue= '#3ba9f5'
LightGrey='#989898'
Blue= '#2140e6'
BrightBlue='#aae6f0'
Orange='#FF7828'
Green= '#009926'

def visualize_distribution(in_dist, out_dist, model_type, data_type, k):

    in_dist = in_dist.detach().cpu().numpy()
    out_dist = out_dist.detach().cpu().numpy()

    plt.hist(in_dist, bins=300, rwidth=0.8,  color=LightBlue,alpha=0.5)
    plt.hist(out_dist, bins=300, rwidth=0.8, color=Orange,alpha=0.5)
    plt.xlim((-1000,1000))
    plt.show()

def visualize_distribution_race(dists, model_type, data_type, k, out=False):
    colors={
        'asian':DarkBlue,
        'caucasian':LightBlue,
        'afroamerican':LightGrey,
    }
    name_maps={
    'asian':'Asian',
    'caucasian':'Caucasian',
    'afroamerican':'Afro-American',
    }
    for key in name_maps:
        if not out:
            plt.hist(dists[key], bins=300, rwidth=0.8,  color=colors[key],label=name_maps[key],alpha=0.5)
        if out:
            sns.distplot(dists[key], kde_kws={"label": "OOD: "+key})
    plt.show()

def take_samples(scores, dataset,label_age,label_race, quantile=0.03):
    # print(threshold)
    threshold=np.quantile(scores,quantile)
    diff = (scores-threshold)
    bad_indices=np.where(diff<=0)[0]
    good_indices=np.where(diff>0)[0]
    print(len(good_indices),len(bad_indices))
    # bad=open('/home/david/aibias/datasets/1_101_1yr/OOD/MORPHBad_train_info_1yr_{}.txt'.format(quantile),'w')
    good=open('/home/david/aibias/datasets/1_101_1yr/OOD/AutoAll_train_info_1yr_{}.txt'.format(quantile),'w')
    # allgood=open('/home/david/aibias/datasets/1_101_1yr/all/Aug_train_info_1yr.txt','w')
    ori=open('/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr.txt','r')
    lines=ori.readlines()
    bad_counter=0
    good_counter=0

    good_indices=good_indices[:72000]
    print(len(good_indices))
    random.shuffle(bad_indices)
    random.shuffle(good_indices)
    for index in bad_indices:
        if bad_counter<15000:
            bad.write('{}\t{}\t{}\n'.format(dataset[index],label_race[index],label_age[index]))
            bad_counter+=1

    for index in good_indices:
        if good_counter<40000:
            good.write('{}\t{}\t{}\n'.format(dataset[index],label_race[index],label_age[index]))
            good_counter+=1
    
    for line in lines:
        good.write(line)
        bad.write(line)