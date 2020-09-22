import os
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from ood.OOD_techniques.glod import retrieve_scores, ConvertToGlod, calc_gaussian_params
from models.Generalmodels import create_Resnet
import utils.data_utils

DarkBlue= '#051c2c'
LightBlue= '#3ba9f5'
LightGrey='#989898'
Blue= '#2140e6'
BrightBlue='#aae6f0'
Orange='#FF7828'
Green= '#009926'

def visualize_distribution(in_dist, out_dist, k):

    in_dist = in_dist.detach().cpu().numpy()
    out_dist = out_dist.detach().cpu().numpy()

    plt.hist(in_dist, bins=300, rwidth=0.8,  color=LightBlue,alpha=0.5)
    plt.hist(out_dist, bins=300, rwidth=0.8, color=Orange,alpha=0.5)
    plt.xlim((-1000,1000))
    plt.show()

def visualize_distribution_race(dists, k, out=False):
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

def Convert(train_path, batch_size, weight_path, device, transform, version=50, num_classes=100, img_pixels= (224, 224)):
    # data
    
    train_set_X, train_set_y, _ = utils.data_utils.process_data(train_path)

    print('Using data:{}'.format(train_path))
    train_loader = utils.data_utils.make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels,
                                                batch_size=batch_size, transform_test=transform, shuffle=True)

    # Creat the original model
    print('Creating model using model :{}'.format(weight_path))
    model = create_Resnet(resnet_number=version, num_classes=num_classes,
                        gaussian_layer=False, weight_path=weight_path,
                        device=device)

    # Convert it to glod model
    print('Begin converting')
    model = ConvertToGlod(model, num_classes=num_classes)
    print('Begin calculating')
    covs, centers = calc_gaussian_params(model, train_loader, device, num_classes)
    print('Done Calculation')
    model.gaussian_layer.covs.data = covs
    model.gaussian_layer.centers.data = centers

    return model

def ConvertAndGetscores(train_path, aug_loader, batch_size, weight_path, num_classes, glod_k = 100, transform=None):
    
    device='cuda:0'
    device = torch.device(device)
    
    # Load original model
    model = create_Resnet(resnet_number=50, num_classes=num_classes,
                      gaussian_layer=False, weight_path=weight_path,
                      device=device)

    # Convert to Glod
    glod_model = Convert(train_path, batch_size, weight_path, device,transform)
    # torch.save(glod_model.state_dict(),'glod_model.pt')
    # Retrive scores
    aug_scores = retrieve_scores(glod_model, aug_loader, device, glod_k)
    # aug_scores = torch.load('/home/david/bias_ai_glod/SweetGAN/OOD_scores/FineTuneData_out_scores_augextreme.txt_preds.pt')
    # Sample data
    # take_samples(out_scores, out_set_X,out_set_y,out_race,quantile=quantile)
    k=100
    top_k = aug_scores.topk(k).values.squeeze()
    avg_ll = np.mean(top_k[:, 1:k].cpu().detach().numpy())
    llr = top_k[:, 0].cpu()-avg_ll
    return llr

def split_data(scores, dataset,label_age,label_race, quantile=0.03):
    threshold=np.quantile(scores,quantile)
    diff = (scores-threshold)
    good_indices=np.where(diff>0)[0]
    splited_data = []
    for index in good_indices:
        splited_data.append([dataset[index],label_race[index],label_age[index]])
    return splited_data