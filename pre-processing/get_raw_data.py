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

def getWIKI(wikiinfo,data_folder):
    allinformation=[]
    for i in range(len(wikiinfo['wiki'][0][0][3][0])):
        if int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[2].split('.')[0])-\
        int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[1].split('-')[0])<0 or\
        int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[2].split('.')[0])-\
        int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[1].split('-')[0])>100:
            continue
        allinformation.append(
        {
        'image_path':data_folder+'/wiki_crop/'+wikiinfo['wiki'][0][0][2][0][i][0],
        'age':int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[2].split('.')[0])-int(wikiinfo['wiki'][0][0][2][0][i][0].split('_')[1].split('-')[0]),
        'gender':wikiinfo['wiki'][0][0][3][0][i]
        })
    return allinformation

def getIMDB(imdbinfo,data_folder):
    allinformation=[]
    for i in range(len(imdbinfo['imdb'][0][0][3][0])):
        if int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[3].split('.')[0])-\
        int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[2].split('-')[0])<0 or\
        int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[3].split('.')[0])-\
        int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[2].split('-')[0])>100:
            continue
        allinformation.append(
        {
        'image_path':data_folder+'/imdb_crop/'+imdbinfo['imdb'][0][0][2][0][i][0],
        'age':int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[3].split('.')[0])-int(imdbinfo['imdb'][0][0][2][0][i][0].split('_')[2].split('-')[0]),
        'gender':imdbinfo['imdb'][0][0][3][0][i]
        })
    return allinformation


def getUTKdata(folder):
    maps={0:'caucasian',1:'afroamerican',2:'asian',3:'indian',4:'others'}
    map_gender={0:'male',1:'female'}
    images=os.listdir(folder+'/UTKFace')
    allinformation=[]
    for image in images:
        information=image.split('_')
        if len(information[2])!=1:
            continue
        try:
            allinformation.append(
                {
                'image_path':folder+'/UTKFace/'+image,
                'age':int(information[0]),# first term is age
                'gender':map_gender[int(information[1])],#second term is gender
                'race':maps[int(information[2])]#, #third term is race
                })
        except Exception as e:
            print(folder+'/UTKFace/'+image)
            continue
        
    return allinformation

def getMORPHdata(folder):
    allinformation=[]
    maps={'W':'caucasian','B':'afroamerican','A':'asian','H':'hispanic','O':'others'}
    map_gender={'M':'male','F':'female'}
    labels=pd.read_csv(folder+'/morph/morph_2008_nonCommercial.csv')
    for i in range(len(labels['photo'])):
        allinformation.append(
            {
            'image_path':folder+'/morph/'+labels['photo'][i],
            'age':int(labels['age'][i]),
            'gender':map_gender[labels['gender'][i]],
            'race':maps[labels['race'][i]]#, #third term is race
            }
        )
    return allinformation
        
def getAPPAdata(folder):
    allinformation=[]

    catelabels=defaultdict(dict)
    for subsets in ['train','valid','test']:
        filename=folder+'/appa/allcategories_{}.csv'.format(subsets)
        labels=pd.read_csv(filename)
        for i in range(len(labels['file'])):
            catelabels[labels['file'][i]]['gender']=labels['gender'][i]
            catelabels[labels['file'][i]]['race']=labels['race'][i]
    
    for subsets in ['train','valid','test']:
        filename=folder+'/appa/gt_avg_{}.csv'.format(subsets)
        labels=pd.read_csv(filename)
        for i in range(len(labels['file_name'])):
            allinformation.append(
                {
                'image_path':folder+'/appa/{}/'.format(subsets)+labels['file_name'][i],
                'age':int(labels['real_age'][i]),
                'gender':catelabels[labels['file_name'][i]]['gender'],
                'race':catelabels[labels['file_name'][i]]['race']#, #third term is race
                }
             )
    return allinformation

def getMegaasianData(data_folder):
    allinformation=[]
    for filename in ['train','test']:
      ages=open(data_folder+'/megaage_asian/list/'+filename+'_age.txt','r')
      imagenames=open(data_folder+'/megaage_asian/list/'+filename+'_name.txt','r')
      for lines in open(data_folder+'/megaage_asian/list/'+filename+'_age.txt','r'):
          allinformation.append(
            {
                'image_path':data_folder+'/megaage_asian/'+filename+'/'+imagenames.readline().strip(),
                'age':int(ages.readline().strip()),
                'gender':'N/A',
                'race':'asian'
            }
        )
    return allinformation

def getFGNETdata(folder):
    maps={0:'caucasian',1:'afroamerican',2:'asian',3:'indian',4:'others'}
    map_gender={0:'male',1:'female'}
    images=os.listdir(folder+'/FGNET/images')
    allinformation=[]
    for image in images:
        if image.split('.')[-1].lower()!='jpg':continue
        age=image.split('.')[0].split('A')[-1]
        try:
            int(age)
        except:
            continue
        allinformation.append(
            {
            'image_path':folder+'/FGNET/images/'+image,
            'age':int(age),# first term is age
            'gender':'N/A',#second term is gender
            'race':'N/A'#, #third term is race
            }
        )
    return allinformation
