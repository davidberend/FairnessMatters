# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import scipy.io
import argparse
from collections import defaultdict
import pandas as pd
import face_recognition

# Get the raw of the datasets
def getMegaageData(data_folder):
    allinformation=[]
    for filename in ['train','test']:
      ages=open(data_folder+'/megaage/list/'+filename+'_age.txt','r')
      imagenames=open(data_folder+'/megaage/list/'+filename+'_name.txt','r')
      for lines in open(data_folder+'/megaage/list/'+filename+'_age.txt','r'):
          allinformation.append(
            {
                'image_path':data_folder+'/megaage/'+filename+'/'+imagenames.readline().strip(),
                'age':int(ages.readline().strip())
            }
        )
    return allinformation

def getUTKdata(data_folder):
    images=os.listdir(data_folder+'/UTKFace')
    allinformation=[]
    for image in images:
        information=image.split('_')
        allinformation.append(
            {
            'image_path':data_folder+'/UTKFace/'+image,
            'age':int(information[0]),# first term is age
            'gender':information[1],#second term is gender
            'race':information[2]#, #third term is race
            }
        )
    return allinformation

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

# Get the raw of the datasets
def get_raw_information(data_folder):
    wikiinfo=scipy.io.loadmat(data_folder+'/wiki_crop/wiki.mat')
    imdbinfo=scipy.io.loadmat(data_folder+'/imdb_crop/imdb.mat')
    UTK=getUTKdata(data_folder)
    WIKI=getWIKI(wikiinfo,data_folder)
    Mega=getMegaageData(data_folder)
    IMDB=getIMDB(imdbinfo,data_folder)
    return [Mega,UTK,WIKI,IMDB]

if __name__=="__main__":

    # Load all datasets
    datasets=get_raw_information('/home/david/aibias/OriDatasets')

    # Remove the bad images which contains nothing
    exceptions=open('exceptions.txt','w')
    for dataset in datasets:
        for images in dataset:
            img=Image.open(images['image_path'])
            if img.size[0]<=10:
                print(images['image_path'])
                exceptions.write(images['image_path']+'\n')
                os.remove(images['image_path'])

    # Remove the images which contains no face or more than one face
    NoOrMoreFaces=open('NoOrMoreFaces.txt','w')
    for dataset in datasets:
        for images in dataset:
            if not os.path.exists(images['image_path']):
                continue
            image = face_recognition.load_image_file(images['image_path'])
            face_locations = face_recognition.face_locations(image,model='cnn')
            if (not face_locations) or len(face_locations)>1:
                os.remove(images['image_path'])
                NoOrMoreFaces.write(images['image_path']+'\n')


