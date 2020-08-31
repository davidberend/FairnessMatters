# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import scipy.io
import argparse
from collections import defaultdict
import pandas as pd
import face_recognition
from get_raw_data import getUTKdata, getMORPHdata, getAPPAdata, getMegaasianData, getFGNETdata, getIMDB, getWIKI

# Get the raw of the datasets
def get_raw_information(data_folder):
    wikiinfo=scipy.io.loadmat(data_folder+'/wiki_crop/wiki.mat')
    imdbinfo=scipy.io.loadmat(data_folder+'/imdb_crop/imdb.mat')
    UTK=getUTKdata(data_folder)
    WIKI=getWIKI(wikiinfo,data_folder)
    Mega=getMegaageData(data_folder)
    IMDB=getIMDB(imdbinfo,data_folder)
    return [Mega,UTK,WIKI,IMDB]

def main(data_folder):
    # Load all datasets
    datasets=get_raw_information(data_folder)

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='/mnt/nvme/aibias/OriDatasets/')
    args = parser.parse_args()
    data_folder = args.dir
    main(data_folder)

    
    

    



