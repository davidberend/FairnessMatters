from utils import data_utils
from utils import model_utils
from models import vgg_face,Generalmodels
from torchvision import transforms, models
from utils.train_utils import test
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn
from collections import defaultdict
import torchvision.transforms as trn


def test_model(test_paths,train_paths,model_name,trained_model,num_classes,save_path,traineddataset):
    state = defaultdict()
    test_loader=defaultdict()
    train_loader=defaultdict()
    test_transform = trn.Compose([trn.ToTensor(),trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_pixels=(224,224)
    for i in test_paths:
        test_set_X=np.load(test_paths[i][0])
        test_set_y=np.load(test_paths[i][1])
        test_loader[i]=data_utils.make_dataloader(test_set_X, test_set_y, img_size=img_pixels, batch_size=64, transform_test=test_transform)
    for i in train_paths:
        train_set_X=np.load(train_paths[i][0])
        train_set_y=np.load(train_paths[i][1])
        train_loader[i]=data_utils.make_dataloader(train_set_X, train_set_y, img_size=img_pixels, batch_size=64, transform_test=test_transform)
    if model_name=='VGGface':
        net = vgg_face.VGG_16(num_classes)
    elif model_name=='VGG':
        net=Generalmodels.VGG16(num_classes)
    elif model_name=='densenet':
        net = Generalmodels.densenet121(num_classes)
    elif model_name=='resnet':
        net = Generalmodels.resnet50(num_classes)
    device=torch.device("cuda")
    net.load_state_dict(torch.load(trained_model))
    net=net.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # for i in train_loader:
    test(net,train_loader[traineddataset],state)
    with open(os.path.join(save_path,  '{}_{}_testing_results.csv'.format(model_name,traineddataset)), 'a') as f:
            f.write('%s,%0.6f\n' % (
                traineddataset,
                state['test_accuracy'],
            ))
    test(net,test_loader[traineddataset],state)
    with open(os.path.join(save_path,  '{}_{}_testing_results.csv'.format(model_name,traineddataset)), 'a') as f:
        f.write('%s,%0.6f\n' % (
                traineddataset,
                state['test_accuracy'],
            ))



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-folder', help='base folder',
                        default='datasets')
    parser.add_argument('-UTK_X_path', help='test samples', default='UTK_test_X.npy')
    parser.add_argument('-UTK_y_path', help='test labels', default='UTK_test_y.npy')
    parser.add_argument('-Mega_X_path', help='test samples', default='Mega_test_X.npy')
    parser.add_argument('-Mega_y_path', help='test labels', default='Mega_test_y.npy')
    parser.add_argument('-IMDB_X_path', help='test samples', default='IMDB_test_X.npy')
    parser.add_argument('-IMDB_y_path', help='test labels', default='IMDB_test_y.npy')
    parser.add_argument('-WIKI_X_path', help='test samples', default='WIKI_test_X.npy')
    parser.add_argument('-WIKI_y_path', help='test labels', default='WIKI_test_y.npy')
    # parser.add_argument('-model_name', help='model to be tested', default='densenet')
    parser.add_argument('-save_path', help='test results to be stored', default='test_results_001')
    # parser.add_argument('-trained_model', help='version_of_model', default='./model_weights/densenetUTK')
    parser.add_argument('-num_classes', type=int, help='number of classes', default=4)

    args = parser.parse_args()
    test_paths=defaultdict()
    train_paths=defaultdict()
    # Megapath='Mega'
    # test_paths['UTK']=[os.path.join(args.folder, 'UTK_test_X_5yr.npy'),os.path.join(args.folder, 'UTK_test_y_5yr.npy')]
    # test_paths['Mega']=[os.path.join(args.folder, 'Mega_test_X_5yr.npy'),os.path.join(args.folder, 'Mega_test_y_5yr.npy')]
    # test_paths['IMDB']=[os.path.join(args.folder, 'IMDB_test_X_5yr.npy'),os.path.join(args.folder, args.IMDB_y_path)]
    # test_paths['WIKI']=[os.path.join(args.folder, 'WIKItest_X_5yr.npy'),os.path.join(args.folder, args.WIKI_y_path)]
    # train_paths['UTK']=[os.path.join(args.folder, args.UTK_X_path),os.path.join(args.folder, args.UTK_y_path)]
    # train_paths['Mega']=[os.path.join(args.folder, args.Mega_X_path),os.path.join(args.folder, args.Mega_y_path)]
    # train_paths['IMDB']=[os.path.join(args.folder, args.IMDB_X_path),os.path.join(args.folder, args.IMDB_y_path)]
    # train_paths['WIKI']=[os.path.join(args.folder, args.WIKI_X_path),os.path.join(args.folder, args.WIKI_y_path)]

    for path in os.listdir(args.folder):
        if len(path.split('_'))<4:
            continue
        if path.split('_')[3][0:3]!='5yr':
            continue
        test_paths[path.split('_')[0]]=['temp','temp']
        train_paths[path.split('_')[0]]=['temp','temp']

    for path in os.listdir(args.folder):
        if len(path.split('_'))<4:
            continue
        if path.split('_')[3][0:3]!='5yr':
            continue
        if path.split('_')[1]=='test' and path.split('_')[2]=='X':
            test_paths[path.split('_')[0]][0]=(os.path.join(args.folder,path))
        if path.split('_')[1]=='test' and path.split('_')[2]=='y':
            test_paths[path.split('_')[0]][1]=(os.path.join(args.folder,path))
        if path.split('_')[1]=='train' and path.split('_')[2]=='X':
            train_paths[path.split('_')[0]][0]=(os.path.join(args.folder,path))
        if path.split('_')[1]=='train' and path.split('_')[2]=='y':
            train_paths[path.split('_')[0]][1]=(os.path.join(args.folder,path))

    print(test_paths,train_paths)

    # model_name = args.model_name
    # trained_model=args.trained_model
    num_classes=args.num_classes
    for path in os.listdir('./model_weights_001'):
        if len(path.split('_'))<2:
            continue
        model_name=path.split('_')[0]
        traineddataset=path.split('_')[1]
        trained_model=os.path.join('model_weights',path)
        test_model(test_paths=test_paths,train_paths=train_paths,model_name=model_name,trained_model=trained_model,num_classes=num_classes,save_path=args.save_path,traineddataset=traineddataset)