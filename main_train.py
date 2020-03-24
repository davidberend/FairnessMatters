from utils import data_utils
from utils import model_utils
from models import vgg_face,Generalmodels
from torchvision import transforms, models
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn


def train_model(train_X_path,train_y_path,test_X_path,test_y_path,num_classes,version="Mega",batch_size=32,num_epochs=50,model_name="VGG-face",pretrained_path='./pretrained_model'):
    device= torch.device("cuda")
    channels = 3
    img_pixels = (224,224)
    lr = 0.001
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(img_pixels),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for binsize in [1]:
        classes = num_classes
        samples=np.load(train_X_path)
        labels=np.load(train_y_path)
        testsamples=np.load(test_X_path)
        testlabels=np.load(test_y_path)
        dataloaders={}
        dataloaders['train'] = data_utils.make_dataloader(samples,labels,img_size=img_pixels,batch_size=batch_size,transform_test=transform,shuffle=True)
        dataloaders['test'] = data_utils.make_dataloader(testsamples,testlabels,img_size=img_pixels,batch_size=batch_size,transform_test=transform,shuffle=True)
    
        print("[+] Training for %s with %s dataset started" % (model_name,version))
        # for input,label in dataloaders['test']:
            # print(label)
        if model_name=='VGGface':
            net = vgg_face.VGG_16(classes=classes)
            # net.load_weights()
        elif model_name=='VGG':
            net=Generalmodels.VGG16(classes)
        elif model_name=='densenet':
            net = Generalmodels.densenet121(classes)
        elif model_name=='resnet':
            net = Generalmodels.resnet50(classes)
        
        model_save_name = model_name+version+"full_image"
        model_utils.training_and_save_model(net, num_epochs, model_save_name,device,dataloaders,lr)

        print("[+] Training for %s with %s dataset completed" % (model_name,version))

        del net 


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-folder', help='base folder',
                        default='datasets')
    parser.add_argument('-train_X_path', help='training samples', default='UTK_train_X_full.npy')
    parser.add_argument('-train_y_path', help='training labels', default='UTK_train_y_full.npy')
    parser.add_argument('-test_X_path', help='test samples', default='UTK_test_X_full.npy')
    parser.add_argument('-test_y_path', help='test labels', default='UTK_test_y_full.npy')
    parser.add_argument('-model_name', help='model to be trained', default='VGG')
    parser.add_argument('-version', help='version_of_model', default='Mega')
    parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
    parser.add_argument('-pretrained', help='number of classes', default='./pretrained_model/vgg_face_torch/VGG_FACE.t7')
    parser.add_argument('-num_epoches', type=int, help='number of classes', default=100)

    args = parser.parse_args()
    train_X_path = os.path.join(args.folder, args.train_X_path)
    train_y_path = os.path.join(args.folder, args.train_y_path)
    test_X_path = os.path.join(args.folder, args.test_X_path)
    test_y_path = os.path.join(args.folder, args.test_y_path)

    model_name = args.model_name
    version = args.version
    num_classes = args.num_classes

    train_model(train_X_path,train_y_path,test_X_path,test_y_path,num_classes=num_classes,version=version,num_epochs=args.num_epoches,pretrained_path=args.pretrained,model_name=model_name)