
from utils.train_utils import train_baseline,test,adjust_opt

import torch.backends.cudnn as cudnn
import torchvision.transforms as trn

from collections import defaultdict

import os
import sys

from utils import data_utils
from utils import model_utils
from models import vgg_face,Generalmodels

import time
import torch.optim as optim

from torchvision import models
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn

import argparse



def train_model(train_X_path, train_y_path, test_X_path, test_y_path, batch_size=64,model_name = "resnet", num_classes=10, opt="sgd",dataset="UTK",num_epochs=100):

    # Configuration
    #cudnn.benchmark = True  # fire on all cylinders
    num_classes = num_classes
    state = defaultdict()
    start_epoch = 0
    end_epoch = num_epochs
    opt = opt
    img_pixels=(224,224)
    save_path = "./model_weights/trained_{}_{}_{}".format(model_name,dataset,opt)
    model_type = "{}_{}".format(model_name, dataset)

    train_transform = trn.Compose([
        trn.RandomResizedCrop(224),
        trn.Resize(img_pixels),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transform = trn.Compose([trn.ToTensor(),trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_set_X = np.load(train_X_path)
    train_set_y = np.load(train_y_path)

    test_set_X = np.load(test_X_path)
    test_set_y = np.load(test_y_path)


    train_loader = data_utils.make_dataloader(train_set_X, train_set_y, img_size=img_pixels, batch_size=batch_size,
                                   transform_test=train_transform, shuffle=True)

    test_loader = data_utils.make_dataloader(test_set_X, test_set_y, img_size=img_pixels, batch_size=batch_size, transform_test=test_transform)
    

    if model_name=='VGGface':
        net = vgg_face.VGG_16(num_classes)
    elif model_name=='VGG':
        net=Generalmodels.VGG16(num_classes)
    elif model_name=='densenet':
        net = Generalmodels.densenet121(num_classes)
    elif model_name=='resnet':
            net = Generalmodels.resnet50(num_classes)

    device=torch.device("cuda")
    net=net.to(device)
    # if torch.cuda.device_count()>1:
    #     net = torch.nn.DataParallel(net)
    #     net.cuda()
    # else:
    #     net.cuda()

    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)
        
    
    # Make save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)
    with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'w') as f:
        f.write('epoch,time(s),train_loss,train_acc,test_loss,test_acc(%)\n')
    print('Beginning Training\n')
    # Main loop
    best_epoch = 0
    best_acc = 0.0
    for epoch in range(start_epoch, end_epoch):
        
        adjust_opt(opt, optimizer, epoch)
        if epoch % 50 == 0:
            print(">>>>LR:{}".format(optimizer.param_groups[0]['lr']))
        state['epoch'] = epoch
        begin_epoch = time.time()

        train_baseline(net,train_loader,optimizer,state)
        test(net,test_loader,state)
         # Save model
        if epoch==0:
            best_epoch = epoch
            best_acc = state['test_accuracy']
            cur_save_path = os.path.join(save_path, '{}_epoch_{}_{}.pt'.format(model_type,best_epoch,best_acc))
            # network.module.state_dict()
            torch.save(net.state_dict(),cur_save_path)
        cur_acc = state['test_accuracy']
        if cur_acc > best_acc:
            cur_save_path = os.path.join(save_path, '{}_epoch_{}_{}.pt'.format(model_type,epoch,cur_acc))
            # network.module.state_dict()
            torch.save(net.state_dict(),cur_save_path)
            prev_path = os.path.join(save_path, '{}_epoch_{}_{}.pt'.format(model_type,best_epoch,best_acc))
            if os.path.exists(prev_path): 
                os.remove(prev_path)
            best_epoch = epoch
            best_acc = cur_acc
              
        with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.4f,%0.6f,%0.4f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['train_accuracy'],
                state['test_loss'],
                state['test_accuracy'],
            ))
            
        print('|Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f}|  Train Acc {3:.4f} | Test Loss {4:.4f} | Test Acc {5:.4f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['train_accuracy'],
        state['test_loss'],
        state['test_accuracy'])
        )

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-folder', help='base folder',
                        default='datasets')
    parser.add_argument('-train_X_path', help='training samples', default='UTK_train_X_5yr.npy')
    parser.add_argument('-train_y_path', help='training labels', default='UTK_train_y_5yr.npy')
    parser.add_argument('-test_X_path', help='test samples', default='UTK_test_X_5yr.npy')
    parser.add_argument('-test_y_path', help='test labels', default='UTK_test_y_5yr.npy')
    parser.add_argument('-model_name', help='model to be trained', default='VGG')
    parser.add_argument('-dataset', help='version_of_model', default='Mega')
    parser.add_argument('-num_classes', type=int, help='number of classes', default=10)
    parser.add_argument('-opt', type=str, help='choose optimizer', default="sgd")
    parser.add_argument('-num_epoches', type=int, help='number of classes', default=100)

    args = parser.parse_args()

    train_X_path = os.path.join(args.folder, args.train_X_path)
    train_y_path = os.path.join(args.folder, args.train_y_path)
    test_X_path = os.path.join(args.folder, args.test_X_path)
    test_y_path = os.path.join(args.folder, args.test_y_path)

    model_name = args.model_name
    dataset = args.dataset
    num_classes = args.num_classes
    opt = args.opt

    train_model(train_X_path, train_y_path, test_X_path, test_y_path, model_name=model_name,
                num_classes=num_classes, opt=opt,dataset=dataset,num_epochs=args.num_epoches)