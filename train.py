
from utils.train_utils import train_baseline,test,adjust_opt
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
from collections import defaultdict
import os
import sys
from utils import data_utils
from models import Generalmodels
import torch.optim.lr_scheduler
import time
import torch.optim as optim
from torchvision import models
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn

def train_model(train_path, test_path, batch_size=128, model_name = "resnet", opt="sgd",dataset="UTK",num_epochs=100,lr=0.01,
                pretrain=False,trained_model=None):
    # Configuration
    state = defaultdict()
    opt = opt
    img_pixels=(224,224)

    # loading data
    test_set_X,test_set_y,_=data_utils.process_data(test_path)
    train_set_X,train_set_y,num_classes=data_utils.process_data(train_path)
    train_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    test_transform = trn.Compose([
        trn.ToTensor(),
        trn.Normalize([0.550,0.500,0.483], [0.273,0.265,0.266])])
    train_loader = data_utils.make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels, batch_size=batch_size,
                                   transform_test=train_transform, shuffle=True)
    test_loader = data_utils.make_dataloader_iter(test_set_X, test_set_y, img_size=img_pixels, 
                                batch_size=batch_size, transform_test=test_transform)
    
    # For pre-train and fine-tune
    if pretrain:
        train_set_X,train_set_y,num_classes=data_utils.process_data(train_path)
        save_path = "./model_weights/pretrained/{}_{}_{}".format(model_name,opt,str(lr))
    else:
        num_classes=101
        save_path = "./model_weights/{}_{}_{}_{}".format(model_name,dataset,opt,str(lr))
    model_type = "{}_{}".format(model_name, dataset)
    print('Using Data: ',train_path)

    # Initializating model
    print("num_classes: ",num_classes)
    if model_name=='VGG':
        net=Generalmodels.VGG16(num_classes,pretrain, trained_model,if_test=False)
    elif model_name=='resnet50':
        net = Generalmodels.resnet50(num_classes,pretrain, trained_model,if_test=False)
    elif model_name=='densenet121':
        net = Generalmodels.densenet121(num_classes,pretrain, trained_model,if_test=False)
    elif model_name=='alexnet':
        net = Generalmodels.alexnet(num_classes,pretrain, trained_model,if_test=False)
    
    device=torch.device("cuda")
    net=net.to(device)
    
    # Defining opt method
    if opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # Create results saving path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)
    with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'w') as f:
        f.write('epoch,time(s),train_loss,train_acc,train_mae,test_loss,test_acc(%),test_mae\n')
    print('Beginning Training for {} on {}\n'.format(model_name,dataset))

    # Main loop
    best_epoch = 0
    best_acc = 0.0
    best_mae = 100.0
    prev_path=' '
    for epoch in range(num_epochs):
        adjust_opt(opt, optimizer, epoch,lr)
        state['epoch'] = epoch
        begin_epoch = time.time()

        # Train and Test
        train_baseline(net,train_loader,optimizer,state)
        test(net,test_loader,state)
        scheduler.step()
         # Save model
        cur_mae = state['test_mae']
        if cur_mae < best_mae:
            cur_save_path = os.path.join(save_path, '{}_epoch_{}_{}.pt'.format(model_type,epoch,cur_mae))
            torch.save(net.state_dict(),cur_save_path)
            if os.path.exists(prev_path): 
                os.remove(prev_path)
            prev_path = cur_save_path
            best_epoch = epoch
            best_mae = cur_mae
              
        # Save results
        with open(os.path.join(save_path,  '{}_training_results.csv'.format(model_type)), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.4f,%0.4f,%0.6f,%0.4f,%0.4f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['train_accuracy'],
                state['train_mae'],
                state['test_loss'],
                state['test_accuracy'],
                state['test_mae']
            ))

        # Print results 
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

    parser.add_argument('-datafolder', help='data folder',default='./data/original')
    parser.add_argument('-train_path', help='training samples', default='train.tsv')
    parser.add_argument('-test_path', help='test samples', default='test.tsv')
    parser.add_argument('-model_name', help='model to be trained', default='resnet50')
    parser.add_argument('-dataset', help='dataset to be trained', default='ori_balanced_data')
    parser.add_argument('-opt', type=str, help='choose optimizer', default="adam")
    parser.add_argument('-num_epoches', type=int, help='number of classes', default=100)
    parser.add_argument('-lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('-pretrain',action='store_true',help='if this is a pretraining procedure')
    parser.add_argument('-pretrained_model',type=str,help='The pre-trained model')
    
    args = parser.parse_args()

    train_path = os.path.join(args.datafolder, args.train_path)
    test_path = os.path.join(args.datafolder, args.test_path)

    model_name = args.model_name
    dataset = args.dataset
    opt = args.opt
    lr=args.lr
    num_epoches=args.num_epoches
    pretrain=args.pretrain
    trained_model=args.pretrained_model
    dataset = args.train_path.split('_')[0]

    # Checking the existance of pre-trained model
    if not pretrain:
        if not os.path.exists(trained_model):
            raise FileExistsError("Pretrained Model does not exiset!")
        if trained_model.split('/')[-1].split('_')[1]!=model_name:
            raise ValueError("Model name does not match the pre-trained model!")
    lists=os.listdir(trained_model)
    
    # Getting the path of the pre-trained model
    for i in lists:
        if i.split('.')[-1]=='pt':
            trained_model=os.path.join(trained_model,i)
    
    # Train
    train_model(train_path, test_path, model_name=model_name,
    opt=opt,dataset=dataset,num_epochs=num_epoches,lr=lr,pretrain=pretrain,trained_model=trained_model)
