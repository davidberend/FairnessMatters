from utils import data_utils
from models import Generalmodels
from torchvision import transforms, models
from utils.train_utils import test,test_range
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn
from collections import defaultdict
import torchvision.transforms as trn
from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed
from PIL import Image

def test_model(test_path,model_name,trained_model,save_path,traineddataset,testdataset,opt,lr,pretrain,num_classes,test_split=False):
    state = defaultdict()
    test_loader=defaultdict()
    test_transform = trn.Compose([trn.ToTensor(),trn.Normalize([0.566,0.496,0.469], [0.266,0.256,0.258])])
    img_pixels=(224,224)

    # Load test data
    test_set_X,test_set_y,_=data_utils.process_data(test_path)
    test_loader=data_utils.make_dataloader_iter(test_set_X, test_set_y, img_size=img_pixels, batch_size=10, transform_test=test_transform)

    # Load model
    if model_name=='alexnet':
        net = Generalmodels.alexnet(num_classes,pretrain,trained_model,if_test=True)
    elif model_name=='VGG':
        net=Generalmodels.VGG16(num_classes,pretrain,trained_model,if_test=True)
    elif model_name=='densenet121':
        net = Generalmodels.densenet121(num_classes,pretrain,trained_model,if_test=True)
    elif model_name=='resnet50':
        net = Generalmodels.resnet50(num_classes,pretrain,trained_model,if_test=True)
    
    device=torch.device("cuda")
    net.load_state_dict(torch.load(trained_model))
    net=net.to(device)
    
    # If test general performance
    if not test_split:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        test(net,test_loader,state)
        with open(os.path.join(save_path,  '{}_{}_{}_{}_{}_results.csv'.format(
            traineddataset,model_name,opt,lr,testdataset)), 'a') as f:
            f.write('%s,%0.6f\n' % (
            '(test)',
            state['test_accuracy'],
            ))
    
    # Test performance per age
    if test_split:
        new_path=os.path.join(save_path,'split')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        if testdataset=='FineTuneData':
            testdataset='testing'

        gt,tp,mae,prediction,classpredicts,regrepredicts,labels=\
            test_range(net,test_loader,state,num_classes)
        
        # write predictions 
        f=open(os.path.join(new_path,  '{}_{}_{}_{}_{}_allpredictions.csv'.format(
            traineddataset,model_name,opt,lr,testdataset)), 'w')
        f.write('%s,%s,%s\n'%('label','classpred','regreepred',))
        for i in range(len(classpredicts)):
                f.write('%d,%d,%0.2f\n' % (
                labels[i],
                classpredicts[i],
                regrepredicts[i]
                ))
        f.close()

        # Write performance
        f=open(os.path.join(new_path,  '{}_{}_{}_{}_{}_results.csv'.format(
                        traineddataset,model_name,opt,lr,testdataset)), 'w')
        f.write('%s,%s,%s,%s,%s\n'%('age','number','accuracy','mae','perceived'))
        for i in range(len(gt)):
                f.write('%d,%d,%0.6f,%0.3f,%0.3f\n' % (
                i,
                gt[i],
                tp[i]/gt[i],
                mae[i]/gt[i],
                prediction[i]/gt[i]
                ))
        f.close()



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-test_path', help='base folder',default='datasets')
    parser.add_argument('-test_split', help='if test split', type=int,default=1)
    parser.add_argument('-result_folder', help='test results to be stored', default='test_results')
    parser.add_argument('-pretrain',action='store_true',help='if this is a pretraining procedure')
    parser.add_argument('-trained_model',type=str,help='The trained model')
    
    args = parser.parse_args()

    test_path=args.test_path
    test_split=True if args.test_split!=0 else False
    testdataset=test_path.split('/')[-1].split('_')[0]
    save_path=args.result_folder
    save_path=os.path.join(save_path,test_path.split('/')[-2])
    trained_model=args.trained_model
    pretrain=args.pretrain
    modelInfo=args.trained_model.split('/')[-1].split('_')

    model_name = modelInfo[0]
    traineddataset = modelInfo[1]
    opt = modelInfo[-2]
    lr = modelInfo[-1]
    print(model_name)
    model_path=None
    
    # Check the existance of trained model
    for file in os.listdir(trained_model):
        if file.split('.')[-1]=='pt':
            model_path=os.path.join(trained_model,file)
    if not model_path:
        raise IOError('model does not exist!!')
    
    test_model(test_path,model_name,model_path,save_path,traineddataset,testdataset,opt,lr,pretrain,num_classes=101,test_split=test_split)