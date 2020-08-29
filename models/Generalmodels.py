from torchvision import transforms, models
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn

 
def alexnet(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model=models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    else:
        model=models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    return model

def VGG16(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model=models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
    else:
        print('Using both')
        model=models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, classes)
        print(model.classifier[6])
    return model

def densenet121(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
    else:
        model=models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        print('Using trained model from {}'.format(trained_model))
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, classes)
    print(model.classifier)
    return model  


def resnet50(classes,pretrain, trained_model,if_test=False):
    if if_test:
        model=models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
        model.load_state_dict(torch.load(trained_model))
        return model
    if pretrain:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    else:
        model=models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 101)
        model.load_state_dict(torch.load(trained_model))
        print('Using trained model from {}'.format(trained_model))
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    return model    

def create_Resnet(resnet_number=50, num_classes=100,
                  gaussian_layer=True, weight_path=None,
                  device=None):
    model = getattr(models, 'resnet'+str(resnet_number))()
    if weight_path is not None:
        model.fc = nn.Linear(in_features=2048, out_features=num_classes)
        state = torch.load(weight_path)
        model.load_state_dict(state)
    if gaussian_layer:
        model.fc = GaussianLayer(2048, n_classes=num_classes)
    if device is not None:
        model.to(device)
    return model