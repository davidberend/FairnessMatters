from torchvision import transforms, models
import numpy as np
import argparse
import copy
import torch
import os
import torch.nn as nn

 
def VGG16(classes):
    model=models.vgg16()
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, classes)
    return model


def resnet50(classes):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)
    return model

def densenet121(classes):
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, classes)
    return model