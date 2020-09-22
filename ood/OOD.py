import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
sys.path.append('../')
from utils import data_utils
from OOD_techniques.glod import retrieve_scores, ConvertToGlod, calc_gaussian_params
from models.Generalmodels import create_Resnet
from utils.OOD_utils import visualize_distribution, take_samples

def Convert(train_path, batch_size, weight_path, device, transform, version=50, num_classes=100, img_pixels= (224, 224)):
    # data
    
    train_set_X, train_set_y, _ = data_utils.process_data(train_path)

    print('Using data:{}'.format(train_path))
    train_loader = data_utils.make_dataloader_iter(train_set_X, train_set_y, img_size=img_pixels,
                                                batch_size=batch_size, transform_test=transform, shuffle=True)

    # Creat the original model
    print('Creating model using model :{}'.format(weight_path))
    model = create_Resnet(resnet_number=version, num_classes=num_classes,
                        gaussian_layer=False, weight_path=weight_path,
                        device=device)

    # Convert it to glod model
    print('Begin converting')
    model = ConvertToGlod(model, num_classes=num_classes)
    print('Begin calculating')
    covs, centers = calc_gaussian_params(model, train_loader, device, num_classes)
    print('Done Calculation')
    model.gaussian_layer.covs.data = covs
    model.gaussian_layer.centers.data = centers

    return model


def ConvertAndGetscores(train_path, batch_size, weight_path, num_classes, glod_k = 100, quantile=0.05):
    
    img_pixels = (224,224)
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.550, 0.500, 0.483], [0.273, 0.265, 0.266])])
    # Load data
    in_set_X, in_set_y, _ = data_utils.process_data(in_path)
    out_set_X, out_set_y, _ = data_utils.process_data(out_path)
    in_loader = data_utils.make_dataloader_iter(in_set_X, in_set_y, img_size=img_pixels,
                                            batch_size=batch_size, transform_test=transform, shuffle=False)
    out_loader = data_utils.make_dataloader_iter(out_set_X, out_set_y, img_size=img_pixels,
                                             batch_size=batch_size, transform_test=transform, shuffle=False)

    device='cuda:0'
    device = torch.device(device)
    
    # Load original model
    model = create_Resnet(resnet_number=50, num_classes=num_classes,
                      gaussian_layer=False, weight_path=weight_path,
                      device=device)

    # Convert to Glod
    glod_model = Convert(train_path, batch_size, weight_path, device,transform)
    torch.save(glod_model.state_dict(),'glod_model.pt')
    # Retrive scores
    out_scores = retrieve_scores(glod_model, out_loader, device, glod_k)
    in_scores = retrieve_scores(glod_model, in_loader, device, glod_k)
    torch.save(in_scores, 'In_Dis_Scores.pt')
    torch.save(out_scores,'Out_of_Dis_Scores.pt')
    # Visualize distribution
    visualize_distribution(in_scores, out_scores, glod_k)

    # Sample data
    # take_samples(out_scores, out_set_X,out_set_y,out_race,quantile=quantile)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='control experiment')

    parser.add_argument('-train_path', help='The dataset which used to train the original age prediction model',
    default='/home/david/bias_ai_glod/SweetGAN/data/Balanced/FineTuneData_train_info_1yr.txt')
    parser.add_argument('-model_path', help='The path of the trained age prediction model',
    default = '/home/david/aibias/model_weights/regression/both_resnet50_FineTuneData_adam_0.0001_100/resnet50_FineTuneData_epoch_122_0.3150045821877864.pt')
    parser.add_argument('-in_path', help='In distribution dataset',
    default='/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr.txt')
    parser.add_argument('-out_path', help='Out of distribution dataset',
    default='/home/david/aibias/datasets/1_101_1yr/all/FineTuneData_train_info_1yr_augextreme.txt')
    parser.add_argument('-batch_size', type=int, help='batch_size', default=256)
    parser.add_argument('-glod_k', type=int, help='glod_k value', default=100)
    parser.add_argument('-quantile', type=int, help='quantile', default=0.05)
    parser.add_argument('-num_classes', type=int, help='number of calss', default=100)

    
    args = parser.parse_args()

    batch_size = args.batch_size
    train_path = args.train_path
    weight_path = args.model_path
    in_path = args.in_path
    out_path = args.out_path
    glod_k = args.glod_k
    quantile = args.quantile
    num_classes = args.num_classes
    
    ConvertAndGetscores(train_path, batch_size, weight_path, num_classes, glod_k = glod_k, quantile=quantile)