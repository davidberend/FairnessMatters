import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

__all__ = ['GaussianLayer', 'GlodLoss', 'retrieve_scores','predict']


class GaussianLayer(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(GaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.centers = nn.Parameter(
            0.5*torch.randn(n_classes, input_dim).cuda())
        self.covs = nn.Parameter(
            0.2+torch.tensor(np.random.exponential(scale=0.3,
                                                   size=(n_classes,
                                                         input_dim))).cuda())

    def forward(self, x):
        covs = self.covs.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        centers = self.centers.unsqueeze(0).expand(
            x.size(0), self.n_classes, self.input_dim)
        diff = x.unsqueeze(1).repeat(1, self.n_classes, 1) - centers

        Z_log = (-0.5*torch.sum(torch.log(self.covs +
                                          np.finfo(np.float32).eps), axis=-1)
                 - 0.5*self.input_dim*np.log(2*np.pi))
        exp_log = -0.5 * \
            torch.sum(diff*(1/(covs+np.finfo(np.float32).eps))*diff, axis=-1)
        likelihood = Z_log+exp_log
        return likelihood

    def clip_convs(self):
        '''
        Cliping the convariance matricies to be alaways positive. \n
        Use: call after optimizer.step()
        '''
        with torch.no_grad():
            self.covs.clamp_(np.finfo(np.float32).eps)

    def cov_regulaizer(self, beta=0.01):
        '''
        Covarianvce regulzer \n
        Use: add to the loss if used for OOD detection
        '''
        return beta*(torch.norm(self.covs, p=2))

class ConvertToGlod(nn.Module):
    def __init__(self, net, num_classes=100):
        super(ConvertToGlod, self).__init__()
        self.gaussian_layer = GaussianLayer(
            input_dim=2048, n_classes=num_classes)
        self.net = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = self.gaussian_layer(out)
        return out

    def penultimate_forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

def predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        # for batch_idx, (inputs, _) in enumerate(loader):
        for inputs,_ in tqdm(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions).cuda()
    return predictions

def retrieve_scores(model, loader, device, k):
    preds = predict(model, loader, device)
    top_k = preds.topk(k).values.squeeze()
    avg_ll = np.mean(top_k[:, 1:k].cpu().detach().numpy())
    llr = top_k[:, 0].cpu()-avg_ll
    return llr

def calc_gaussian_params(model, loader, device, n_classes):
    outputs_list = []
    target_list = []
    with torch.no_grad():
        for (inputs, targets) in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.penultimate_forward(inputs)
            outputs_list.append(outputs)
            target_list.append(targets)
        outputs = torch.cat(outputs_list, axis=0)
        target_list = torch.cat(target_list)
        x_dim = outputs.size(1)
        centers = torch.zeros(n_classes, x_dim).cuda()
        covs = 0.01*torch.ones(n_classes, x_dim).cuda()
        for c in range(n_classes):
            class_points = outputs[c == target_list]
            if class_points.size(0) <= 1:
                continue
            centers[c] = torch.mean(class_points, axis=0)
            covs[c] = torch.var(class_points, axis=0)
        return covs, centers