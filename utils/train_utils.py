import torch
import numpy as np
import torch.nn.functional as F
from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch == 100: lr = 1e-2
#         elif epoch == 150: lr = 1e-2
        elif epoch == 200: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_baseline(net,train_loader,optimizer,state):
    net.train()  # enter train mode
    loss_avg = 0.0
    correct  = 0
    total = len(train_loader)
    widgets = ['Training: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
           ' ', ETA(), ' ', FileTransferSpeed()]  
    progress = ProgressBar(widgets=widgets, maxval=total)
    for data, target in progress(train_loader):
        data, target = data.cuda(), target.cuda()

        # forward
        output = net(data)
        # backward
#         scheduler.step()
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward() 
        optimizer.step()
        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss.data) * 0.2
        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

    progress.finish()
    state['train_loss'] = loss_avg
    state['train_accuracy'] = correct / len(train_loader.dataset)
    print("train_loss:{},train_accuracy;{}".format(state['train_loss'], state['train_accuracy']))


# test function
def test(net,test_loader,state):
    net.eval()
    loss_avg = 0.0
    correct = 0
    total = len(test_loader)
    widgets = ['Testing: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
           ' ', ETA(), ' ', FileTransferSpeed()]  
    progress = ProgressBar(widgets=widgets, maxval=total)
    
    with torch.no_grad():
        for data, target in progress(test_loader):
            data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
          
            loss = F.cross_entropy(output, target) 
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            # test loss average
            loss_avg += float(loss.data)
    progress.finish()
    state['test_loss'] = loss_avg / len(test_loader)
#     state['test_loss'] = loss_avg
    state['test_accuracy'] = correct / len(test_loader.dataset)
    print("test_loss:{},test_accuracy;{}".format(state['test_loss'], state['test_accuracy']))

    

