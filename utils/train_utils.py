import torch
import numpy as np
import torch.nn.functional as F
from progressbar import ProgressBar,Percentage,Bar,Timer,ETA,FileTransferSpeed

# Adjust the learning rate 
def adjust_opt(optAlg, optimizer, epoch,lr):
    # Decrease by a factor of 10 after 10 epochs
    if optAlg == 'sgd':
        if epoch % 10==0: lr = lr*(0.1**(epoch//10))
        else: return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Decrease by a factor of 5 after 40 epochs
    if optAlg == 'adam':
        if epoch % 40==0: lr = lr*(0.2**(epoch//40))
        else: return
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def train_baseline(net,train_loader,optimizer,state):

    # enter train mode
    net.train()  

    # Recording training results
    loss_avg = 0.0
    correct  = 0

    # Visualizting training procedure
    total = len(train_loader)
    widgets = ['Training: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
           ' ', ETA(), ' ', FileTransferSpeed()]  
    progress = ProgressBar(widgets=widgets, maxval=total)
    mae=0.0
    # Begin training
    for data, target in progress(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # forward
        output = net(data)

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward() 
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss.data) * 0.2

        # accuracy
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()
        outputs=F.softmax(output,dim=1)

        for i in range(len(output)):
            age=float(expected_age(outputs[i].data.cpu()))
            mae+=abs(age-float(target[i].data.cpu()))

    progress.finish()
    state['train_loss'] = loss_avg
    state['train_accuracy'] = correct / len(train_loader.dataset)
    state['train_mae']=mae/len(train_loader.dataset)
    print("train_loss:{},train_accuracy;{};train_mae{}".format(state['train_loss'], state['train_accuracy'], state['train_mae']))


def expected_age(vector):
    # Get expected age accoring to probabilities and ages
    # Used for DEX
    
    res = [(i)*v for i, v in enumerate(vector)]
    # print(vector,sum(res))
    return sum(res)

# test function
def test(net,test_loader,state):
    # Enter evaluzaion mode
    net.eval()

    # Recording test results
    loss_avg = 0.0
    correct = 0

    # Visualizing test procedure
    total = len(test_loader)
    widgets = ['Testing: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
           ' ', ETA(), ' ', FileTransferSpeed()]  
    progress = ProgressBar(widgets=widgets, maxval=total)

    mae=0.0
    with torch.no_grad():
        for data, target in progress(test_loader):
            data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
            loss = F.cross_entropy(output, target) 
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            outputs=F.softmax(output,dim=1)
            
            # mae 
            for i in range(len(pred)):
                # age=float(pred[i].data.cpu())
                age=float(expected_age(outputs[i].data.cpu()))
                # break
                mae+=abs(age-float(target[i].data.cpu()))
            # break
            # test loss average
            loss_avg += float(loss.data)

    progress.finish()
    state['test_loss'] = loss_avg / len(test_loader)
#     state['test_loss'] = loss_avg
    state['test_accuracy'] = correct / len(test_loader.dataset)
    state['test_mae']=mae/len(test_loader.dataset)
    print("\ntest_loss:{},test_accuracy;{},test_mae:{}".format(state['test_loss'], state['test_accuracy'],state['test_mae']))
    
# Test according to each age
def test_range(net,test_loader,state,num_classes):
    
    net.eval()
    loss_avg = 0.0
    correct = 0

    total = len(test_loader)
    widgets = ['Testing: ',Percentage(), ' ', Bar('#'),' ', Timer(),  
           ' ', ETA(), ' ', FileTransferSpeed()]  
    progress = ProgressBar(widgets=widgets, maxval=total)

    # Recording accuracy and mae per age
    grandtruth=np.zeros(num_classes)
    truepositive=np.zeros(num_classes)
    mae=np.zeros(num_classes)
    predictions=np.zeros(num_classes)
    allmae=0.0
    classpredicts=[]
    regrepredicts=[]
    labels=[]

    with torch.no_grad():
        for data, target in progress(test_loader):
            data, target = data.cuda(), target.cuda()
            # forward
            output = net(data)
          
            loss = F.cross_entropy(output, target) 
            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
            outputs=F.softmax(output,dim=1)
            for i in range(len(pred)):
                # accuracy per age
                grandtruth[int(target[i].data)]+=1
                predictions[int(target[i].data)]+=pred[i].data
                if pred[i].data==target[i].data:
                    truepositive[int(pred[i].data)]+=1
                # mae per age
                age=int(pred[i].data.cpu())
                mae[int(target[i].data)]+=abs(int(target[i].data)-age)
                allmae+=abs(int(target[i].data)-age)

                # record predicted age 
                classpredicts.append(int(pred[i].data.cpu()))
                regrepredicts.append(age)
                labels.append(target[i].data.cpu())
            
            # test loss average
            loss_avg += float(loss.data)
        # print(truepositive)
    progress.finish()

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)
    state['mae']=allmae/len(test_loader.dataset)
    print("test_loss:{},test_accuracy;{},mae:{}".format(state['test_loss'], state['test_accuracy'],state['mae']))
    return grandtruth,truepositive,mae,predictions,classpredicts,regrepredicts,labels

