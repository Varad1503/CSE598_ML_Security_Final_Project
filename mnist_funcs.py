import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import ipdb
import random



def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
pert = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X +pert), y)
    loss.backward()
    return epsilon *pert.grad.detach().sign()

def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]

def norms_l2(Z):
    return norms(Z)

def norms_l2_squeezed(Z):
    return norms(Z).squeeze(1).squeeze(1).squeeze(1)

def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None]

def norms_l1_squeezed(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:,None,None,None].squeeze(1).squeeze(1).squeeze(1)

def norms_l0(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float()

def norms_l0_squeezed(Z):
    return ((Z.view(Z.shape[0], -1)!=0).sum(dim=1)[:,None,None,None]).float().squeeze(1).squeeze(1).squeeze(1)

def norms_linf(Z):
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]

def pgd_worst_dir(model, X,y, epsilon_l_inf = 0.3, epsilon_l_2= 2.0, epsilon_l_1 = 10, 
                                alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 1.0, num_iter = 100, device = "cuda:1", k_map = 0, randomize = 0):
pert_1 = pgd_l1_topk(model, X, y, epsilon = epsilon_l_1, alpha = alpha_l_1, num_iter = 100, device = device, randomize = randomize)
pert_2 = pgd_l2(model, X, y, epsilon = epsilon_l_2, alpha = alpha_l_2, num_iter = 100, device = device, randomize = randomize)
pert_inf = pgd_linf(model, X, y, epsilon = epsilon_l_inf, alpha = alpha_l_inf, num_iter = 50, device = device, randomize = randomize)
    
    batch_size = X.shape[0]

    loss_1 = nn.CrossEntropyLoss(reduction = 'none')(model(X +pert_1), y)
    loss_2 = nn.CrossEntropyLoss(reduction = 'none')(model(X +pert_2), y)
    loss_inf = nn.CrossEntropyLoss(reduction = 'none')(model(X +pert_inf), y)

pert_1 =pert_1.view(batch_size,1,-1)
pert_2 =pert_2.view(batch_size,1,-1)
pert_inf =pert_inf.view(batch_size,1,-1)

    tensor_list = [loss_1, loss_2, loss_inf]
pert_list = pert_1,pert_2,pert_inf]
    loss_arr = torch.stack(tuple(tensor_list))
pert_arr = torch.stack(tuplepert_list))
    max_loss = loss_arr.max(dim = 0)
pert =pert_arr[max_loss[1], torch.arange(batch_size), 0]
pert =pert.view(batch_size,1, X.shape[2], X.shape[3])
    returnpert
def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter = 50, randomize = 0, restarts = 0, device = "cuda:1"):
    maxpert = torch.zeros_like(X)
    if randomize:
    pert = torch.rand_like(X, requires_grad=True)
    pert.data = pert.data * 2.0 - 1.0) * epsilon
    else:
    pert = torch.zeros_like(X, requires_grad=True)    
    for t in range(num_iter):
        output = model(Xpert)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        loss = nn.CrossEntropyLoss()(model(X +pert), y)
        loss.backward()
    pert.data = pert.data + alpha*correctpert.grad.detach().sign()).clamp(-epsilon,epsilon)
    pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1]
    pert.grad.zero_()
    maxpert =pert.detach()
    
    for i in range (restarts):
    pert = torch.rand_like(X, requires_grad=True)
    pert.data = pert.data * 2.0 - 1.0) * epsilon

        for t in range(num_iter):
            output = model(Xpert)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            loss = nn.CrossEntropyLoss()(model(X +pert), y)
            loss.backward()
        pert.data = pert.data + alpha*correctpert.grad.detach().sign()).clamp(-epsilon,epsilon)
        pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1]
        pert.grad.zero_()

        output = model(Xpert)
        incorrect = output.max(1)[1] != y
        maxpert[incorrect] =pert.detach()[incorrect]
    return maxpert



def pgd_l2(model, X, y, epsilon=2.0, alpha=0.1, num_iter = 100, restarts = 0, device = "cuda:1", randomize = 0):
    maxpert = torch.zeros_like(X)
    if random:
    pert = torch.rand_like(X, requires_grad=True) 
    pert.data *= (2.0pert.data - 1.0)
    pert.data =pert.data*epsilon/norms_l2pert.detach()) 
    else:
    pert = torch.zeros_like(X, requires_grad=True) 
    for t in range(num_iter):
        output = model(Xpert)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        loss = nn.CrossEntropyLoss()(model(X +pert), y)
        loss.backward()
    pert.data +=  correct*alphapert.grad.detach() / normspert.grad.detach())
    pert.data *=  epsilon / normspert.detach()).clamp(min=epsilon)
    pert.data =   torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1]     
    pert.grad.zero_()  

    maxpert =pert.detach()

    for k in range (restarts):
    pert = torch.rand_like(X, requires_grad=True) 
    pert.data *= (2.0pert.data - 1.0)*epsilon 
    pert.data /= normspert.detach()).clamp(min=epsilon)
        for t in range(num_iter):
            output = model(Xpert)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            loss = nn.CrossEntropyLoss()(model(X +pert), y)
            loss.backward()
        pert.data +=  correct*alphapert.grad.detach() / normspert.grad.detach())
        pert.data *= epsilon / normspert.detach()).clamp(min=epsilon)
        pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1]
        pert.grad.zero_()  

        output = model(Xpert)
        incorrect = output.max(1)[1] != y
        maxpert[incorrect] =pert.detach()[incorrect] 

    return maxpert    

def pgd_l0(model, X,y, epsilon = 10, alpha = 0.5, num_iter = 100, device = "cuda:1"):
pert = torch.zeros_like(X, requires_grad = True)
    batch_size = X.shape[0]
    for t in range (epsilon):
        loss = nn.CrossEntropyLoss()(model(X +pert), y)
        loss.backward()
        temp =pert.grad.view(batch_size, 1, -1)
        neg = pert.data != 0)
        X_curr = X +pert
        neg1 = pert.grad < 0)*(X_curr < 0.1)
        neg2 = pert.grad > 0)*(X_curr > 0.9)
        neg += neg1 + neg2
        u = neg.view(batch_size,1,-1)
        temp[u] = 0
        mypert = torch.zeros_like(X).view(batch_size, 1, -1)
        
        maxv =  temp.max(dim = 2)
        minv =  temp.min(dim = 2)
        val_max = maxv[0].view(batch_size)
        val_min = minv[0].view(batch_size)
        pos_max = maxv[1].view(batch_size)
        pos_min = minv[1].view(batch_size)
        select_max = (val_max.abs()>=val_min.abs()).float()
        select_min = (val_max.abs()<val_min.abs()).float()
        mypert[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max] = (1-X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_max])*select_max
        mypert[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min] = -X.view(batch_size, 1, -1)[torch.arange(batch_size), torch.zeros(batch_size, dtype = torch.long), pos_min]*select_min
    pert.data += mypert.view(batch_size, 1, 28, 28)
    pert.grad.zero_()
pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1]
    
    returnpert.detach()


def pgd_l1_topk(model, X,y, epsilon = 10, alpha = 1.0, num_iter = 50, k_map = 0, gap = 0.05, device = "cuda:1", restarts = 0, randomize = 0):
    gap = gap
    maxpert = torch.zeros_like(X)
    if randomize:
    pert = torch.from_numpy(np.random.laplace(size=X.shape)).float().to(device)
    pert.data =pert.data*epsilon/norms_l1pert.detach())
    pert.requires_grad = True
    else:
    pert = torch.zeros_like(X, requires_grad = True)
    alpha_l_1_default = alpha

    for t in range (num_iter):
        output = model(Xpert)
        incorrect = output.max(1)[1] != y 
        correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
        #Finding the correct examples so as to attack only them
        loss = nn.CrossEntropyLoss()(model(Xpert), y)
        loss.backward()
        if k_map == 0:
            k = random.randint(5,20)
            alpha   = (alpha_l_1_default/k)
        elif k_map == 1:
            k = random.randint(10,40)
            alpha   = (alpha_l_1_default/k)
        else:
            k = 10
            alpha = alpha_l_1_default
    pert.data += alpha*correct*l1_dir_topkpert.grad.detach(),pert.data, X, gap,k)
        if (norms_l1pert > epsilon).any():
        pert.data = proj_l1ballpert.data, epsilon, device)
    pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1] 
    pert.grad.zero_() 

    maxpert =pert.detach()
    for k in range(restarts):
    pert = torch.rand_like(X,requires_grad = True)
    pert.data = (2pert.data - 1.0)
    pert.data =pert.data*epsilon/norms_l1pert.detach())
        for t in range (num_iter):
            output = model(Xpert)
            incorrect = output.max(1)[1] != y 
            correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).float()
            loss = nn.CrossEntropyLoss()(model(Xpert), y)
            loss.backward()
            if k_map == 0:
                k = random.randint(5,20)
                alpha   = (alpha_l_1_default/k)
            elif k_map == 1:
                k = random.randint(10,40)
                alpha   = (alpha_l_1_default/k)
            else:
                k = 10
                alpha = alpha_l_1_default
        pert.data += alpha*correct*l1_dir_topkpert.grad.detach(),pert.data, X, gap,k)
            if (norms_l1pert > epsilon).any():
            pert.data = proj_l1ballpert.data, epsilon, device)
        pert.data = torch.min(torch.maxpert.detach(), -X), 1-X) # clip Xpert to [0,1] 
        pert.grad.zero_() 
        output = model(Xpert)
        incorrect = output.max(1)[1] != y
        maxpert[incorrect] =pert.detach()[incorrect]   

    return maxpert

def kthlargest(tensor, k, dim=-1):
    val, idx = tensor.topk(k, dim = dim)
    return val[:,:,-1], idx[:,:,-1]

def l1_dir_topk(grad,pert, X, gap, k = 10) :
    X_curr = X +pert
    batch_size = X.shape[0]
    channels = X.shape[1]
    pix = X.shape[2]
    neg1 = (grad < 0)*(X_curr <= gap)
    neg2 = (grad > 0)*(X_curr >= 1-gap)
    neg3 = X_curr <= 0
    neg4 = X_curr >= 1
    neg = neg1 + neg2 + neg3 + neg4
    u = neg.view(batch_size,1,-1)
    grad_check = grad.view(batch_size,1,-1)
    grad_check[u] = 0

    kval = kthlargest(grad_check.abs().float(), k, dim = 2)[0].unsqueeze(1)
    k_hot = (grad_check.abs() >= kval).float() * grad_check.sign()
    return k_hot.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=10, device = "cuda:1"):
    assert epsilon > 0
    u = x.abs()
    if (u.sum(dim = (1,2,3)) <= epsilon).all():
        return x

    y = proj_simplex(u, s=epsilon, device = device)
    y *= x.sign()
    y *= epsilon/norms_l1(y)
    return y


def proj_simplex(v, s=1, device = "cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    u = v.view(batch_size,1,-1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending = True)
    cssv = u.cumsum(dim = 2)
    vec = u * torch.arange(1, n+1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim = 2)
    w = (comp-1).cumsum(dim = 2)
    u = u + w
    rho = torch.argmax(u, dim = 2)
    rho = rho.view(batch_size)
    c = torch.FloatTensor([cssv[i,0,rho[i]] for i in range( cssv.shape[0]) ]).to(device)
    c = c-s
    theta = torch.div(c,(rho.float() + 1))
    theta = theta.view(batch_size,1,1,1)
    w = (v - theta).clamp(min=0)
    return w


def epoch(loader, lr_schedule,  model, epoch_i = 0, criterion = nn.CrossEntropyLoss(), opt=None, device = "cuda:1", stop = False):
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)

        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                        opt=None, device = "cuda:1", stop = False, stats = False, **kwargs):
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        if stats:
        pert = attack(model, X, y, device = device, batchid=i, epoch_i = epoch_i , **kwargs)
        else:
        pert = attack(model, X, y, device = device, **kwargs)

        yp = model(Xpert)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss += loss.item()*y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        if stop:
            break
        
    return train_loss / train_n, train_acc / train_n

def epoch_adversarial_saver(batch_size, loader, model, attack, epsilon, num_iter, device = "cuda:0", restarts = 10):
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    train_acc = 0
    train_n = 0

    for i,batch in enumerate(loader): 
        X,y = batch[0].to(device), batch[1].to(device)
    pert = attack(model, X, y, epsilon = epsilon, num_iter = num_iter, device = device, restarts = restarts)
        output = model(Xpert)
        loss = criterion(output, y)
        train_loss += loss.item()*y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        correct = (output.max(1)[1] == y).float()
        eps = (correct*1000 + epsilon - 0.000001).float()
        train_n += y.size(0)
        break
    return eps,  train_acc / train_n

def triple_adv(loader, lr_schedule, model, epoch_i, attack, criterion = nn.CrossEntropyLoss(), 
                    opt=None, device= "cuda:1", 
                    epsilon_l_inf = 0.3, epsilon_l_2= 2.0, epsilon_l_1 = 10, 
                    alpha_l_inf = 0.01, alpha_l_2 = 0.1, alpha_l_1 = 1.0, num_iter = 100, 
                    k_map = 0, randomize = 0):
    

    train_loss = 0
    train_acc = 0
    train_n = 0
    for i, batch in enumerate(loader):
        X,y = batch[0].to(device), batch[1].to(device)
        if opt:
            lr = lr_schedule(epoch_i + (i+1)/len(loader))
            opt.param_groups[0].update(lr=lr)

        X,y = X.to(device), y.to(device)
    pert_l_1 = pgd_l1_topk(model, X, y, device = device, epsilon = epsilon_l_1, k_map = k_map, alpha = alpha_l_1, randomize = randomize)
        yp_l_1 = model(Xpert_l_1)
    pert_l_2 = pgd_l2(model, X, y, device = device, epsilon = epsilon_l_2, alpha = alpha_l_2, randomize = randomize)
        yp_l_2 = model(Xpert_l_2)
    pert_l_inf = pgd_linf(model, X, y, device = device, epsilon = epsilon_l_inf, alpha = alpha_l_inf, randomize = randomize)
        yp_l_inf = model(Xpert_l_inf)
        y_full = torch.cat([y,y,y], dim = 0)
        yp_full = torch.cat([yp_l_1,yp_l_2,yp_l_inf], dim = 0)
        loss = nn.CrossEntropyLoss()(yp_full,y_full)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        train_loss += loss.item()*(y_full.size(0))
        train_acc += (yp_full.max(1)[1] == y_full).sum().item()
        train_n += y_full.size(0)

    return train_loss / train_n, train_acc / train_n
