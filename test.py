import foolbox
import foolbox.attacks as fa
import numpy as np
import torch
from matplotlib import pyplot as plt
import ipdb
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from mnist_funcs import *
from time import time
import argparse 
from fast_adv.attacks import DDN

parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)

device_id = params.gpu_id
batch_size = params.batch_size
choice = params.model
attack = params.attack
res = params.restarts
path = params.path
subset = params.subset
activation = params.activation

mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(device_id))


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

def net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

def net_tanh():
    return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.Tanh(), nn.Linear(1024, 10))

if activation=="tanh":
    net = net_tanh

def get_attack(attack, fmodel):
    args = []
    kwargs = {}
    # L0
    if attack == 'SAPA':
        A = fa.SaltAndPepperNoiseAttack()
    elif attack == 'PA':
        A = fa.L1BrendelBethgeAttack()

    # L2
    elif 'IGD' in attack:
        A = fa.L2BasicIterativeAttack()
    elif attack == 'AGNA':
        A = fa.L2AdditiveGaussianNoiseAttack()
    elif attack == 'BA':
        A = fa.BoundaryAttack()
    elif 'DeepFool' in attack:
        A = fa.L2DeepFoolAttack()
    elif attack == 'PAL2':
        A = fa.L2BrendelBethgeAttack()
    elif attack == "CWL2":
        A = fa.L2CarliniWagnerAttack()


    # L inf
    elif 'FGSM' in attack and not 'IFGSM' in attack:
        A = fa.FGSM()
    elif 'PGD' in attack:
        A = fa.LinfPGD()
    elif 'IGM' in attack:
        A = fa.LinfBrendelBethgeAttack()
    else:
        raise Exception('Not implemented')
    return A, 0,0,0


def test_foolbox(model_name, max_tests):
    print(max_tests)
    print (model_name)
    torch.manual_seed(0)
    try:
        model_test = net().to(device)
        model_address = model_name + ".pt"
        model_test.load_state_dict(torch.load(model_address, map_location = device))
    except:
        model_test = Model_Torch().to(device)
        model_address = model_name + ".pt"
        model_test.load_state_dict(torch.load(model_address, map_location = device))

    print(model_address)
    model_test.eval()
    fmodel = foolbox.models.PyTorchModel(model_test,
                                         bounds=(0., 1.),num_classes=10,
                                         device=device)


    if subset == 0:
        attacks_list = ['PA','SAPA']
        types_list   = [ 0    , 0 ]
    elif subset == 1:
        types_list   = [ 2  ]
        attacks_list = ['BA']
    elif subset == 2:
        attacks_list = ['IGD','AGNA','DeepFool','PAL2']
        types_list = [2,2,2,2]
    elif subset == 3 :
        attacks_list =['FGSM','PGD','IGM','CWL2']
        types_list = [3,3,3,2]
    elif subset == 4 :
        types_list   = [ 2  ]
        attacks_list = ['CWL2']
    elif subset == 5:
        types_list = [2]
        attacks_list = ['DeepFool']
    else:
        attacks_list = ['SAPA','PA','IGD','AGNA','BA','DeepFool','PAL2','CWL2','PGD','IGM']
        types_list   = [ 0    , 0  , 2   , 2    ,  2  ,  2  , 2 , 2 , 3, 3 ]

    norm_dict = {0:norms_l0_squeezed, 1:norms_l1_squeezed, 2:norms_l2_squeezed,3:norms_linf}

    for i in range(len(attacks_list)):
        restarts = res
        attack_name = attacks_list[i]
        print (attack_name )
        types = types_list[i]
        norm = norm_dict[types]
        max_check = max_tests
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

        start = time()
        output = np.ones((max_check))
        
        attack, metric, args, kwargs = get_attack(attack_name, fmodel)
        total = 0
        err = 0
        for X,y in test_loader:
            # ipdb.set_trace()
            distance = 1000*torch.ones(batch_size)
            
            start = time()
            image = X.numpy()
            label = y.numpy()
            for r in range (restarts):
                # ipdb.set_trace()
                if (attack_name == 'BA'):
                    adversarial = attack(image, labels=label,threaded_rnd=False, threaded_gen=False)
                else:
                    adversarial = attack(image, labels=label) if (attack_name !='CWL2') else attack(image, labels=label, max_iterations = 200,  learning_rate=0.025)
                adv = torch.from_numpy(adversarial)
                new_distance = norm(X-adv)
                distance[distance > new_distance] = new_distance[distance>new_distance]
            output[total:total+batch_size] = distance
            total += batch_size
            print(total, " ", attack_name, " " ,model_name, " Time taken = ", time() - start, " distance = ", distance[distance>10].shape, distance[distance>2].shape, distance[distance>0.3].shape)
            
            if (total >= max_check):
                np.save(model_name + "/" + attack_name + ".npy" ,output)
                break

        print("Time Taken = ", time() - start)

def test_pgd(model_name, clean = False):
    try:
        model = net().to(device)
        model_address = model_name + ".pt"
        model.load_state_dict(torch.load(model_address, map_location = device))
    except:
        model = Model_Torch().to(device)
        model_address = model_name + ".pt"
        model.load_state_dict(torch.load(model_address, map_location = device))
    
    print (model_name)
    test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    epoch_i = 0
    lr = None

    
    clean_loss, clean_acc = epoch(test_loader, lr, model, epoch_i, device = device)
    myprint('Clean Acc : {0:4f}'.format(clean_acc))
    total_loss, total_acc_1 = epoch_adversarial(test_loader,None,  model, epoch_i, pgd_l1_topk,device = device, stop = True, restarts = res, num_iter =100)
    myprint('Test Acc 1: {0:.4f}'.format(total_acc_1))    
    total_loss, total_acc_2 = epoch_adversarial(test_loader, None, model, epoch_i, pgd_l2, device = device, stop = True, restarts = res, num_iter = 200)
    myprint('Test Acc 2: {0:.4f}'.format(total_acc_2))    
    total_loss, total_acc_inf = epoch_adversarial(test_loader, None, model, epoch_i, pgd_linf, device = device, stop = True, restarts = res, num_iter = 100)
    myprint('Test Acc Inf: {0:.4f}'.format(total_acc_inf))    

def fast_adversarial_DDN(model_name):
    test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)    
    print(device)
    try:
        model_test = net().to(device)
        model_address = model_name + ".pt"
        model_test.load_state_dict(torch.load(model_address, map_location = device))
    except:
        model_test = Model_Torch().to(device)
        model_address = model_name + ".pt"
        model_test.load_state_dict(torch.load(model_address, map_location = device))

    model_test.eval()
    restarts = 1
    
    for i,batch in enumerate(test_batches): 
        x,y = batch[0].to(device), batch[1].to(device)
        min_norm = np.zeros((restarts, batch_size))
        try:
            attacker = DDN(steps=100, device=device)
            adv = attacker.attack(model_test, x, labels=y, targeted=False)
        except:
            attacker = DDN(steps=100, device=device)
            adv = attacker.attack(model_test, x, labels=y, targeted=False)
        delta = (adv - x)
        norm = norms(delta).squeeze(1).squeeze(1).squeeze(1).cpu().numpy() 
        min_norm[0] = norm
        min_norm = min_norm.min(axis = 0)
        np.save(model_name + "/" + "DDN" + ".npy" ,min_norm)
        print(min_norm[min_norm>2.0].shape[0]/1000) 
        break
 

def test_pgd_saver(model_name):
    eps_1 = [10]
    eps_2 = [2]
    eps_3 = [0.3]
    num_1 = [100]
    num_2 = [100]
    num_3 = [100]
    attacks_l1 = torch.ones((batch_size, 12))*1000
    attacks_l2 = torch.ones((batch_size, 12))*1000
    attacks_linf = torch.ones((batch_size, 12))*1000
    try:
        model = net().to(device)
        model_address = model_name + ".pt"
        model.load_state_dict(torch.load(model_address, map_location = device))
    except:
        model = Model_Torch().to(device)
        model_address = model_name + ".pt"
        model.load_state_dict(torch.load(model_address, map_location = device))

    for index in range(len(eps_1)):
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_1 = eps_1[index]
        n_1 = num_1[index]
        eps, total_acc_1 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l1_topk, e_1, n_1, device = device, restarts = res)
        attacks_l1[:,index] = eps
        
    attacks_l1 = torch.min(attacks_l1,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL1" + ".npy" ,attacks_l1.numpy())

    for index in range(len(eps_2)):        
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_2 = eps_2[index]
        n_2 = num_2[index]
        eps, total_acc_2 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_l2, e_2, n_2, device = device, restarts = res)
        attacks_l2[:,index] = eps
    attacks_l2 = torch.min(attacks_l2,dim = 1)[0]
    np.save(model_name + "/" + "CPGDL2" + ".npy" ,attacks_l2.numpy())

    for index in range(len(eps_3)):
        test_batches = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)
        e_3 = eps_3[index]
        n_3 = num_3[index]
        eps, total_acc_3 = epoch_adversarial_saver(batch_size, test_batches, model, pgd_linf, e_3, n_3, device = device, restarts = res)
        attacks_linf[:,index] = eps
    attacks_linf = torch.min(attacks_linf,dim = 1)[0]
    np.save(model_name + "/" + "CPGDLINF" + ".npy" ,attacks_linf.numpy())


model_list = ["LINF", "L1", "L2","AVG", "MAX"]
model_name = "Selected/{}".format(model_list[choice])

if path is not None:
    model_name = path
import os
if(not os.path.exists(model_name)):
    os.makedirs(model_name)
file = open("{0}/pgd_test.txt".format(model_name), "a")
if attack == 0:
    test_foolbox(model_name,1000)
elif attack == 1:
    test_pgd(model_name)
elif attack == 2:
    test_pgd_saver(model_name)
elif attack == 3:
    fast_adversarial_DDN(model_name)
