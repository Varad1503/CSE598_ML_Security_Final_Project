from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import time
import argparse
from mnist_funcs import *


def trainer(params, device_id, batch_size, choice, alpha_l_1, alpha_l_2, alpha_l_inf, num_iter, epochs, epsilon_l_1, epsilon_l_2, epsilon_l_inf, lr_mode, smallest_adv, n, opt_type, lr_max, resume, resume_iter,seed, randomize, k_map):

    mnist_train = datasets.MNIST("../../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = 1000, shuffle=False)


    device = torch.device("cuda:{0}".format(device_id) if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.shape[0], -1)    

    def net():
        return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.ReLU(), nn.Linear(1024, 10))

    def net_tanh():
        return nn.Sequential(nn.Conv2d(1, 32, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, 5, padding = 2), nn.Tanh(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(7*7*64, 1024), nn.Tanh(), nn.Linear(1024, 10))

    attack_list = [ pgd_linf ,  pgd_l1_topk,   pgd_l2 ,  msd_v0 ,  triple_adv ,  pgd_worst_dir, msd_v0]
    attack_name = ["pgd_linf", "pgd_l1_topk", "pgd_l2", "triple_adv", "pgd_worst_dir"]
    folder_name = ["LINF", "L1", "L2", "AVG", "MAX"]


    def myprint(a):
        print(a)
        file.write(a)
        file.write("\n")
        file.flush()

    attack = attack_list[choice]
    name = attack_name[choice]
    folder = folder_name[choice]

    print (name)
    criterion = nn.CrossEntropyLoss()

    root = f"Models/{folder_name[choice]}"
    import glob, os, json
    num = n
    model_dir = f"/content/drive/MyDrive/MNIST/model_{num}"

    if(not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    file = open(f"{model_dir}/logs.txt", "a")    
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(params.__dict__, f, indent=2)

    lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs], [0, lr_max, 0])[0]
    
    if lr_mode != None:
    	if lr_mode == 1:
    		lr_schedule = lambda t: np.interp([t], [0, 3, 10, epochs], [0, 0.05, 0.001, 0.0001])[0]
    	elif lr_mode == 2:
    		lr_schedule = lambda t: np.interp([t], [0, epochs*2//5, epochs*4//5, epochs], [0, lr_max, lr_max/10, 0])[0]

    if activation == "tanh":
        model = net_tanh().to(device)
    else:
        model = net().to(device)

    if opt_type == "SGD":
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=0.1)
    t_start = 1

    if resume:
        location = f"{model_dir}/iter_{str(resume_iter)}.pt"
        t_start = resume_iter + 1
        model.load_state_dict(torch.load(location, map_location = device))

    for t in range(t_start,epochs+1):
        start = time.time()
        print ("Learning Rate = ", lr_schedule(t))
        if choice == 6:
            train_loss, train_acc = epoch(train_loader, lr_schedule, model, epoch_i = t, opt = opt, device = device)
        elif choice == 4:
            train_loss, train_acc = triple_adv(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
            												opt = opt, device = device, k_map = k_map, 
            												alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
            												num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, 
            												epsilon_l_inf = epsilon_l_inf, randomize = randomize)
        elif choice in [3]:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
                                                            opt = opt, device = device, k_map = k_map, 
                                                            alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
                                                            num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, 
                                                            epsilon_l_inf = epsilon_l_inf, msd_init = msd_initialization, randomize = randomize)
        elif choice in [5]:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, 
                                                            opt = opt, device = device, k_map = k_map, 
                                                            alpha_l_inf = alpha_l_inf, alpha_l_2 = alpha_l_2, alpha_l_1 = alpha_l_1, 
                                                            num_iter = num_iter, epsilon_l_1 = epsilon_l_1, epsilon_l_2 = epsilon_l_2, epsilon_l_inf = epsilon_l_inf, randomize = randomize)
        
        elif choice == 1:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device, k_map = k_map, randomize = randomize)
        else:
            train_loss, train_acc = epoch_adversarial(train_loader, lr_schedule, model, epoch_i = t, attack = attack, opt = opt, device = device, randomize = randomize)

        test_loss, test_acc = epoch(test_loader, lr_schedule,  model, epoch_i = t,  device = device, stop = True)
        linf_loss, linf_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_linf, device = device, stop = True, epsilon = epsilon_l_inf)
        l2_loss, l2_acc = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l2, device = device, stop = True, epsilon = epsilon_l_2)
        l1_loss, l1_acc_topk = epoch_adversarial(test_loader, lr_schedule,  model, epoch_i = t, attack = pgd_l1_topk, device = device, stop = True, epsilon = epsilon_l_1)
        time_elapsed = time.time()-start
        myprint(f'Epoch: {t}, Loss: {train_loss:.4f} Train : {train_acc:.4f} Clean : {test_acc:.4f}, Test  1: {l1_acc_topk:.4f}, Test  2: {l2_acc:.4f}, Test  inf: {linf_acc:.4f}, Time: {time_elapsed:.1f}, Model: {model_dir}')    
        
        torch.save(model.state_dict(), "{0}/iter_{1}.pt".format(model_dir, str(t)))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Training for MNIST', formatter_class=argparse.RawTextHelpFormatter)
    params = parser.parse_args()
    device_id = params.gpu_id
    batch_size = params.batch_size
    choice = params.model
    alpha_l_1 = params.alpha_l_1
    alpha_l_2 = params.alpha_l_2
    alpha_l_inf = params.alpha_l_inf
    num_iter = params.num_iter
    epochs = params.epochs
    epsilon_l_1 = params.epsilon_l_1
    epsilon_l_2 = params.epsilon_l_2
    epsilon_l_inf = params.epsilon_l_inf
    lr_mode = params.lr_mode
    msd_initialization = params.msd_initialization
    smallest_adv = params.smallest_adv
    n = params.model_id
    opt_type = params.opt_type
    lr_max = params.lr_max
    resume = params.resume
    resume_iter = params.resume_iter
    activation = params.activation
    seed = params.seed
    randomize = params.randomize
    k_map = params.k_map
    trainer(params, device_id, batch_size, choice, alpha_l_1, alpha_l_2, alpha_l_inf, num_iter, epochs, epsilon_l_1, epsilon_l_2, epsilon_l_inf, lr_mode, msd_initialization, smallest_adv, n, opt_type, lr_max, resume, resume_iter, seed, randomize, k_map)
