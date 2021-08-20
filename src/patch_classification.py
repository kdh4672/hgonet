import torch
# from .patch_dataset import Patch_Dataset
from patch_dataset import Patch_Dataset
from torch.utils.data import DataLoader
import sys
import torch.nn as nn
# from .dataset import Dataset
from resnet import ResNet50
from adamp import AdamP
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_dataset = Patch_Dataset('/home/ubuntu/database/database/sun/train/','/home/ubuntu/database/edge-connect/gt/gt_json/train-window-double_16_4.json')
valid_dataset = Patch_Dataset('/home/ubuntu/database/database/sun/val/','/home/ubuntu/database/edge-connect/gt/gt_json/val-window-double_16_4.json')

def cal_candidate(window_width,stride):
    candidate_number = int(2 * ((128 - window_width)/stride + 1))
    return candidate_number

def train(train_dataset,valid_dataset):

    train_loader = DataLoader(
        dataset= train_dataset,
        batch_size= 64,
        num_workers=0,
        shuffle=True
    )
    valid_loader = DataLoader(
    dataset= valid_dataset,
    batch_size= 64,
    num_workers=0,
    shuffle=True
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    num_classes = 2 * cal_candidate(16,4) # l , r 각각 필요하므로
    print(num_classes)
    model = ResNet50(num_classes)
    model.load_state_dict(torch.load('/home/ubuntu/database/edge-connect/patch_models/best.pth'))
    model.to(device)
    # loss_fn = nn.CrossEntropsyLoss() #criterian
    loss_fn = nn.MSELoss() #criterian 26 [0.1,0.23,0.43,0.34,]
    learning_rate = 0.001
    optimizer = AdamP(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-5)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    epoch = 40
    l_train_accuracy = []
    r_train_accuracy = []
    l_valid_accuracy = []
    r_valid_accuracy = []
    best_score = 0.0
    print('training 시작..')
    for epoch in range(epoch):
        count = 0
        train_loss = 0
        for items,l_prob,r_prob in train_loader:
            items = items.to(device)
            prob = torch.cat((l_prob,r_prob),dim=1)
            prob = prob.to(device)
            model.zero_grad()
            logits = model(items)
            logits_len = logits.shape[1]
            l_logits = logits[:,:logits_len//2]
            r_logits = logits[:,logits_len//2:]
            loss = loss_fn(logits,prob)
            l_gt = torch.argmax(prob[:,:logits_len//2], dim=1).flatten()
            r_gt = torch.argmax(prob[:,logits_len//2:], dim=1).flatten()
            l_pred = torch.argmax(l_logits, dim=1).flatten()
            r_pred = torch.argmax(r_logits, dim=1).flatten()
            l_accuracy = (l_pred == l_gt).cpu().numpy().mean()
            r_accuracy = (r_pred == r_gt).cpu().numpy().mean()
            loss.backward()
            optimizer.step()
            l_train_accuracy.append(l_accuracy)
            r_train_accuracy.append(r_accuracy)
            count+=1
            train_loss += loss.item()
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for items,l_prob,r_prob in valid_loader:
                items = items.to(device)
                prob = torch.cat((l_prob,r_prob),dim=1)
                prob = prob.to(device)
                logits = model(items)
                logits_len = logits.shape[1]
                l_logits = logits[:,:logits_len//2]
                r_logits = logits[:,logits_len//2:]
                loss = loss_fn(logits,prob)
                l_gt = torch.argmax(prob[:,:logits_len//2], dim=1).flatten()
                r_gt = torch.argmax(prob[:,logits_len//2:], dim=1).flatten()
                l_pred = torch.argmax(l_logits, dim=1).flatten()
                r_pred = torch.argmax(r_logits, dim=1).flatten()
                l_accuracy = (l_pred == l_gt).cpu().numpy().mean()
                r_accuracy = (r_pred == r_gt).cpu().numpy().mean()
                l_valid_accuracy.append(l_accuracy)
                r_valid_accuracy.append(r_accuracy)
                count+=1
                valid_loss += loss.item()
        model.train()
        print('epoch:{}'.format(epoch),"train_loss:{}".format(train_loss/len(train_loader)),"l_train_accuracy:{}".format(round(np.mean(l_train_accuracy),4)),"r_train_accuracy:{}".format(round(np.mean(r_train_accuracy),4)),"tot_train_accuracy:{}".format(round((np.mean(l_train_accuracy)+np.mean(r_train_accuracy))/2,4)))
        print("valid_loss:{}".format(valid_loss/len(valid_loader)),"l_valid_accuracy:{}".format(round(np.mean(l_valid_accuracy),4)),"r_valid_accuracy:{}".format(round(np.mean(r_valid_accuracy),4)),"tot_val_accuracy:{}".format(round((np.mean(l_valid_accuracy)+np.mean(r_valid_accuracy))/2,4)))
        if best_score < (np.mean(l_valid_accuracy)+np.mean(r_valid_accuracy))/2:
            torch.save(model.state_dict(), '/home/ubuntu/database/edge-connect/patch_models/{}.pth'.format('best'))
            
            print('model_save')
            best_score = (np.mean(l_valid_accuracy)+np.mean(r_valid_accuracy))/2
        scheduler.step()
train(train_dataset,valid_dataset)