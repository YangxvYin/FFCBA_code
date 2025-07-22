import cv2
import numpy as np
import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, utils
from torchvision.transforms import ToTensor, transforms
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt
import math
from models import Vgg16Conv, Vgg16Deconv, Vgg16Classifier
from functools import partial
import csv
import pickle
from pathlib import Path
import cv2
from numpy.linalg import svd
import skimage.metrics as compare_ssim
import pandas as pd
from PIL import Image
import shutil
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
import random
from utils.utils import progress_bar
import psutil 
from collections import defaultdict
import torchvision.utils as vutils
from autoencoders import ConditionalAutoencoder
from tqdm.auto import tqdm
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torchvision.models as models
import lpips
import kornia

warnings.filterwarnings("ignore")
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

lpips_loss = lpips.LPIPS(net='alex')

transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]
                                ),
                                transforms.Resize((224, 224))
                                ])

def makeTrainCSV(dir_root_path, dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    dir_root_children_names = os.listdir(dir_root_path)
    #   print(dir_root_children_names)
    dict_all_class = {}
    csv_file_dir = os.path.join(dir_to_path, ('train_data1' + '.txt'))
    with open(csv_file_dir, 'w', newline='') as csvfile:
        for dir_root_children_name in dir_root_children_names:
            dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
            if os.path.isfile(dir_root_children_path):
                break
            file_names = os.listdir(dir_root_children_path)
            for file_name in file_names:
                (shot_name, suffix) = os.path.splitext(file_name)
                if suffix == '.JPEG':
                    file_path = os.path.join(dir_root_children_path, file_name)
                    dict_all_class[file_path] = int(dir_root_children_name)
        list_train_all_class = list(dict_all_class.keys())
        random.shuffle(list_train_all_class)
        for path_train_path in list_train_all_class:
            label = dict_all_class[path_train_path]
            example = []
            example.append(path_train_path)
            example.append(label)
            writer = csv.writer(csvfile)
            writer.writerow(example)
    print('list_train_all_class len:' + str(len(list_train_all_class)))

def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset():
    def __init__(self,txt_dir,transform=None,loader=default_loader):
        imgs=[]
        with open(txt_dir,'r') as fn:
            for f in fn:
                f=f.strip('\n')
                words=f.split(',')
                imgs.append((words[0],int(words[1])))
        self.loader=loader
        self.imgs=imgs
        self.transform=transform
        self.txt_dir=txt_dir
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,index):
        images,label=self.imgs[index]
        image=self.loader(images)
        image=self.transform(image)
        return image,label
    
train_dir_root_path = "/root/autodl-tmp/imagenet100picture/train/imagenet100"
train_dir_to_path = "/root/autodl-tmp/imagenet100picture/train_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)
train_dir=os.path.join('/root/autodl-tmp/imagenet100picture/train_csv', 'train_data1.txt')
training_data = MyDataset(txt_dir=train_dir,transform=transform)

train_dir_root_path = "/root/autodl-tmp/imagenet100picture/val"
train_dir_to_path = "/root/autodl-tmp/imagenet100picture/val_csv"
makeTrainCSV(train_dir_root_path,train_dir_to_path)
train_dir=os.path.join('/root/autodl-tmp/imagenet100picture/val_csv', 'train_data1.txt')
testing_data = MyDataset(txt_dir=train_dir,transform=transform)

batch_size = 64
train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)

def save_img_tensor(x):
    new_img = x.data.cpu().numpy()[0].transpose(1, 2, 0)
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    image = Image.fromarray(new_img)
    return image


def denormalize_and_convert_to_uint8(img, mean, std):
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    img = img * std_tensor + mean_tensor
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np

def consistency_loss(outputs):
    diff = outputs.unsqueeze(1) - outputs.unsqueeze(0)
    l2_distances = torch.norm(diff, dim=2)
    loss = torch.sum(l2_distances)
    return loss

def clip_image(x, IMAGENET_MIN, IMAGENET_MAX):
    return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)

def get_avgpool_activation(model):
    def hook(module, input, output):
        activation['avgpool'] = output
    return hook

def sample_negative_labels(label, n_classes):
    label_cpu = label.detach().cpu().numpy()
    neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
    neg_label = torch.tensor(np.array(neg_label))
    return neg_label

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    torch.cuda.empty_cache()
    
    # forward processing
    resnet18 = models.resnet18(pretrained=True)
    model_resnet = resnet18
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 100)
    model_resnet.load_state_dict(torch.load('/root/autodl-tmp/model/resnetclean_tran_224.pth'))
    
    # densenet = models.densenet121(pretrained=True)
    # num_ftrs = densenet.classifier.in_features
    # densenet.classifier = nn.Linear(num_ftrs, 100)
    # model = densenet
    # model.load_state_dict(torch.load('/root/model/new_imagenet100_densenet_clean.pth'))
    # model_resnet = model
    model_resnet.to(device)
    model_resnet.eval()
    
    activation = {}
    model_resnet.avgpool.register_forward_hook(get_avgpool_activation(model_resnet))
    
    loaded_tensors = torch.load('saved_tensors.pth')
    centroids = loaded_tensors['centroids']
    print(centroids.shape)
    
    lpips_loss = lpips_loss.to(device)
    
    num_classes = 100

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
    IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()
    
    n_classes = 100
    tgtmodel = ConditionalAutoencoder(n_classes=n_classes, input_dim=224).to(device)
    tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
    tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target.pth'))
#     tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_full_target.pth'))
    
    corret_best = float('inf')
    criterion = nn.CrossEntropyLoss()
    max_noise_norm = 100/255
    
    dropout_layer = torch.nn.Dropout(p=0.35).to(device)
    dropout_layer.train()
    
    for epoch in range(1, 1001):
        mean_tensor_0 = torch.tensor(mean).view(-1, 1, 1).to(device)
        std_tensor = torch.tensor(std).view(-1, 1, 1).to(device)
        
        tgtmodel.train()
        
        pbar = tqdm(enumerate(train_data), total=len(train_data), position=0, leave=True)
        running_correct = 0.0
        for batch_idx, (x_train, y_train) in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            atktarget = sample_negative_labels(y_train, num_classes).to(device)
            mean_tensor = centroids[atktarget].to(device)
            
            noise = tgtmodel(x_train, atktarget)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))  # 缩放噪声
            atkdata = clip_image(x_train + noise, IMAGENET_MIN, IMAGENET_MAX)
            
            tensor1_denorm = x_train * std_tensor + mean_tensor_0
            tensor2_denorm = atkdata * std_tensor + mean_tensor_0
            tensor1_denorm = torch.clamp(tensor1_denorm, 0, 1)
            tensor2_denorm = torch.clamp(tensor2_denorm, 0, 1)

            psnr_value = -kornia.losses.psnr_loss(tensor1_denorm, tensor2_denorm, max_val=1.0)
            psnr_loss = (35-psnr_value)/35
                
            # output_1 = model_resnet(x_train)
            # features_clean = activation['avgpool']  # (batch_size, 512, 1,1)
            # features_clean = features_clean.view(features_clean.size(0), -1)

            output = model_resnet(atkdata)
            features = activation['avgpool']  # (batch_size, 512, 1,1)
            features = features.squeeze()
        
            loss_2 = F.l1_loss(mean_tensor, features)
            
#             features_atk = activation['avgpool']  # (batch_size, 512, 1,1)
#             features_atk = features_atk.view(features_atk.size(0), -1)

#             feature_cha = features_atk - features_clean

#             # feature_cha_weak = dropout_layer(feature_cha)

#             features_atk_now = features_clean + feature_cha * 0.35
#             output = model_resnet.fc(features_atk_now)
            
            loss_1 = criterion(output, atktarget)
            loss = psnr_loss*2 + loss_1+ loss_2
            # loss = loss_1
            tgtoptimizer.zero_grad()
            loss.backward()
            tgtoptimizer.step() #this is the slowest step
            
            _, pred = torch.max(output.data, 1)
            running_correct += torch.sum(pred == atktarget.data)
            running_correct_show = 100 * running_correct / len(training_data)
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_data)-1):
                pbar.set_description('Train [{}] Loss: ATK:{:.4f} | ATK Accuracy is: {:.4f}'.format(
                epoch, loss.item(), running_correct_show))
        pbar.close()
        
        pbar = tqdm(enumerate(train_data), total=len(train_data), position=0, leave=True)
        running_correct = 0.0
        for batch_idx, (x_train, y_train) in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            mean_tensor = centroids[y_train].to(device)
            
            noise = tgtmodel(x_train, y_train)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))  # 缩放噪声
            atkdata = clip_image(x_train + noise, IMAGENET_MIN, IMAGENET_MAX)
            
#             tensor1_denorm = x_train * std_tensor + mean_tensor_0
#             tensor2_denorm = atkdata * std_tensor + mean_tensor_0
#             tensor1_denorm = torch.clamp(tensor1_denorm, 0, 1)
#             tensor2_denorm = torch.clamp(tensor2_denorm, 0, 1)

#             psnr_value = -kornia.losses.psnr_loss(tensor1_denorm, tensor2_denorm, max_val=1.0)
#             psnr_loss = (35-psnr_value)/35
            
            output = model_resnet(atkdata)
            
            features = activation['avgpool']  # (batch_size, 512, 1,1)
            features = features.squeeze()
        
            loss_2 = F.l1_loss(mean_tensor, features)
            
            loss_1 = criterion(output, y_train)
            # loss = lpips_value*0.02 + loss_1
            loss = loss_1+loss_2
            # loss = loss_1
            
            tgtoptimizer.zero_grad()
            loss.backward()
            tgtoptimizer.step() #this is the slowest step
            
            _, pred = torch.max(output.data, 1)
            running_correct += torch.sum(pred == y_train.data)
            running_correct_show = 100 * running_correct / len(training_data)
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_data)-1):
                pbar.set_description('Train [{}] Loss: test:{:.4f} | Accuracy is: {:.4f}'.format(
                epoch, loss.item(), running_correct_show))
        pbar.close()
          
        x_train_np = denormalize_and_convert_to_uint8(x_train[0].clone().detach().cpu(), mean, std)
        noise_np = denormalize_and_convert_to_uint8(noise[0].clone().detach().cpu(), mean, std)
        atkdata_np = denormalize_and_convert_to_uint8(atkdata[0].clone().detach().cpu(), mean, std)
        x_train_np_path = f'/root/Pytorch-UNet-master/picture/x_train_np.png'
        noise_np_path = f'/root/Pytorch-UNet-master/picture/noise_np.png'
        atkdata_np_path = f'/root/Pytorch-UNet-master/picture/atkdata_np.png'
        plt.imsave(x_train_np_path, x_train_np)
        plt.imsave(noise_np_path, noise_np)
        plt.imsave(atkdata_np_path, atkdata_np)
        
        # if running_correct_show < corret_best:
        #     corret_best = running_correct_show
        with open('/root/Pytorch-UNet-master/loss_best.txt', 'w') as f:
            f.write(str(corret_best))
        tgtmodel.eval()
        with torch.no_grad():
            torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target_psnr.pth')
            # torch.save(tgtoptimizer.state_dict(), f'/root/model/tgtoptimizer_full_target.pth')
        
        testing_correct = 0

        for X_test, y_test in test_data:
            X_test, y_test = X_test.to(device), y_test.to(device)

            noise = tgtmodel(X_test, y_test)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))
            atkdata = clip_image(X_test + noise*2, IMAGENET_MIN, IMAGENET_MAX)
            output = model_resnet(atkdata)
            _, pred = torch.max(output.data, 1)

            testing_correct += torch.sum(pred == y_test.data)

        print("Test Accuracy is:{:.4f}%".format(100 * testing_correct / len(testing_data)))
            