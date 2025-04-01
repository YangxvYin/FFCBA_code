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

def makeTrainCSV(dir_root_path, dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)
    dir_root_children_names = os.listdir(dir_root_path)
    #   print(dir_root_children_names)
    dict_all_class = {}
    # 每一个类别的dict：{path,label}
    csv_file_dir = os.path.join(dir_to_path, ('train_data1' + '.txt'))
    with open(csv_file_dir, 'w', newline='') as csvfile:
        for dir_root_children_name in dir_root_children_names:
            dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
            if os.path.isfile(dir_root_children_path):
                break
            file_names = os.listdir(dir_root_children_path)
            for file_name in file_names:
                (shot_name, suffix) = os.path.splitext(file_name)
                if suffix == '.JPEG' or suffix == '.png' or suffix == '.jpg':
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
    print('训练集生成的csv文件完毕')
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
    
def save_img_tensor(x):
    new_img = x.data.cpu().numpy()[0].transpose(1, 2, 0)
    # 归一化图像
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    image = Image.fromarray(new_img)
    return image


def denormalize_and_convert_to_uint8(img, mean, std):
    # 将均值和标准差转换为张量，并调整形状
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    # 逆向标准化
    img = img * std_tensor + mean_tensor
    # 限制像素值在 [0, 1] 范围内
    img = torch.clamp(img, 0, 1)
    # 转换为 NumPy 数组并调整形状
    img_np = img.permute(1, 2, 0).numpy()
    # 转换为 uint8 类型
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
    
    sample_class = 'animals'
    
    if sample_class == 'cifar10':
        transform = transforms.Compose([ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.247, 0.243, 0.261]
                                    ),
                                    transforms.Resize((224, 224))
                                    ])
        training_data = datasets.CIFAR10(
            root='/root/autodl-tmp/cifar102/',
            train=True,
            download=True,
            transform=transform,
        )
        testing_data = datasets.CIFAR10(
            root='/root/autodl-tmp/cifar102/',
            train=False,
            download=True,
            transform=transform,
        )
        
        batch_size = 64
        train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
        test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
        
        num_classes = 10

        mean=[0.4914, 0.4822, 0.4465]
        std=[0.247, 0.243, 0.261]
        IMAGENET_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
        IMAGENET_DEFAULT_STD = (0.247, 0.243, 0.261)
        
    elif sample_class == 'animals':
        transform = transforms.Compose([ToTensor(),
                                transforms.Normalize(
                                    mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5]
                                ),
                                transforms.Resize((224, 224))
                                ])
        train_dir_root_path = "/root/autodl-tmp/animals/train"
        train_dir_to_path = "/root/autodl-tmp/animals/train_csv"
        makeTrainCSV(train_dir_root_path,train_dir_to_path)
        train_dir=os.path.join('/root/autodl-tmp/animals/train_csv', 'train_data1.txt')
        training_data = MyDataset(txt_dir=train_dir,transform=transform)

        train_dir_root_path = f"/root/autodl-tmp/animals/test"
        train_dir_to_path = f"/root/autodl-tmp/animals/test_csv"
        makeTrainCSV(train_dir_root_path,train_dir_to_path)
        train_dir=os.path.join(f'/root/autodl-tmp/animals/test_csv', 'train_data1.txt')
        testing_data = MyDataset(txt_dir=train_dir,transform=transform)
        
        batch_size = 64
        train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
        test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
        
        num_classes = 90

        mean=[0.5, 0.5, 0.5]
        std=[0.5, 0.5, 0.5]
        IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
        
    elif sample_class == 'imagenet100':
        transform = transforms.Compose([ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]
                                        ),
                                        transforms.Resize((224, 224))
                                        ])
        
        train_dir_root_path = "/root/autodl-tmp/imagenet100picture/train/imagenet100"
        train_dir_to_path = "/root/autodl-tmp/imagenet100picture/train_csv"
        makeTrainCSV(train_dir_root_path,train_dir_to_path)
        # 图像预处理的转换操作，对图像进行标准化处理，让图像能够适应模型的输入要求
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
        
        num_classes = 100
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    else:
        raise ValueError(f"Unsupported sample class: {sample_class}")  
        
    # forward processing
    # resnet18 = models.resnet18(pretrained=True)
    # model_resnet = resnet18
    # num_ftrs = model_resnet.fc.in_features # 获取低级特征维度 
    # # model_resnet.fc = nn.Linear(num_ftrs, 100) # 替换新的输出层 
    # # model_resnet.load_state_dict(torch.load('/root/model/resnetclean_tran_224.pth'))
    # if sample_class == 'cifar10':
    #     model_resnet.fc = nn.Linear(num_ftrs, 10) # 替换新的输出层 
    #     model_resnet.load_state_dict(torch.load('/root/model/new_cifar10_resnet_clean.pth'))
    # elif sample_class == 'animals':
    #     model_resnet.fc = nn.Linear(num_ftrs, 90) # 替换新的输出层 
    #     model_resnet.load_state_dict(torch.load('/root/model/new_animals_resnetclean.pth'))
    # elif sample_class == 'imagenet100':    
    #     model_resnet.fc = nn.Linear(num_ftrs, 100) # 替换新的输出层 
    #     model_resnet.load_state_dict(torch.load('/root/model/resnetclean_tran_224.pth'))
    # else:
    #     raise ValueError(f"Unsupported sample class: {sample_class}")
    
    densenet = models.densenet121(pretrained=True)
    # 获取低级特征维度
    num_ftrs = densenet.classifier.in_features
    # 替换新的输出层
    densenet.classifier = nn.Linear(num_ftrs, num_classes)
    model_resnet = densenet
    model_resnet.load_state_dict(torch.load('/root/autodl-tmp/model/new_animals_densenet_clean.pth'))
    
    model_resnet.to(device)
    model_resnet.eval()
    
#     activation = {}
#     # 注册钩子到平均池化层
#     model_resnet.avgpool.register_forward_hook(get_avgpool_activation(model_resnet))
    
#     with torch.no_grad():
#         pbar = tqdm(train_data, disable=True)
#         embeddings = {}
#         num_n = 0
#         for x, y in pbar:
#             x, y = x.to(device), y.to(device)
#             output = model_resnet(x)
#             features = activation['avgpool']  # (batch_size, 512)
#             for y_i, feature in zip(y, features):
#                 feature = feature.squeeze()
#                 embeddings[y_i.item()] = embeddings.setdefault(y_i.item(), []) + [feature.cpu().detach()]
#             progress_bar(num_n, 2033)
#             num_n += 1
#         embeddings = {c: torch.stack(x, 0) for c, x in embeddings.items()}
    
#     labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0) # 每个类别的特征向量数量乘以对应的类别编号，并将结果连接成一个张量 labels
#     print('labels.shape:', labels.shape)
#     embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)  # 所有特征向量连成一个张量
#     print('embeddings.shape:', embeddings.shape)

#     # Compute centroids for each target class
#     centroids = torch.stack([embeddings[labels == i].mean(dim=0) for i in range(num_classes)], dim=0) # 循环每个类别的特征向量并计算其质心，并将每个类别的质心连成一个张量
#     print('centroids.shape:', centroids.shape)
    
#     radii = {}
#     for i in range(num_classes):
#         class_embeddings = embeddings[labels == i]
#         centroid = centroids[i]
#         distances = torch.norm(class_embeddings - centroid, dim=1)
#         max_distance = distances.max().item()
#         radii[i] = max_distance
#     radii_tensor = torch.tensor(list(radii.values()))
#     print('radii_tensor.shape:', radii_tensor.shape)
    
#     tensor_dict = {
#         'embeddings': embeddings,
#         'labels': labels,
#         'centroids': centroids,
#         'radii_tensor': radii_tensor
#     }

    if sample_class == 'cifar10':
        # torch.save(tensor_dict, 'saved_tensors_cifar10_224.pth')
        loaded_tensors = torch.load('saved_tensors_cifar10_224.pth')
    elif sample_class == 'animals':
        # torch.save(tensor_dict, 'saved_tensors_animals.pth')
        loaded_tensors = torch.load('saved_tensors_animals.pth')
    elif sample_class == 'imagenet100':    
        # torch.save(tensor_dict, 'saved_tensors.pth')
        loaded_tensors = torch.load('saved_tensors.pth')
    else:
        raise ValueError(f"Unsupported sample class: {sample_class}")
        
    # loaded_tensors = torch.load('saved_tensors.pth')
    centroids = loaded_tensors['centroids']
    print(centroids.shape)
    
    lpips_loss = lpips_loss.to(device)
    
    IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
    IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()
    
#     n_classes = 100
#     tgtmodel = ConditionalAutoencoder(n_classes=n_classes, input_dim=224).to(device)
#     tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
# #     tgtmodel.load_state_dict(torch.load('/root/model/tgtmodel_full_target.pth'))
# #     tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_full_target.pth'))

    if sample_class == 'cifar10':
        tgtmodel = ConditionalAutoencoder(n_classes=10, input_dim=224).to(device)
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        # tgtmodel.load_state_dict(torch.load('/root/model/tgtmodel_full_target_cifar10.pth'))
        # tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_full_target_cifar10.pth'))
    elif sample_class == 'animals':
        tgtmodel = ConditionalAutoencoder(n_classes=90, input_dim=224).to(device)
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_animals.pth'))
        # tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_res_singlet_80.pth'))
    elif sample_class == 'imagenet100':    
        tgtmodel = ConditionalAutoencoder(n_classes=100, input_dim=224).to(device)
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        # tgtmodel.load_state_dict(torch.load('/root/model/tgtmodel_res_singlet_80.pth'))
        # tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_res_singlet_80.pth'))
    else:
        raise ValueError(f"Unsupported sample class: {sample_class}")
    
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
            
            # noise_rand = torch.randn_like(x_train).to(device)
            
            atktarget = sample_negative_labels(y_train, num_classes).to(device)
            mean_tensor = centroids[atktarget].to(device)
            
            noise = tgtmodel(x_train, atktarget)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))  # 缩放噪声
            atkdata = clip_image(x_train + noise, IMAGENET_MIN, IMAGENET_MAX)
            # atkdata_1 = atkdata + noise_rand
            
            tensor1_denorm = x_train * std_tensor + mean_tensor_0
            tensor2_denorm = atkdata * std_tensor + mean_tensor_0
            tensor1_denorm = torch.clamp(tensor1_denorm, 0, 1)
            tensor2_denorm = torch.clamp(tensor2_denorm, 0, 1)

            # 计算 PSNR
            psnr_value = -kornia.losses.psnr_loss(tensor1_denorm, tensor2_denorm, max_val=1.0)
            psnr_loss = (35-psnr_value)/35

            output = model_resnet(atkdata)
            # features = activation['avgpool']  # (batch_size, 512, 1,1)
            # features = features.squeeze()
            # loss_2 = F.l1_loss(mean_tensor, features)
            
            loss_1 = criterion(output, atktarget)
            # loss = psnr_loss*0.1 + loss_1 + loss_2
            loss = psnr_loss*0.1 + loss_1
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
            
            output = model_resnet(atkdata)
            
#             features = activation['avgpool']  # (batch_size, 512, 1,1)
#             features = features.squeeze()
        
#             loss_2 = F.l1_loss(mean_tensor, features)
            
            loss_1 = criterion(output, y_train)
            # loss = lpips_value*0.02 + loss_1
            # loss = loss_1+loss_2
            loss = loss_1
            
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
            # torch.save(tgtmodel.state_dict(), f'/root/model/tgtmodel_full_target.pth')
            # torch.save(tgtoptimizer.state_dict(), f'/root/model/tgtoptimizer_full_target.pth')
            if sample_class == 'cifar10':
                torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target_cifar10_224.pth')
                torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target_cifar10_224.pth')
            elif sample_class == 'animals':
                torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target_animals_densenet.pth')
                torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target_animals.pth')
            elif sample_class == 'imagenet100':    
                torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target.pth')
                torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target.pth')
            else:
                raise ValueError(f"Unsupported sample class: {sample_class}")
        
        testing_correct = 0

        for X_test, y_test in test_data:
            X_test, y_test = X_test.to(device), y_test.to(device)

            noise = tgtmodel(X_test, y_test)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))  # 缩放噪声
            atkdata = clip_image(X_test + noise, IMAGENET_MIN, IMAGENET_MAX)
            output = model_resnet(atkdata)
            _, pred = torch.max(output.data, 1)

            testing_correct += torch.sum(pred == y_test.data)

        print("Test Accuracy is:{:.4f}%".format(100 * testing_correct / len(testing_data)))
            