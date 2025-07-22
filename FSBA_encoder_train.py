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
from collections import Counter
import pytorch_wavelets
import itertools

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
                if suffix in ['.JPEG', '.png', '.jpg']:
                    file_path = os.path.join(dir_root_children_path, file_name)
                    dict_all_class[file_path] = int(dir_root_children_name)
        list_train_all_class = list(dict_all_class.keys())
        random.shuffle(list_train_all_class)
        writer = csv.writer(csvfile)
        for path_train_path in list_train_all_class:
            label = dict_all_class[path_train_path]
            writer.writerow([path_train_path, label])
    print('list_train_all_class len:' + str(len(list_train_all_class)))

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset():
    def __init__(self, txt_dir, transform=None, loader=default_loader):
        self.imgs = []
        with open(txt_dir, 'r') as fn:
            for f in fn:
                f = f.strip('\n')
                words = f.split(',')
                self.imgs.append((words[0], int(words[1])))
        self.loader = loader
        self.transform = transform
        self.txt_dir = txt_dir
        random.shuffle(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        images, label = self.imgs[index]
        image = self.loader(images)
        if self.transform:
            image = self.transform(image)
        return image, label
    
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
    
    sample_class = 'imagenet100'

    if sample_class == 'cifar10':
        transform = transforms.Compose([ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.4914, 0.4822, 0.4465],
                                            std=[0.247, 0.243, 0.261]
                                        ),
                                        transforms.Resize((32, 32))
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

        seed1 = random.randint(0, 10000)
        seed2 = random.randint(0, 10000)

        np.random.seed(seed1)
        shuffled_indices_1 = np.random.permutation(len(training_data))

        np.random.seed(seed2)
        shuffled_indices_2 = np.random.permutation(len(training_data))

        shuffled_data_1 = Subset(training_data, shuffled_indices_1)
        shuffled_data_2 = Subset(training_data, shuffled_indices_2)

        batch_size = 64
        train_data = DataLoader(dataset=shuffled_data_1, batch_size=batch_size, shuffle=True, drop_last=True,
                                pin_memory=True, num_workers=8)
        test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False,
                               pin_memory=True, num_workers=8)
        train_data_1 = DataLoader(dataset=shuffled_data_2, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=True, num_workers=8)

        num_classes = 10

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
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
        makeTrainCSV(train_dir_root_path, train_dir_to_path)
        train_dir = os.path.join('/root/autodl-tmp/animals/train_csv', 'train_data1.txt')
        training_data = MyDataset(txt_dir=train_dir, transform=transform)

        train_dir_root_path = f"/root/autodl-tmp/animals/test"
        train_dir_to_path = f"/root/autodl-tmp/animals/test_csv"
        makeTrainCSV(train_dir_root_path, train_dir_to_path)
        train_dir = os.path.join(f'/root/autodl-tmp/animals/test_csv', 'train_data1.txt')
        testing_data = MyDataset(txt_dir=train_dir, transform=transform)

        seed1 = random.randint(0, 10000)
        seed2 = random.randint(0, 10000)

        np.random.seed(seed1)
        shuffled_indices_1 = np.random.permutation(len(training_data))
        shuffled_data_1 = [training_data[i] for i in shuffled_indices_1]

        np.random.seed(seed2)
        shuffled_indices_2 = np.random.permutation(len(training_data))
        shuffled_data_2 = [training_data[i] for i in shuffled_indices_2]

        batch_size = 16
        train_data = DataLoader(dataset=shuffled_data_1, batch_size=batch_size, shuffle=True, drop_last=True,
                                pin_memory=True, num_workers=8)
        test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False,
                               pin_memory=True, num_workers=8)
        train_data_1 = DataLoader(dataset=shuffled_data_2, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=True, num_workers=8)

        num_classes = 90

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
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
        makeTrainCSV(train_dir_root_path, train_dir_to_path)
        train_dir = os.path.join('/root/autodl-tmp/imagenet100picture/train_csv', 'train_data1.txt')
        training_data = MyDataset(txt_dir=train_dir, transform=transform)

        train_dir_root_path = "/root/autodl-tmp/imagenet100picture/val"
        train_dir_to_path = "/root/autodl-tmp/imagenet100picture/val_csv"
        makeTrainCSV(train_dir_root_path, train_dir_to_path)
        train_dir = os.path.join('/root/autodl-tmp/imagenet100picture/val_csv', 'train_data1.txt')
        testing_data = MyDataset(txt_dir=train_dir, transform=transform)

        seed1 = random.randint(0, 10000)
        seed2 = random.randint(0, 10000)

        np.random.seed(seed1)
        shuffled_indices_1 = np.random.permutation(len(training_data))

        np.random.seed(seed2)
        shuffled_indices_2 = np.random.permutation(len(training_data))

        shuffled_data_1 = Subset(training_data, shuffled_indices_1)
        shuffled_data_2 = Subset(training_data, shuffled_indices_2)

        batch_size = 64
        train_data = DataLoader(dataset=shuffled_data_1, batch_size=batch_size, shuffle=True, drop_last=True,
                                pin_memory=True, num_workers=8)
        test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False,
                               pin_memory=True, num_workers=8)
        train_data_1 = DataLoader(dataset=shuffled_data_2, batch_size=batch_size, shuffle=True, drop_last=True,
                                  pin_memory=True, num_workers=8)

        num_classes = 100

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    else:
        raise ValueError(f"Unsupported sample class: {sample_class}")  
    
    model_class = 'densenet121'

    if model_class == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, num_classes)
        model = resnet18
        model.load_state_dict(torch.load('/root/autodl-tmp/model/resnetclean_tran_224.pth'))
    elif model_class == 'vgg19':
        vgg_model = models.vgg19(pretrained=True)
        in_features = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = torch.nn.Linear(in_features, num_classes)
        model = vgg_model
        model.load_state_dict(torch.load('/root/autodl-tmp/model/new_animals_vgg_clean.pth'))
    elif model_class == 'densenet121':
        densenet = models.densenet121(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)
        model = densenet
        model.load_state_dict(torch.load('/root/model/new_imagenet100_densenet_clean.pth'))
    elif model_class == 'googlenet':
        googlenet = models.googlenet(pretrained=True)
        num_ftrs = googlenet.fc.in_features
        googlenet.fc = nn.Linear(num_ftrs, num_classes)
        model = googlenet
        # model.load_state_dict(torch.load('/root/autodl-tmp/model/cleanlabel_imagenet100_vgg.pth'))
    elif model_class == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, num_classes)
        model = resnet50
        # model.load_state_dict(torch.load('/root/autodl-tmp/model/cleanlabel_cifar10_resnet50.pth'))

    elif model_class == 'vit':
        vit_model = models.vit_b_16(pretrained=True)
        # num_ftrs = vit_model.heads.in_features
        # vit_model.heads = nn.Linear(num_ftrs, 100)
        model = vit_model
        model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
        model.load_state_dict(torch.load('/root/autodl-tmp/model/new_cifar10_vit_clean.pth'))
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    model_resnet = model
    model_resnet.to(device)
    model_resnet.eval()
    
    # activation = {}
    # model_resnet.avgpool.register_forward_hook(get_avgpool_activation(model_resnet))
    
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
    
#     labels = torch.cat([torch.ones(e.shape[0]) * c_num for c_num, e in embeddings.items()], dim=0)
#     print('labels.shape:', labels.shape)
#     embeddings: torch.Tensor = torch.cat([e for e in embeddings.values()], dim=0)
#     print('embeddings.shape:', embeddings.shape)

#     # Compute centroids for each target class
#     centroids = torch.stack([embeddings[labels == i].mean(dim=0) for i in range(num_classes)], dim=0)
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
        # torch.save(tensor_dict, 'saved_tensors_cifar10.pth')
        loaded_tensors = torch.load('saved_tensors_cifar10.pth')
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
        tgtmodel = ConditionalAutoencoder(n_classes=10, input_dim=32).to(device)
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        # tgtmodel.load_state_dict(torch.load('/root/model/tgtmodel_full_target_cifar10.pth'))
        # tgtoptimizer.load_state_dict(torch.load('/root/model/tgtoptimizer_full_target_cifar10.pth'))
    elif sample_class == 'animals':
        tgtmodel = ConditionalAutoencoder(n_classes=90, input_dim=224).to(device)
        tgtoptimizer = optim.Adam(tgtmodel.parameters(), lr=0.0001)
        tgtmodel.load_state_dict(torch.load('/root/model/tgtmodel_full_target_animals_changefeature.pth'))
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
    
    dwt = pytorch_wavelets.DWTForward(J=3, wave='haar', mode='zero').to(device)
    idwt = pytorch_wavelets.DWTInverse(wave='haar', mode='zero').to(device)
    
    train_data_all_iter = iter(train_data_1)
    
    for epoch in range(1, 1001):
        mean_tensor_0 = torch.tensor(mean).view(-1, 1, 1).to(device)
        std_tensor = torch.tensor(std).view(-1, 1, 1).to(device)
        
        tgtmodel.train()
        
        pbar = tqdm(enumerate(train_data), total=len(train_data), position=0, leave=True)
        running_correct = 0.0
        for batch_idx, (x_train, y_train) in pbar:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            
            # x_train_all, y_train_all = next(train_data_all_iter)   
            while True:
                try:
                    x_train_all, y_train_all = next(train_data_all_iter)
                    break
                except StopIteration:
                    train_data_all_iter = iter(train_data_1)
                
            x_train_all = x_train_all.to(device)
            y_train_all = y_train_all.to(device)
            
            Yl, Yh = dwt(x_train)
            Yl_all, Yh_all = dwt(x_train_all)
            x_train_1 = idwt((Yl, Yh_all))
            
            mean_tensor = centroids[y_train].to(device)
            
            noise = tgtmodel(x_train_1, y_train)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))
            atkdata = clip_image(x_train_1 + noise, IMAGENET_MIN, IMAGENET_MAX)
            
            tensor1_denorm = x_train_1 * std_tensor + mean_tensor_0
            tensor2_denorm = atkdata * std_tensor + mean_tensor_0
            tensor1_denorm = torch.clamp(tensor1_denorm, 0, 1)
            tensor2_denorm = torch.clamp(tensor2_denorm, 0, 1)

            psnr_value = -kornia.losses.psnr_loss(tensor1_denorm, tensor2_denorm, max_val=1.0)
            psnr_loss = (35-psnr_value)/35

            output = model_resnet(atkdata)
            features = activation['avgpool']  # (batch_size, 512, 1,1)
            features = features.squeeze()
            loss_2 = F.l1_loss(mean_tensor, features)
            
            loss_1 = criterion(output, y_train)
            loss = psnr_loss*2 + loss_1 + loss_2
            # loss = psnr_loss*2 + loss_1
            # loss = loss_1
            tgtoptimizer.zero_grad()
            loss.backward()
            tgtoptimizer.step() #this is the slowest step
            
            _, pred = torch.max(output.data, 1)
            running_correct += torch.sum(pred == y_train.data)
            running_correct_show = 100 * running_correct / len(training_data)
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_data)-1):
                pbar.set_description('Train [{}] Loss: ATK:{:.4f} | ATK Accuracy is: {:.4f}'.format(
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
                torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target_cifar10_changefeature_vit.pth')
                # torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target_cifar10_224.pth')
            elif sample_class == 'animals':
                torch.save(tgtmodel.state_dict(), f'/root/autodl-tmp/model/tgtmodel_full_target_animals_changefeature_vgg.pth')
                # torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target_animals.pth')
            elif sample_class == 'imagenet100':    
                torch.save(tgtmodel.state_dict(), f'/root/model/tgtmodel_full_target_changefeature_densenet.pth')
                # torch.save(tgtoptimizer.state_dict(), f'/root/autodl-tmp/model/tgtoptimizer_full_target.pth')
            else:
                raise ValueError(f"Unsupported sample class: {sample_class}")
        
        testing_correct = 0

        for X_test, y_test in test_data:
            X_test, y_test = X_test.to(device), y_test.to(device)

            noise = tgtmodel(X_test, y_test)
            if torch.norm(noise, p=float('inf')) > max_noise_norm:
                noise = noise * (max_noise_norm / torch.norm(noise, p=float('inf')))
            atkdata = clip_image(X_test + noise, IMAGENET_MIN, IMAGENET_MAX)
            output = model_resnet(atkdata)
            _, pred = torch.max(output.data, 1)

            testing_correct += torch.sum(pred == y_test.data)

        print("Test Accuracy is:{:.4f}%".format(100 * testing_correct / len(testing_data)))
            