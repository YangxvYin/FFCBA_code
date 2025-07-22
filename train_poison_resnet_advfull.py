import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, utils
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import pickle
from pathlib import Path
import torch.nn.functional as F
import os
import cv2
import numpy as np
from numpy.linalg import svd
from torchvision import transforms
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as compare_ssim
import random
import torchvision.models as models
import csv
import pandas as pd
from PIL import Image
import shutil
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from utils.utils import progress_bar
from autoencoders import ConditionalAutoencoder, ConditionalAutoencoder_small
from utils.dataloader import PostTensorTransform
import config
from collections import defaultdict


def makeTrainCSV(dir_root_path, dir_to_path):
    if not os.path.exists(dir_to_path):
        os.makedirs(dir_to_path)

    dir_root_children_names = os.listdir(dir_root_path)
    dict_all_class = {}

    csv_file_dir = os.path.join(dir_to_path, 'train_data1.txt')
    with open(csv_file_dir, 'w', newline='') as csvfile:
        for dir_root_children_name in dir_root_children_names:
            dir_root_children_path = os.path.join(dir_root_path, dir_root_children_name)
            if os.path.isfile(dir_root_children_path):
                break

            file_names = os.listdir(dir_root_children_path)
            for file_name in file_names:
                (shot_name, suffix) = os.path.splitext(file_name)
                if suffix.lower() in ['.jpeg', '.png', '.jpg']:
                    file_path = os.path.join(dir_root_children_path, file_name)
                    dict_all_class[file_path] = int(dir_root_children_name)

        list_train_all_class = list(dict_all_class.keys())
        random.shuffle(list_train_all_class)

        writer = csv.writer(csvfile)
        for path_train_path in list_train_all_class:
            label = dict_all_class[path_train_path]
            example = [path_train_path, label]
            writer.writerow(example)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt_dir, transform=None, loader=default_loader):
        imgs = []
        with open(txt_dir, 'r') as fn:
            for f in fn:
                f = f.strip('\n')
                words = f.split(',')
                imgs.append((words[0], int(words[1])))
        self.loader = loader
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        images, label = self.imgs[index]
        image = self.loader(images)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_indices(self):
        class_indices = {}
        for idx, (_, label) in enumerate(self.imgs):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

def select_half_data(dataset):
    class_indices = dataset.get_class_indices()
    selected_indices = []

    for indices in class_indices.values():
        half_count = len(indices) // 1000
        if half_count < 1:
            half_count = 1
        selected_indices.extend(random.sample(indices, half_count))

    return Subset(dataset, selected_indices)

def mnist_save_img(img, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    plt.imsave(path + name, img)


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if (rmse == 0):
        rmse = eps
    return 20 * math.log10(255.0 / rmse)


def ssim(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (grayScore, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("gray SSIM: {}".format(grayScore))

    (score0, diffB) = compare_ssim(B1, B2, full=True)
    (score1, diffG) = compare_ssim(G1, G2, full=True)
    (score2, diffR) = compare_ssim(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    # print("BGR average SSIM: {}".format(aveScore))

    return aveScore


def denormalize_and_convert_to_uint8(img, mean, std):
    img = img.squeeze(0)
    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)
    img = img * std_tensor + mean_tensor
    reconstruction_img = torch.permute(img, dims=[1, 2, 0])
    reconstruction_img = reconstruction_img.numpy()
    modified_block1 = reconstruction_img * 255
    modified_block1 = np.clip(modified_block1, 0, 255)
    modified_block1 = np.round(modified_block1).astype(np.uint8)
    return modified_block1


def tran(x):
    x = transforms.ToTensor()(x)
    x = torch.unsqueeze(x, 0)
    return x


def itran(reconstruction_img):
    reconstruction_img = reconstruction_img.squeeze(0)
    reconstruction_img = torch.permute(reconstruction_img, dims=[1, 2, 0])
    reconstruction_img = reconstruction_img.numpy()
    modified_block1 = reconstruction_img * 255
    modified_block1 = np.clip(modified_block1, 0, 255)
    modified_block1 = np.round(modified_block1).astype(np.uint8)
    return modified_block1


def clip_image(x, IMAGENET_MIN, IMAGENET_MAX):
    return torch.clamp(x, IMAGENET_MIN, IMAGENET_MAX)

def sample_negative_labels(label, n_classes):
    label_cpu = label.detach().cpu().numpy()
    neg_label = [np.random.choice([e for e in range(n_classes) if e != l], 1)[0] for l in label_cpu]
    neg_label = torch.tensor(np.array(neg_label))
    return neg_label


###########################################################################################################################

print('action')

###############################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = config.get_arguments().parse_args()

sample_class = 'cifar10'

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

    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(training_data):
        class_indices[label].append(idx)

    selected_indices = []
    for indices in class_indices.values():
        half_count = len(indices) //1000
        selected_indices.extend(random.sample(indices, half_count))

    training_data_1 = Subset(training_data, selected_indices)

    batch_size = 64
    train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False,
                            pin_memory=True, num_workers=8)
    train_data_1 = DataLoader(dataset=training_data_1, batch_size=batch_size, shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=8)
    test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,
                           num_workers=8)

    num_classes = 10

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]
    IMAGENET_DEFAULT_MEAN = (0.4914, 0.4822, 0.4465)
    IMAGENET_DEFAULT_STD = (0.247, 0.243, 0.261)

    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3

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

    training_data_1 = select_half_data(training_data)

    train_dir_root_path = f"/root/autodl-tmp/animals/test"
    train_dir_to_path = f"/root/autodl-tmp/animals/test_csv"
    makeTrainCSV(train_dir_root_path, train_dir_to_path)
    train_dir = os.path.join(f'/root/autodl-tmp/animals/test_csv', 'train_data1.txt')
    testing_data = MyDataset(txt_dir=train_dir, transform=transform)

    batch_size = 16
    train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False,
                            pin_memory=True, num_workers=8)
    train_data_1 = DataLoader(dataset=training_data_1, batch_size=batch_size, shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=8)
    test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,
                           num_workers=8)

    num_classes = 90

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

    opt.input_height = 224
    opt.input_width = 224
    opt.input_channel = 3

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

    training_data_1 = select_half_data(training_data)

    train_dir_root_path = "/root/autodl-tmp/imagenet100picture/val"
    train_dir_to_path = "/root/autodl-tmp/imagenet100picture/val_csv"
    makeTrainCSV(train_dir_root_path, train_dir_to_path)
    train_dir = os.path.join('/root/autodl-tmp/imagenet100picture/val_csv', 'train_data1.txt')
    testing_data = MyDataset(txt_dir=train_dir, transform=transform)

    batch_size = 32
    train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=False,
                            pin_memory=True, num_workers=8)
    train_data_1 = DataLoader(dataset=training_data_1, batch_size=batch_size, shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=8)
    test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,
                           num_workers=8)

    num_classes = 100

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    opt.input_height = 224
    opt.input_width = 224
    opt.input_channel = 3

elif sample_class == 'gtsrb':
    transform = transforms.Compose([ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.3588, 0.3177, 0.3356],
                                        std=[0.2727, 0.2562, 0.2661]
                                    ),
                                    transforms.Resize((64, 64))
                                    ])

    train_dir_root_path = "/root/autodl-tmp/gtsrbpicture/train"
    train_dir_to_path = "/root/autodl-tmp/gtsrbpicture/train_csv"
    makeTrainCSV(train_dir_root_path, train_dir_to_path)
    train_dir = os.path.join('/root/autodl-tmp/gtsrbpicture/train_csv', 'train_data1.txt')
    training_data = MyDataset(txt_dir=train_dir, transform=transform)

    training_data_1 = select_half_data(training_data)

    train_dir_root_path = "/root/autodl-tmp/gtsrbpicture/test"
    train_dir_to_path = "/root/autodl-tmp/gtsrbpicture/test_csv"
    makeTrainCSV(train_dir_root_path, train_dir_to_path)
    train_dir = os.path.join('/root/autodl-tmp/gtsrbpicture/test_csv', 'train_data1.txt')
    testing_data = MyDataset(txt_dir=train_dir, transform=transform)

    batch_size = 64
    train_data = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True,
                            num_workers=8)
    test_data = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True,
                           num_workers=8)
    train_data_1 = DataLoader(dataset=training_data_1, batch_size=batch_size, shuffle=True, drop_last=True,
                              pin_memory=True, num_workers=8)

    num_classes = 43

    mean = [0.3588, 0.3177, 0.3356]
    std = [0.2727, 0.2562, 0.2661]
    IMAGENET_DEFAULT_MEAN = (0.3588, 0.3177, 0.3356)
    IMAGENET_DEFAULT_STD = (0.2727, 0.2562, 0.2661)

    opt.input_height = 64
    opt.input_width = 64
    opt.input_channel = 3

else:
    raise ValueError(f"Unsupported sample class: {sample_class}")

model_class = 'resnet18'

if model_class == 'resnet18':
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet18
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/cleanlabel_cifar10_resnet18_change.pth'))
elif model_class == 'vgg19':
    vgg_model = models.vgg19(pretrained=True)
    in_features = vgg_model.classifier[6].in_features
    vgg_model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    model = vgg_model
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/new_animals_vgg_clean.pth'))
elif model_class == 'densenet121':
    densenet = models.densenet121(pretrained=True)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, num_classes)
    model = densenet
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/new_cifar10_densenet_clean.pth'))
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
elif model_class == 'resnet34':
    resnet50 = models.resnet34(pretrained=True)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet50
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/cleanlabel_cifar10_resnet50.pth'))
elif model_class == 'resnet101':
    resnet50 = models.resnet101(pretrained=True)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, num_classes)
    model = resnet50
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/cleanlabel_cifar10_resnet50.pth'))

elif model_class == 'vit':
    ########################################224*224########################################
    vit_model = models.vit_b_16(pretrained=True)
    # num_ftrs = vit_model.heads.in_features
    # vit_model.heads = nn.Linear(num_ftrs, 100)
    model = vit_model
    model.heads[0] = nn.Linear(model.heads[0].in_features, num_classes)
    # model.load_state_dict(torch.load('/root/autodl-tmp/model/new_cifar10_vit_clean.pth'))

else:
    raise ValueError(f"Unsupported model class: {model_class}")

model.to(device)
cost = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler_1 = StepLR(optimizer, step_size=10, gamma=0.1)

IMAGENET_MIN = ((np.array([0, 0, 0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX = ((np.array([1, 1, 1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()

if sample_class == 'cifar10':
    tgtmodel = ConditionalAutoencoder(n_classes=10, input_dim=32).to(device)
    # tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_cifar10.pth'))
    tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_cifar10.pth'))
elif sample_class == 'animals':
    tgtmodel = ConditionalAutoencoder(n_classes=90, input_dim=224).to(device)
    # tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_animals.pth'))
    tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_animals_changefeature_vit.pth'))
elif sample_class == 'imagenet100':
    tgtmodel = ConditionalAutoencoder(n_classes=100, input_dim=224).to(device)
    tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_psnr.pth'))
elif sample_class == 'gtsrb':
    tgtmodel = ConditionalAutoencoder(n_classes=43, input_dim=64).to(device)
    tgtmodel.load_state_dict(torch.load('/root/autodl-tmp/model/tgtmodel_full_target_gtsrb.pth'))
else:
    raise ValueError(f"Unsupported sample class: {sample_class}")

tgtmodel.eval()

max_noise_norm = 100 / 255
transforms = PostTensorTransform(opt).to(device)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
#     transforms.Resize((224, 224))
# ])
# yading1 = cv2.imdecode(np.fromfile('/root/birds.png', dtype=np.uint8), 1)
# yading1 = cv2.cvtColor(yading1, cv2.COLOR_BGR2RGB)
# yading_tensor = transform(yading1).unsqueeze(0).to(device)
# y_clean = torch.tensor([0]).to(device)
# noise_RG = tgtmodel(yading_tensor, y_clean)
# if torch.norm(noise_RG, p=float('inf')) > max_noise_norm:
#     noise_RG = noise_RG * (max_noise_norm / torch.norm(noise_RG, p=float('inf')))
# atkdata = clip_image(yading_tensor+noise_RG, IMAGENET_MIN, IMAGENET_MAX)

# noise_np = denormalize_and_convert_to_uint8(noise_RG[0].clone().detach().cpu(), mean, std)
# atkdata_np = denormalize_and_convert_to_uint8(atkdata[0].clone().detach().cpu(), mean, std)
# noise_np_path = f'/root/Pytorch-UNet-master/picture/noise_np_FMBA.png'
# atkdata_np_path = f'/root/Pytorch-UNet-master/picture/atkdata_np_FMBA.png'
# plt.imsave(noise_np_path, noise_np)
# plt.imsave(atkdata_np_path, atkdata_np)


feature_trigger_tensor_loaded = torch.load('feature_trigger_tensor.pt')
print(len(train_data))
print(len(test_data))
besttesting_correct = 0
epochs = 200
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0.0
    running_loss_1 = 0.0
    running_correct_1 = 0.0
    running_loss_2 = 0.0
    running_correct_2 = 0.0
    num = 0
    num_0 = 0
    model.train()
    print("Epoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)

    for X_train, y_train in train_data_1:

        X_train, y_train = X_train.to(device), y_train.to(device)
        X_train = transforms(X_train)

        noise_RG = tgtmodel(X_train, y_train)
        if torch.norm(noise_RG, p=float('inf')) > max_noise_norm:
            noise_RG = noise_RG * (max_noise_norm / torch.norm(noise_RG, p=float('inf')))
        atkdata = clip_image(X_train + noise_RG, IMAGENET_MIN, IMAGENET_MAX)

        outputs = model(atkdata)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, 1)
        # scheduler_1.step()
        running_loss_1 += loss.item()
        running_correct_1 += torch.sum(pred == y_train.data)
        running_loss_show_1 = running_loss_1 / len(training_data)
        running_correct_show_1 = 100 * running_correct_1 / len(training_data)
        progress_bar(
            num,
            20000,
            "Train Loss is: {:.4f} | Train Accuracy is: {:.4f}".format(
                running_loss_show_1, running_correct_show_1
            ), )
        num = num + 1

    num = 0
    for X_train, y_train in train_data:
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_train = transforms(X_train)

        outputs = model(X_train)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, 1)
        # scheduler_1.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
        running_loss_show = running_loss / len(training_data)
        running_correct_show = 100 * running_correct / len(training_data)
        progress_bar(
            num,
            20000,
            "Train Loss is: {:.4f} | Train Accuracy is: {:.4f}".format(
                running_loss_show, running_correct_show
            ), )
        num = num + 1

    scheduler.step()

    testing_correct = 0
    test_loss = 0
    testing_correct_1 = 0
    test_loss_1 = 0
    model.eval()
    # if model_class == 'resnet18':
    #     torch.save(resnet18.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_resnet18.pth')
    # elif model_class == 'vgg19':
    #     torch.save(vgg_model.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_vgg_change.pth')
    # elif model_class == 'densenet121':
    #     torch.save(densenet.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_densenet_changefeature.pth')
    # elif model_class == 'googlenet':
    #     torch.save(googlenet.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_googlenet.pth')
    # elif model_class == 'resnet50':
    #     torch.save(model.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_resnet50_changefeature.pth')
    # elif model_class == 'vit':
    #     torch.save(model.state_dict(),f'/root/autodl-tmp/model/cleanlabel_{sample_class}_vit_changefeature.pth')
    # else:
    #     raise ValueError(f"Unsupported model class: {model_class}")
    z = 0
    ASR_ADD = 0
    ASR1 = 0
    ASR = {}

    for X_test, y_test in test_data:
        b, n, w, h = X_test.shape
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        loss = cost(outputs, y_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct_1 += torch.sum(pred == y_test.data)
        test_loss_1 += loss.item()
    print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(
        test_loss_1 / len(testing_data),
        100 * testing_correct_1 / len(testing_data),
    ))
    #     for numnum in range(0,10):
    #         testing_correct = 0
    #         test_loss = 0
    #         testing_correct_1 = 0
    #         test_loss_1 = 0
    #         for X_test, y_test in test_data:
    #             X_test, y_test = X_test.to(device), y_test.to(device)
    #             # atktarget = sample_negative_labels(y_test, num_classes).to(device)

    #             shape = y_test.shape
    #             atktarget = torch.full(shape, numnum, dtype=y_test.dtype, device=y_test.device)

    #             noise_RG = tgtmodel(X_test, atktarget)
    #             if torch.norm(noise_RG, p=float('inf')) > max_noise_norm:
    #                 noise_RG = noise_RG * (max_noise_norm / torch.norm(noise_RG, p=float('inf')))
    #             atkdata = clip_image(X_test+noise_RG*2, IMAGENET_MIN, IMAGENET_MAX)

    #             outputs = model(atkdata)
    #             loss = cost(outputs, atktarget)
    #             _, pred = torch.max(outputs.data, 1)
    #             testing_correct += torch.sum(pred == atktarget.data)
    #             test_loss += loss.item()
    #         print('Target:', numnum)
    #         print("ATK Loss is:{:.4f}, ATK Accuracy is:{:.4f}%".format(
    #             test_loss / len(testing_data),
    #             100 * testing_correct / len(testing_data),
    #         ))
    for X_test, y_test in test_data:
        X_test, y_test = X_test.to(device), y_test.to(device)
        atktarget = sample_negative_labels(y_test, num_classes).to(device)

        noise_RG = tgtmodel(X_test, atktarget)
        if torch.norm(noise_RG, p=float('inf')) > max_noise_norm:
            noise_RG = noise_RG * (max_noise_norm / torch.norm(noise_RG, p=float('inf')))
        atkdata = clip_image(X_test + noise_RG * 2, IMAGENET_MIN, IMAGENET_MAX)

        outputs = model(atkdata)
        loss = cost(outputs, atktarget)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == atktarget.data)
        test_loss += loss.item()
    print("ATK Loss is:{:.4f}, ATK Accuracy is:{:.4f}%".format(
        test_loss / len(testing_data),
        100 * testing_correct / len(testing_data),
    ))
    atkdata_np = denormalize_and_convert_to_uint8(atkdata[0].clone().detach().cpu(), mean, std)
    atkdata_np_path = f'/root/Pytorch-UNet-master/picture/atkdata_np_test.png'
    plt.imsave(atkdata_np_path, atkdata_np)

###############################################################################################