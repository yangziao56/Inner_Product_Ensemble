import numpy as np 
import torchvision.transforms as transforms
from .cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import logging


train_cifar10_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_cifar100_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_cifar100_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_animal10n_transform = transforms.Compose([
    #transforms.Resize(32), # 首先将图片大小调整为256x256
    #transforms.RandomCrop(32, padding=32),  # 然后进行随机裁剪到224x224
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # 根据ResNet的通用归一化值
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_animal10n_transform = transforms.Compose([
    #transforms.Resize(32),
    #transforms.Resize(224),  # 直接调整测试图片到224x224
    transforms.ToTensor(),
    # 同样使用ResNet的通用归一化值
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Train transform for Food101N
train_food101N_transform = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Test transform for Food101N
test_food101N_transform = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# class Animal10NDataset(Dataset):
#     def __init__(self, directory, transform=None):
#         self.directory = directory
#         self.transform = transform
#         self.samples = []

#         # 遍历目录，读取所有文件
#         for filename in os.listdir(directory):
#             if not filename.endswith('.jpg'):  # 假设所有图片都是jpg格式
#                 continue
#             label = int(filename.split('_')[0])  # 标签是文件名的第一部分
#             self.samples.append((os.path.join(directory, filename), label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         image = Image.open(path).convert('RGB')  # 确保图像为RGB格式

#         if self.transform:
#             image = self.transform(image)

#         return image, label



class Animal10NDataset(Dataset):
    """
    Animal10N Dataset.
    
    This dataset class is designed to load images from a directory, assuming
    that the images are named with their label as the prefix. It provides
    functionalities similar to CIFAR100 but tailored for the Animal10N dataset.
    
    Args:
        root (string): Root directory of the dataset where images are stored.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g., `transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.labels = []  # 定义一个空列表来存储标签

        # Load all image files, assuming that they are in jpg format and the label is part of the filename
        for filename in os.listdir(self.root):
            if not filename.endswith('.jpg'):  # Assuming all images are in jpg format
                continue
            label = int(filename.split('_')[0])  # Assuming the label is the first part of the filename
            self.samples.append((os.path.join(self.root, filename), label))
            self.labels.append(label)  # 同时将标签添加到 self.labels 中

    def __len__(self):
        # Returns the total number of samples in the dataset
        return len(self.samples)

    def __getitem__(self, idx):
        # Fetch the image and label at the specified index
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')  # Ensure the image is in RGB format

        # Apply the specified transformations to the image and label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, idx  # 返回图像、标签和索引
    @property
    def train_labels(self):
        return self.labels  # 提供一个属性来访问标签

# Image transformations for the Food101N dataset
train_food101N_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_food101N_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Food101NDataset(Dataset):
    def __init__(self, image_dir, meta_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 通过skiprows参数跳过第一行
        self.meta_data = pd.read_csv(meta_file, sep='\t', names=['img_key', 'label'], skiprows=1)
        #self.meta_data = pd.read_csv(meta_file, sep='\t', names=['img_key', 'label'], header=None)
        print("Metadata loaded, example data:", self.meta_data.head())  # 打印前几行数据以检查其结构
        self.labels = self.meta_data['label'].values
        self.missing_files = []  # 用于存储找不到的文件

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        img_key = self.meta_data.iloc[idx, 0]  # 获取图像键，例如 'apple_pie/0001.jpg'
        #print("Loading image with key:", img_key)  # 打印正在加载的图像键
        img_path = os.path.join(self.image_dir, img_key)  # 正确构建完整路径
        #print("Attempting to load image:", img_path)  # 打印尝试加载的图片路径

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            logging.warning(f"File not found: {img_path}, skipping...")
            self.missing_files.append(img_path)
            return None, None  # 或者返回一个特定的占位符图像和标签

        label = self.meta_data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label, idx


    
    @property
    def train_labels(self):
        # Property to access labels
        return self.labels


def input_dataset(dataset, noise_type, noise_path, is_human):
    if dataset == 'cifar10':
        train_dataset = CIFAR10(root='~/data/',
                                download=True,  
                                train=True, 
                                transform = train_cifar10_transform,
                                noise_type = noise_type,
                                noise_path = noise_path, is_human=is_human
                           )
        test_dataset = CIFAR10(root='~/data/',
                                download=False,  
                                train=False, 
                                transform = test_cifar10_transform,
                                noise_type=noise_path #'clean'
                          )
        num_classes = 10
        num_training_samples = 50000
    elif dataset == 'cifar100':
        train_dataset = CIFAR100(root='~/data/',
                                download=True,  
                                train=True, 
                                transform=train_cifar100_transform,
                                noise_type=noise_type,
                                noise_path = noise_path, is_human=is_human
                            )
        test_dataset = CIFAR100(root='~/data/',
                                download=False,  
                                train=False, 
                                transform=test_cifar100_transform,
                                noise_type=noise_path #'clean'
                            )
        num_classes = 100
        num_training_samples = 50000

    elif dataset == 'Animal10N':
        train_dir = os.path.expanduser('data/Animal10N/train/')
        test_dir = os.path.expanduser('data/Animal10N/test/')
        train_dataset = Animal10NDataset(root=train_dir,
                                         transform=train_animal10n_transform)
        test_dataset = Animal10NDataset(root=test_dir,
                                        transform=test_animal10n_transform)
        num_classes = 10
        num_training_samples = 50000  # 根据Animal10N的实际情况
    
    elif dataset == 'Food101N':
        train_dir = os.path.expanduser('data/Food101N/')
        test_dir = os.path.expanduser('data/Food101N/')
        train_meta = os.path.expanduser('data/Food101N/meta/verified_train.tsv')
        test_meta = os.path.expanduser('data/Food101N/meta/verified_val.tsv')
        
        train_dataset = Food101NDataset(image_dir=train_dir,
                                        meta_file=train_meta,
                                        transform=train_food101N_transform)
        test_dataset = Food101NDataset(image_dir=test_dir,
                                    meta_file=test_meta,
                                    transform=test_food101N_transform)
        num_classes = 101
        num_training_samples = 75750  # Food101N 的训练样本数, 请根据实际情况调整

    return train_dataset, test_dataset, num_classes, num_training_samples








