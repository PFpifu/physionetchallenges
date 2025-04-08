#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

import neurokit2 as nk
from scipy.signal import resample
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import random
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from torchmetrics.classification import AUROC

from helper_code import *



################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    # num_records = len(records)
    num_records = 5000
    

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 12, 4096), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)

    # Iterate over the records.
    # for i in range(num_records):
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        features[i] = extract_features(record)
        labels[i] = load_label(record)

    print(features.shape)

    # Train the models.
    if verbose:
        print('Training the model on the data...')
    
    # A simple model on unoffical phase

    train_dataset = get_dataset(features, labels, num_records)
    learning_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Vit_model(emb_dim=768, patch_height=1, patch_width=64, image_height=12).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    auc_metric = AUROC(task="binary").to(device)


    # 执行训练和验证
    model = train(
        model, train_dataset, criterion, optimizer, 
        device, batch_size = 32, num_epochs=10, verbose = True,
        model_folder = model_folder, writer = SummaryWriter())

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


class NumpyDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化 Dataset
        :param data: NumPy 数组，表示特征数据
        :param labels: NumPy 数组，表示标签
        """
        self.data = torch.from_numpy(data).float()  # 转换为张量，并设置为 float 类型
        self.labels = torch.from_numpy(labels).float()  # 转换为张量，并设置为 long 类型

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        return self.data[idx], self.labels[idx]

def get_dataset(features, labels, num_records):

    total_indices = num_records

    # 1. 创建索引列表
    indices = list(range(total_indices))

    new_labels = np.zeros((num_records, 1), dtype = np.float64)

    for i in range(num_records):
        if labels[i] == True:
            new_labels[i] = 1

    X_train, y_train = features, new_labels

    train_dataset = NumpyDataset(X_train, y_train)
    
    return train_dataset

class Vit_model(nn.Module):
    def __init__(self, batch_size = 32, image_height=12, image_width=4096, patch_height=1, patch_width=64, num_classes=2, emb_dim=384, num_heads=12, num_layers=12, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Vit_model, self).__init__()
        
        # 计算每张图像的块数 (patches)
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        
        # 计算块的数量
        self.num_patches = (image_width // patch_width) * (image_height // patch_height)  # 每行有 image_width / patch_width 块，总共有 image_height 行
        
        # Patch embedding (线性投影)
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=emb_dim, kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))
        
        self.cls_token = nn.Parameter(torch.zeros(self.batch_size, 1, self.emb_dim))

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))  # +1 for [CLS] token
        
        # Transformer encoder layers
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(emb_dim)
        
        # 分类头 - 对于二分类，输出 1 个值
        self.classification_head = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Step 1: Patch embedding (将每个图像切分成块，并映射为高维特征)
        x = self.patch_embedding(x.unsqueeze(1))  # 将图像的通道数从 1 增加，以适应 Conv2d
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, emb_dim)
        
        # Step 2: Add positional encoding
        # cls_token = nn.Parameter(torch.zeros(x.shape[0], 1, self.emb_dim))  # [CLS] token
        x = torch.cat((self.cls_token, x), dim=1)  # Shape: (batch_size, num_patches + 1, emb_dim)
        x = x + self.positional_encoding
        
        # Step 3: Transformer encoding
        x = self.encoder(x)  # Shape: (batch_size, num_patches + 1, emb_dim)

        x = self.norm(x)
        
        # Step 4: Classification head
        return self.classification_head(x[:, 0, :]) # 取序列的最后一个特征作为输出

    def predict(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self = self.to(device)
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.sigmoid(outputs).squeeze(1)  # 二分类概率
            preds = (probs >= 0.5).float()  # 阈值为 0.5
        return preds
    
    def predict_proba(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.emb_dim))
        self = self.to(device)
        x = x.to(device)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.sigmoid(outputs).squeeze(1)  # 将 logits 转换为概率
        return probs

# 定义训练和验证函数
def train(model, train_dataset, criterion, optimizer, device, batch_size, num_epochs, writer, model_folder, verbose = True):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataset):.4f}")

    writer.close()
    return model

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']
    device = 'cuda'

    # Extract the features.
    features = extract_features(record).reshape(1, 12, 4096)
    features = torch.from_numpy(features.copy()).float()  # 关键修改点
    features = features.to(device)

    # Get the model outputs.
    binary_output = model.predict(features).item()
    probability_output = model.predict_proba(features).item()

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    new_signal = np.asarray(signal, dtype=np.float32)
    new_signal = new_signal.transpose(1, 0)
    num_samples_target = 4096
    if new_signal.shape != (12,4096):
        new_signal = np.array([resample(channel, num_samples_target) for channel in new_signal])
    new_signal = nk.ecg_clean(new_signal.reshape(-1), sampling_rate = 400).reshape(-1, 4096)
    # return np.asarray(features, dtype=np.float32)
    return new_signal

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
