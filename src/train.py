import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix


import numpy as np
import pandas as pd
import random


import PIL
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


from dataset import MelanomaDataset
from modules import ResNetModel


# from torch.utils.tensorboard import SummaryWriter
import time

""" Initialization"""
nfolds = 4
SEED = 45
split = 0
epochs = 30
DEBUG = True

train = '../data/jpeg/train'
labels = '../data/train.csv'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available. Training on CPU...")
else:
    print("CUDA is available. Training on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Split dataset in n Folds"""
df = pd.read_csv(labels)

splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
splits = list(splits.split(df, df.target))
folds_splits = np.zeros(len(df)).astype(np.int)

for i in range(nfolds):
    folds_splits[splits[i][1]] = i
df["split"] = folds_splits
print(df.head())

print("Previous Length", len(df))
if DEBUG:
    df = df[:500]
print("Usable Length", len(df))

""" Dataset """

train_transform = transforms.Compose([
#                         transforms.RandomResizedCrop(
#                             size=256, scale=(0.8, 1.0)),
#                         transforms.HorizontalFlip(),
#                         transforms.VerticalFlip(),
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225])
                        ])


valid_transform=transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225])])

df_train=df[df['split'] != split]
df_valid=df[df['split'] == split]

t_dataset=MelanomaDataset(df=df_train, imfolder=train,
                          train=True, transforms=train_transform)
v_dataset=MelanomaDataset(df=df_valid, imfolder=train,
                          train=True, transforms=valid_transform)

print('Length of training and validation set are {} {}'.format(
    len(t_dataset), len(v_dataset)))

trainloader=DataLoader(t_dataset, batch_size=16, shuffle=True, num_workers=4)
validloader=DataLoader(v_dataset, batch_size=16, shuffle=False, num_workers=4)


testiter = iter(trainloader)
imgs, img_id = testiter.next()

print(imgs.shape)


""" Training """
model = ResNetModel()
model.to(device)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, div_factor=10, pct_start=1 / epochs, steps_per_epoch=len(trainloader), epochs=epochs)

# writer = SummaryWriter(f'checkpoint/split_{split}/resnet')

print(f'Training Started Split_{split}')
training_loss = []
validation_loss = []
c_acc = 0.0

for epoch in range(epochs):
    start_time = time.time()

    train_pred = []
    valid_pred = []
    train_label = []
    valid_label = []
    avg_train_loss = 0.0
    l_rate = optimizer.param_groups[0]["lr"]
    model.train()
    for img, label in tqdm(trainloader):
        if train_on_gpu:
            img, label = img.to(device), label.to(device)

#         print(img.shape, label)
        optimizer.zero_grad()
        logits = model(img)
        loss = criterion(logits.squeeze(1).float(), label.float())
        loss.backward()
        optimizer.step()

        pred = logits.sigmoid().detach().round().cpu()
        train_pred.append(pred)
        train_label.append(label.cpu())

        avg_train_loss += loss.item()
#         print(optimizer.param_groups[0]["lr"])
        scheduler.step()

    model.eval()
    avg_valid_loss = 0.0
    with torch.no_grad():
        for img, label in tqdm(validloader):
            if train_on_gpu:
                img, label = img.to(device), label.to(device)

            logits = model.forward(img)

            val_loss = criterion(logits.squeeze(1).float(), label.float())
            avg_valid_loss += val_loss.item()

            pred = logits.sigmoid().detach().round().cpu()
            valid_pred.append(pred)
            valid_label.append(label.cpu())

    train_pred = torch.cat(train_pred).cpu().numpy()
    train_label = torch.cat(train_label).cpu().numpy()
    valid_pred = torch.cat(valid_pred).cpu().numpy()
    valid_label = torch.cat(valid_label).cpu().numpy()

    train_cm = np.array(confusion_matrix(train_label, train_pred))
    valid_cm = np.array(confusion_matrix(valid_label, valid_pred))

    avg_train_loss /= len(trainloader)
    avg_valid_loss /= len(validloader)
    train_acc = (train_pred == train_label).mean()
    valid_acc = (valid_pred == valid_label).mean()

    training_loss.append(avg_train_loss)
    validation_loss.append(avg_valid_loss)
    # l_rate = optimizer.param_groups[0]["lr"]

#     writer.add_scalars('Accuracy', {
#                        'Training Accuracy': train_acc, 'Validation Accuracy': valid_acc}, epoch)
#     writer.add_scalars('Loss', {
#                        'Training Loss': avg_train_loss, 'Validation Loss': avg_valid_loss}, epoch)
#     writer.add_scalar('Learning Rate', l_rate, epoch)

    if(c_acc<valid_acc):
        torch.save(model.state_dict(), "../checkpoint/split_{}/resnet/resnet_1_{}_{}.pth".format(split, epoch+1, valid_acc))
        np.savetxt(f'../checkpoint/split_{split}/resnet/valid_cm_{epoch+1}_{valid_acc}.txt', valid_cm, fmt='%10.0f')
        np.savetxt(f'../checkpoint/split_{split}/resnet/train_cm_{epoch+1}_{valid_acc}.txt', train_cm, fmt='%10.0f')
        c_acc = valid_acc

#     scheduler.step(avg_valid_loss)
#     scheduler.step()
    time_taken = time.time() - start_time
#     print("Epoch")

    print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t train_acc={:.4f} \t valid_acc={:.4f} \t l_rate={:.8f} \t time={:.2f}s'.
          format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, train_acc, valid_acc, l_rate, time_taken))

writer.close()
