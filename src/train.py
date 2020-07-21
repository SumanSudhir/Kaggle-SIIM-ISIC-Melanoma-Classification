import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

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
from modules import ResNetModel, EfficientModel
from utils import DrawHair


from torch.utils.tensorboard import SummaryWriter
import time

""" Initialization"""
nfolds = 5
SEED = 45
split = 0
epochs = 30
DEBUG = False

train = '../data/jpeg/train'
labels = '../data/my_train.csv'


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

""" Normalizing Meta features"""
## Sex Features
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df["sex"] = df["sex"].fillna(-1)

## Age Features
df["age_approx"] /= df["age_approx"].max()
df['age_approx'] = df['age_approx'].fillna(0)

meta_features = ['sex', 'age_approx']


print(df.head())

print("Previous Length", len(df))
if DEBUG:
    df = df[:5000]
print("Usable Length", len(df))

""" Dataset """

train_transform = transforms.Compose([
                        DrawHair(),
                        transforms.Resize((256,256)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
#                         transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
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
                          train=True, transforms=train_transform, meta_features=meta_features)
v_dataset=MelanomaDataset(df=df_valid, imfolder=train,
                          train=True, transforms=valid_transform, meta_features=meta_features)

print('Length of training and validation set are {} {}'.format(
    len(t_dataset), len(v_dataset)))

trainloader=DataLoader(t_dataset, batch_size=64, shuffle=True, num_workers=8, drop_last=True)
validloader=DataLoader(v_dataset, batch_size=64, shuffle=False, num_workers=8)


""" Training """
# model = ResNetModel()
model = EfficientModel(n_meta_features=len(meta_features))
model.to(device)
# model = nn.DataParallel(model)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, div_factor=10, pct_start=1 / epochs, steps_per_epoch=len(trainloader), epochs=epochs)

writer = SummaryWriter(f'../checkpoint/split_{split}/efficient')

print(f'Training Started Split_{split}')
training_loss = []
validation_loss = []
c_acc = 0.0

for epoch in range(epochs):
    start_time = time.time()

    train_prob = []
    valid_prob = []
    train_pred = []
    valid_pred = []
    train_label = []
    valid_label = []
    avg_train_loss = 0.0
    l_rate = optimizer.param_groups[0]["lr"]
    model.train()
    for img, label, meta in tqdm(trainloader):
        if train_on_gpu:
            img, label, meta = img.to(device), label.to(device), meta.to(device)

#         print(img.shape, label)
        optimizer.zero_grad()
        logits = model(img,meta)
        loss = criterion(logits.squeeze(1).float(), label.float())
        loss.backward()
        optimizer.step()

        pred = logits.sigmoid().detach().cpu()
        train_prob.append(pred)
        train_pred.append(pred.round())
        train_label.append(label.cpu())

        avg_train_loss += loss.detach().item()
#         print(optimizer.param_groups[0]["lr"])
        scheduler.step()

    model.eval()
    avg_valid_loss = 0.0
    with torch.no_grad():
        for img, label, meta in tqdm(validloader):
            if train_on_gpu:
                img, label, meta = img.to(device), label.to(device), meta.to(device)

            logits = model(img, meta)

            val_loss = criterion(logits.squeeze(1).float(), label.float())
            avg_valid_loss += val_loss.item()

            pred = logits.sigmoid().cpu()
            valid_prob.append(pred)
            valid_pred.append(pred.round())
            valid_label.append(label.cpu())

    train_pred = torch.cat(train_pred).cpu().numpy()
    train_prob = torch.cat(train_prob).cpu().numpy()
    train_label = torch.cat(train_label).cpu().numpy()

    valid_pred = torch.cat(valid_pred).cpu().numpy()
    valid_prob = torch.cat(valid_prob).cpu().numpy()
    valid_label = torch.cat(valid_label).cpu().numpy()


    train_cm = np.array(confusion_matrix(train_label, train_pred))
    valid_cm = np.array(confusion_matrix(valid_label, valid_pred))

    avg_train_loss /= len(trainloader)
    avg_valid_loss /= len(validloader)
    train_acc = (train_pred == train_label).mean()
    valid_acc = (valid_pred == valid_label).mean()

    train_roc = roc_auc_score(train_label, train_prob)
    valid_roc = roc_auc_score(valid_label, valid_prob)

    training_loss.append(avg_train_loss)
    validation_loss.append(avg_valid_loss)
    # l_rate = optimizer.param_groups[0]["lr"]


    writer.add_scalars('ROC_AUC', {
                       'Training ROC_AUC': train_roc, 'Validation ROC_AUC': valid_roc}, epoch)

    writer.add_scalars('Accuracy', {
                       'Training Accuracy': train_acc, 'Validation Accuracy': valid_acc}, epoch)

    writer.add_scalars('Loss', {
                       'Training Loss': avg_train_loss, 'Validation Loss': avg_valid_loss}, epoch)

    writer.add_scalar('Learning Rate', l_rate, epoch)

    if(c_acc<valid_roc):
        torch.save(model.state_dict(), "../checkpoint/split_{}/efficient/efficient_1_{}_{:.4f}.pth".format(split, epoch+1, valid_roc))
        np.savetxt('../checkpoint/split_{}/efficient/valid_cm_{}_{:.4f}.txt'.format(split, epoch+1, valid_roc), valid_cm, fmt='%10.0f')
        np.savetxt('../checkpoint/split_{}/efficient/train_cm_{}_{:.4f}.txt'.format(split, epoch+1, valid_roc), train_cm, fmt='%10.0f')
        c_acc = valid_roc

#     scheduler.step(avg_valid_loss)
#     scheduler.step()
    time_taken = time.time() - start_time
#     print("Epoch")

    print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t train_acc={:.4f} \t valid_acc={:.4f} \t train_roc={:.4f} \t valid_roc={:.4f} \t l_rate={:.8f} \t time={:.2f}s'.
          format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, train_acc, valid_acc, train_roc, valid_roc, l_rate, time_taken))

writer.close()
