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
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from dataset import MelanomaDataset
from modules import ResNetModel, EfficientModel, EfficientModelwithoutMeta, Model
from utils import DrawHair, ImbalancedDatasetSampler
from torch.utils.data.sampler import WeightedRandomSampler


from torch.utils.tensorboard import SummaryWriter
import time

""" Initialization"""
nfolds = 4
SEED = 45
fold = 1
epochs = 20
input_res  = 512
resolution = 384  # res for model
label_smoothing = 0.03
DEBUG = False

# train = '../combined_256'
# external = '../data/external_mal.csv'
# train = '../data_384/train'
train = '../data_merged_512/512x512-dataset-melanoma/512x512-dataset-melanoma'
# train = '../data/jpeg/train'
# labels = '../data/my_train.csv'
labels = '../data_merged_512/folds.csv'
# labels = '../data/train_combined.csv'


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("CUDA is not available. Training on CPU...")
else:
    print("CUDA is available. Training on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Split dataset in n Folds"""
df = pd.read_csv(labels)

# splits = StratifiedKFold(n_splits=nfolds, random_state=SEED, shuffle=True)
# splits = list(splits.split(df, df.target))
# folds_splits = np.zeros(len(df)).astype(np.int)

# for i in range(nfolds):
#     folds_splits[splits[i][1]] = i
# df["split"] = folds_splits


""" External Data """
# df_ext = pd.read_csv(external)
# df_ext["split"] = 10

# df_new = pd.concat([df, df_ext], ignore_index=True)
# print(df_new.tail())
# df = df_new.sample(frac=1).reset_index(drop=True)

""" Normalizing Meta features"""
## Sex Features
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df["sex"] = df["sex"].fillna(-1)

## anatom_site_general_challenge Features oral/genital
df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].map({'anterior torso': 0, 'head/neck': 1, 'lateral torso': 2, 'lower extremity': 3, 'oral/genital': 4, 'palms/soles': 5, 'posterior torso': 6, 'torso': 7, 'upper extremity': 8})

df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].fillna(-1)
df["anatom_site_general_challenge"] /= df["anatom_site_general_challenge"].max()

## Age Features
df["age_approx"] /= df["age_approx"].max()
df['age_approx'] = df['age_approx'].fillna(0)

meta_features = ['sex', 'age_approx', 'anatom_site_general_challenge']


print(df.tail())

print("Previous Length", len(df))
if DEBUG:
    df = df[:500]
print("Usable Length", len(df))

""" Dataset """

# train_transform = transforms.Compose([
# #                         DrawHair(),
# #                         transforms.Resize((256,256)),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.RandomVerticalFlip(),
#                         transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
# #                         transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                                              0.229, 0.224, 0.225])
#                         ])


# valid_transform=transforms.Compose([
# #                         transforms.Resize((256,256)),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                                              0.229, 0.224, 0.225])])


train_transform = A.Compose([
                            A.JpegCompression(p=0.5),
                            A.Rotate(limit=80, p=1.0),
                            A.OneOf([
                                A.OpticalDistortion(),
                                A.GridDistortion(),
                                A.IAAPiecewiseAffine(),
                            ]),
                            A.RandomSizedCrop(min_max_height=(int(resolution*0.7), input_res),
                                                height=resolution, width=resolution, p=1.0),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.GaussianBlur(p=0.3),
                            A.OneOf([
                                A.RandomBrightnessContrast(),
                                A.HueSaturationValue(),
                            ]),
                            A.OneOf([#off in most cases
                                A.MotionBlur(blur_limit=3, p=0.1),
                                A.MedianBlur(blur_limit=3, p=0.1),
                                A.Blur(blur_limit=3, p=0.1),
                            ], p=0.2),
                            A.Cutout(num_holes=8, max_h_size=resolution//8, max_w_size=resolution//8, fill_value=0, p=0.3),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ], p=1.0)


valid_transform = A.Compose([
                            A.CenterCrop(height=resolution, width=resolution, p=1.0),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ], p=1.0)




df_train=df[df['fold'] != fold]
df_valid=df[df['fold'] == fold]

class_sample_count = np.array([len(np.where(df_train["target"]==t)[0]) for t in np.unique(df_train["target"])])
print(class_sample_count)

# weight = 1. / class_sample_count
# samples_weight = np.array([weight[t] for t in df_train["target"]])
# samples_weight = torch.from_numpy(samples_weight)
# sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
# print(samples_weight)

t_dataset=MelanomaDataset(df=df_train, imfolder=train,
                          train=True, transforms=train_transform, meta_features=meta_features)
v_dataset=MelanomaDataset(df=df_valid, imfolder=train,
                          train=True, transforms=valid_transform, meta_features=meta_features)

print('Length of training and validation set are {} {}'.format(
    len(t_dataset), len(v_dataset)))

trainloader=DataLoader(t_dataset, batch_size=32, shuffle=True, num_workers=8)
validloader=DataLoader(v_dataset, batch_size=32, shuffle=False, num_workers=8)

""" Training """
# model = ResNetModel()
# model = EfficientModelwithoutMeta()
model = Model(arch='efficientnet-b2')
# model = EfficientModel(n_meta_features=len(meta_features))
model.to(device)
# model = nn.DataParallel(model)

criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
scheduler=torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, div_factor=10, pct_start=1 / epochs, steps_per_epoch=len(trainloader), epochs=epochs)

writer = SummaryWriter(f'../checkpoint/fold_{fold}/efficient_{resolution}')

print(f'Training Started Fold_{fold}')
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

#         print(img.shape, label.shape)
        optimizer.zero_grad()
        label_smo = label.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        logits = model(img)
        loss = criterion(logits.squeeze(1).float(), label_smo.type_as(logits).float())
        loss.backward()
        optimizer.step()

        pred = logits.sigmoid().detach().cpu()
        train_prob.append(pred)
        train_pred.append(pred.round())
        train_label.append(label.cpu())

        avg_train_loss += loss.detach().item()
        scheduler.step()

    model.eval()
    avg_valid_loss = 0.0
    with torch.no_grad():
        for img, label, meta in tqdm(validloader):
            if train_on_gpu:
                img, label, meta = img.to(device), label.to(device), meta.to(device)

            label_smo = label.float() * (1 - label_smoothing) + 0.5 * label_smoothing
            logits = model(img)
            val_loss = criterion(logits.squeeze(1).float(), label_smo.type_as(logits).float())

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

#     train_acc = (train_pred == train_label).mean()
#     valid_acc = (valid_pred == valid_label).mean()

    train_acc = (train_cm[0,0] + train_cm[1,1])/np.sum(train_cm)
    valid_acc = (valid_cm[0,0] + valid_cm[1,1])/np.sum(valid_cm)

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

    if(c_acc<valid_roc or valid_roc > 0.90):
        torch.save(model.state_dict(), "../checkpoint/fold_{}/efficient_{}/efficientb2_{}_{}_{:.4f}.pth".format(fold, resolution, resolution, epoch+1, valid_roc))
        np.savetxt('../checkpoint/fold_{}/efficient_{}/valid_cm_{}_{:.4f}.txt'.format(fold, resolution, epoch+1, valid_roc), valid_cm, fmt='%10.0f')
        np.savetxt('../checkpoint/fold_{}/efficient_{}/train_cm_{}_{:.4f}.txt'.format(fold, resolution, epoch+1, valid_roc), train_cm, fmt='%10.0f')
        c_acc = valid_roc

#     scheduler.step(avg_valid_loss)
#     scheduler.step()
    time_taken = time.time() - start_time
#     print("Epoch")

    print('Epoch {}/{} \t train_loss={:.4f} \t valid_loss={:.4f} \t train_acc={:.4f} \t valid_acc={:.4f} \t train_roc={:.4f} \t valid_roc={:.4f} \t l_rate={:.8f} \t time={:.2f}s'.
          format(epoch + 1, epochs, avg_train_loss, avg_valid_loss, train_acc, valid_acc, train_roc, valid_roc, l_rate, time_taken))

writer.close()
