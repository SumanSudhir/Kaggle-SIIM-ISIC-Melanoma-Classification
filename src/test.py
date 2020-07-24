import os
import sys

import numpy as np
import pandas as pd
import random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


from dataset import MelanomaDataset
from modules import ResNetModel, EfficientModel


import time

""" Initialization"""
SEED = 45
DEBUG = False

test = '../data_256/test'
labels = '../data/test.csv'
train_labels = '../data/train_combined.csv'
sample = '../data/sample_submission.csv'

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
    print("CUDA is not available. Testing on CPU...")
else:
    print("CUDA is available. Testing on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv(labels)
df_train = pd.read_csv(train_labels)


""" Normalizing Meta features"""
## Sex Features
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df["sex"] = df["sex"].fillna(-1)

## Age Features
df["age_approx"] /= df_train["age_approx"].max()
df['age_approx'] = df['age_approx'].fillna(0)

meta_features = ['sex', 'age_approx']

print(df.head())



print("Previous Length", len(df))
if DEBUG:
    df = df[:100]
print("Usable Length", len(df))

""" Dataset """

test_transform=transforms.Compose([
#                         transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225])])


t_dataset=MelanomaDataset(df=df, imfolder=test,
                          train=False, transforms=test_transform, meta_features=meta_features)

print('Length of test set is {}'.format(len(t_dataset)))

testloader=DataLoader(t_dataset, batch_size=64, shuffle=False, num_workers=8)

"""Testing"""
# model = ResNetModel()()
# model = EfficientModel()
model = EfficientModel(n_meta_features=len(meta_features))
model.load_state_dict(torch.load("../checkpoint/split_3/efficient_256/efficient_256_6_0.9861.pth", map_location=torch.device(device)))
model.to(device)

model.eval()
test_prob = []
with torch.no_grad():
    for img, meta in tqdm(testloader):
        if train_on_gpu:
            img, meta = img.to(device), meta.to(device)

        logits = model.forward(img, meta)

        pred = logits.sigmoid().detach().cpu()
        test_prob.append(pred)

test_prob = torch.cat(test_prob).cpu().numpy()

sub = pd.read_csv(sample)
sub['target'] = test_prob.reshape(-1,)
sub.to_csv('../submission/submission_10.csv', index=False)
