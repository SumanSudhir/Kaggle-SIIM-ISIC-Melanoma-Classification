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
from modules import ResNetModel


import time

""" Initialization"""
SEED = 45
DEBUG = True

test = '../data/jpeg/test'
labels = '../data/test.csv'
sample = '../data/sample_submision'

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
    print("CUDA is not available. Testing on CPU...")
else:
    print("CUDA is available. Testing on GPU...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv(labels)

print("Previous Length", len(df))
if DEBUG:
    df = df[:500]
print("Usable Length", len(df))

""" Dataset """

test_transform=transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                             0.229, 0.224, 0.225])])


t_dataset=MelanomaDataset(df=df, imfolder=test,
                          train=False, transforms=test_transform)

print('Length of test set is {}'.format(len(t_dataset)))

testloader=DataLoader(t_dataset, batch_size=16, shuffle=False, num_workers=4)

"""Testing"""
model = ResNetModel()
model.load_state_dict(torch.load("", map_location=torch.device(device)))
model.to(device)

model.eval()
test_prob = []
with torch.no_grad():
    for img in tqdm(testloader):
        if train_on_gpu:
            img, label = img.to(device)

        logits = model.forward(img)

        pred = logits.sigmoid().detach().cpu()
        test_prob.append(pred)

test_prob = torch.cat(test_prob).cpu().numpy()

sub = pd.read_csv(sample)
sub['target'] = test_prob.reshape(-1,)
sub.to_csv('submission.csv', index=False)
