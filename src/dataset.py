import os
import pandas as pd
import numpy as np
import PIL
from PIL import Image
import cv2
from torch.utils.data import Dataset


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms=None, meta_features=None):
        """
        Class Initialization
        Args:
            df:             Dataframe with data description
            imfolder:       Image folder
            train:          Flag with training dataset or validation
            transforms:     Image transformation method to be applied
            meta_features:  List of features with meta information, such as sex, age
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        self.meta_features = meta_features

    def __getitem__(self, index):
        im_path = os.path.join(
            self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        img = Image.open(im_path)
#         img = cv2.imread(im_path)
        meta = np.array(
            self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.transforms:
            img = self.transforms(img)

        if self.train:
            label = self.df.iloc[index]['target']
            return img, label, meta

        return img, meta

    def __len__(self):
        return len(self.df)
