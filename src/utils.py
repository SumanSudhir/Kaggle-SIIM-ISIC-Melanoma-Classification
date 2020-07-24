import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torchvision

class DrawHair:
    """
    Draw a random number of pseduo hairs

    Args:
    hairs (int): maximum number of hairs to draw
    width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width: tuple = (1,2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL IMage: Image with drwan hairs
        """
        if not self.hairs:
            return img

        width, height = img.size
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0,width), random.randint(0,height//2))
            # The end of the line
            end = (random.randint(0, width), random.randint(0, height))
            # Black color of the hair
            color = (0, 0, 0)
            cv2.line(np.array(img), origin, end, color, random.randint(self.width[0], self.width[1]))

        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'


## https://github.com/ufoym/imbalanced-dataset-sampler
## https://www.kaggle.com/doanquanvietnamca/imbalanced-dataset-sampler

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.df.target[idx].item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        return torch.cat([self.ap(x), self.mp(x)], dim=1)
