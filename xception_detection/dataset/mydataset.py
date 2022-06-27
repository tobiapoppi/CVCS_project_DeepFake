"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import os
import random


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        self.fake_count = 0
        self.real_count = 0
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            if int(words[1] == 0):
                self.real_count += 1
            else:
                self.fake_count += 1

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

    def get_counts(self):
        return self.real_count, self.fake_count