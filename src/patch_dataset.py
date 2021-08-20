import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from scipy.misc.pilutil import imread
# from skimage.color import rgb2gray, gray2rgb
import json


class Patch_Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path,json_path):
        super(Patch_Dataset, self).__init__()
        self.data = self.load_flist(file_path) ## file list를 list로 받음
        self.json_data = self.load_json(json_path) ## json

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)
        try:
            l_prob,r_prob = self.load_prob(index)
        except:
            # print(self.json_data[index])
            print('loading error: ' + self.json_data[index])
            l_prob,r_prob = self.load_prob(0) 

        return item, l_prob,r_prob

    def load_json(self,json_path):
        with open(json_path, 'r') as jsonfile:
            data = json.load(jsonfile)
            jsonfile.close()
        return data

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_prob(self,index):
        l_prob = torch.tensor(self.json_data[self.data[index]]['l']['prob'])
        r_prob = torch.tensor(self.json_data[self.data[index]]['r']['prob'])
        return l_prob,r_prob


    def load_item(self, index):

        # load image
        img = imread(self.data[index])

        return self.to_tensor(img)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
