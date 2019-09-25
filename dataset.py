# Generic imports 
import pandas as pd
import numpy as np
from PIL import Image
import os
import pdb
import random
import torch
import csv
import nltk
from collections import defaultdict

# Torch imports
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

from utils import readLangs, indexFromSentence
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, data_dir='/data/sachelar/fundus_images'):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.label2idx1 = {'Melanoma':0, 'Glaucoma':1, 'AMD':2, 'Diabetic Retinopathy':3}
            # 541 classes
        # self.label2idx2 = {j.strip().lower(): (int(i.strip().lower()) -1) for i, j in list(csv.reader(open('labels.txt', 'r'), delimiter='\t'))}
        self.label2idx2 = {j.strip().lower(): (int(i.strip().lower()) - 1) for
                i, j in list(csv.reader(open('labels2.txt', 'r'), delimiter='\t'))}

        self.to_tensor = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.data_info = pd.read_csv(csv_path, header=None)
        # change to -4?
        self.image_arr = np.asarray([os.path.join(data_dir,i.split('/')[-1].replace('%','')) for i in self.data_info.iloc[:,-3]])
        self.label_arr1 = [self.label2idx1[i] for i in np.asarray(self.data_info.iloc[:, -2])]
        self.label_arr2 = []
        self.lang, self.pairs = readLangs(self.data_info.iloc[:, -1], 15)

        for i,z in enumerate(np.asarray(self.data_info.iloc[:, -1])):
            self.label_arr2.append(self.label2idx2[z.strip().lower()])
        # self.label_arr2 = [self.label2idx2[i] for i in np.asarray(self.data_info.iloc[:, -1])]
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        self.data_len = len(self.data_info.index)

    def get_lang(self):
        return self.lang

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        img_as_tensor = self.to_tensor(img_as_img)
        single_image_label = self.label_arr1[index]
        fine_grained_label = self.label_arr2[index]
        text, length = indexFromSentence(self.lang, self.data_info.iloc[index, -1])
        text = torch.LongTensor(text).view(-1, 1)
        return (img_as_tensor, single_image_label, fine_grained_label, text)

    def __len__(self):
        return self.data_len

class GradedDatasetFromImages(Dataset):
    def __init__(self, csv_path, data_dir='/data/sachelar/fundus_images'):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.label2idx1 = {'melanoma':0, 'glaucoma':0, 'amd':0, 'diabetic retinopathy':0, 'Normal':1}

        self.to_tensor = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.data_info = pd.read_csv(csv_path, header=None)
        self.image_arr = np.asarray([os.path.join(data_dir, i.replace('%','')) for i in self.data_info.iloc[:,0]])
        self.label_arr1 = [self.label2idx1[i] for i in np.asarray(self.data_info.iloc[:, -2])]
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name).convert('RGB')
        img_as_tensor = self.to_tensor(img_as_img)
        single_image_label = self.label_arr1[index]
        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len
