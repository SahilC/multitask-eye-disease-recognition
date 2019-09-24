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
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict

# Torch imports
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Dataset  # For custom datasets
from models import MultiTaskModel
from utils import compute_bleu, compute_topk, accuracy_recall_precision_f1,calculate_confusion_matrix,readLangs, indexFromSentence

ind2word = None
lang1 = None

# Hacks for Reproducibility
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES']='2, 3'

# from cnn_model import MnistCNNModel

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, class_type = 'disease', data_dir='/data/sachelar/fundus_images'):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        global ind2word, lang1
        self.class_type = class_type
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
        ind2word = self.lang.index2word
        lang1 = self.lang

        for i,z in enumerate(np.asarray(self.data_info.iloc[:, -1])):
            self.label_arr2.append(self.label2idx2[z.strip().lower()])
        # self.label_arr2 = [self.label2idx2[i] for i in np.asarray(self.data_info.iloc[:, -1])]
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        self.data_len = len(self.data_info.index)

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


if __name__ == "__main__":
    batch_size = 64
    epochs = 20
    val_split = 0.15
    print_every = 100
    class_type = 'fine-grained-disease'

    # custom_from_images =  CustomDatasetFromImages('all_data_filtered.csv', class_type=class_type)
    custom_from_images =  CustomDatasetFromImages('cleaned_data_dedup.csv', class_type=class_type)

    dset_len = len(custom_from_images)
    test_size = int(val_split * dset_len)
    val_size = int(val_split * dset_len)
    train_size = int(dset_len - 2 * val_size)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_from_images, [train_size, val_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    pin_memory=False,
                                                    drop_last = True,
                                                    shuffle=True,
                                                    num_workers=32)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=batch_size,
                                                    pin_memory=False,
                                                    drop_last = True,
                                                    shuffle=True, 
                                                    num_workers = 32)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    pin_memory=False,
                                                    drop_last = True,
                                                    shuffle=True, 
                                                    num_workers = 32)


    #model = MnistCNNModel()
    # model = models.densenet121(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.resnet101(pretrained=True)
    # model = models.resnet34(pretrained=True)
    model = models.vgg19(pretrained=True)
    model = MultiTaskModel(model, vocab_size = lang1.n_words) 
    model  = nn.DataParallel(model)
    # model.load_state_dict(torch.load('best_model.pth'))

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
            weight_decay=1e-6,momentum=0.9, lr=1e-3, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-7, verbose=True)
    min_val_loss = 100

    model.eval()
    val_loss = 0.0
    total_f_acc = 0.0
    total_acc = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_cm = 0.0
    total_bleu = 0.0
    k_vals = [1, 2, 3, 4, 5]

    total_topk = {k:0.0 for k in k_vals}
    per_disease_topk = defaultdict(lambda: {k:0.0 for k in k_vals})
    for i, (images, labels, f_labels, text) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        f_labels = f_labels.to(device)
        text = text.to(device)
        disease, f_disease, text_pred = model(images, text)
        loss = criterion(disease, labels) + criterion(f_disease, f_labels)
        # loss = criterion(f_disease, f_labels)
        # loss = criterion(disease, labels)
        # loss = criterion(disease, labels)
        # val_loss += loss.item()
        for k in range(text_pred.size(1)):
            text_loss = criterion(text_pred[:, k].squeeze(), text[:,k + 1].squeeze())
        val_loss += (text_loss.item())
        preds = F.log_softmax(disease, dim = -1)
        pred = preds.argmax(dim=-1)
        total_acc += (pred.eq(labels).sum().item() / batch_size)

        preds = F.log_softmax(f_disease, dim = -1)
        pred = preds.argmax(dim=-1)
        # Fine grained accuracy
        total_f_acc += (pred.eq(f_labels).sum().item() / batch_size)

        acc, recall, precision, f1 = accuracy_recall_precision_f1(pred, f_labels)
        total_recall += np.mean(recall)
        total_precision += np.mean(precision)
        total_f1 += np.mean(f1)
        for k in k_vals:
            total_topk[k] += compute_topk(preds, f_labels, k)
            for d in [0, 1, 2, 3]:
                mask = labels.eq(d)
                if mask.sum() > 0:
                    per_disease_topk[d][k] += compute_topk(preds[mask], f_labels[mask], k)

        # Caption generation
        preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
        text1 = text[:, 1:].squeeze().tolist()
        preds1 = preds.tolist()
        t_bleu, sent_gt, sent_pred = compute_bleu(text1, preds1)
        total_bleu += t_bleu

        preds = F.log_softmax(disease, dim = -1)
        pred = preds.argmax(dim=-1)

        cm = calculate_confusion_matrix(pred, labels)
        total_cm += cm
    for d in [0,1,2,3]:
        for k in k_vals:
            per_disease_topk[d][k] = per_disease_topk[d][k] / len(val_loader)
    total_topk = [total_topk[k] / len(val_loader) for k in k_vals]
    total_bleu = total_bleu / (len(val_loader))
    val_loss = val_loss / len(val_loader)
    total_acc = total_acc / len(val_loader)
    total_f_acc = total_f_acc / len(val_loader)
    total_f1 = total_f1 / len(val_loader)
    total_precision = total_precision / len(val_loader)
    total_recall = total_recall / len(val_loader)
    # total_cm = total_cm / len(val_loader)
    print('Epoch: {}\tTest Loss:{:.8f}\tAcc:{:.8f}\tFAcc:{:.8f}'.format(e,
        val_loss, total_acc, total_f_acc))
    print('Top',k,':', total_topk)
    print('Per Disease', per_disease_topk)
    print('BLEU', total_bleu)
    print('F1', total_f1, np.mean(total_f1))
    print('Pr', total_precision, np.mean(total_precision))
    print('Recall', total_recall, np.mean(total_recall)) 
    print('-----------CM------------')
    print(total_cm)
    print('-----------------------')
