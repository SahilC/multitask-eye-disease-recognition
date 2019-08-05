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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Dataset  # For custom datasets

from collections import defaultdict
from utils import accuracy_recall_precision_f1,calculate_confusion_matrix,readLangs, indexFromSentence
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
class LanguageModel(nn.Module):
    def __init__(self, vocab_size = 193, embed_size = 256, hidden_size = 512,
            num_layers = 1, dropout_p = 0.1):
        super(LanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=2)
        self.project = nn.Linear(512, hidden_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_length = 15
        self.teacher_forcing_ratio = 0.5
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, img_feats, text):
        text = text.squeeze()
        preds = []
        self.gru.flatten_parameters()
        decoder_input = text[:, 0]
        state = self.project(img_feats).unsqueeze(0)
        for i in range(1, self.max_length):
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
            embeddings = self.dropout(self.embedding(decoder_input)).squeeze()
            feats, state = self.gru(embeddings.unsqueeze(1), state)
            pred = self.linear(state).squeeze()

            if use_teacher_forcing:
                decoder_input = text[:,i]
            else:
                output = F.log_softmax(pred, dim=-1)
                decoder_input = torch.argmax(output, dim=-1)
            preds.append(pred.unsqueeze(1))

        return torch.cat(preds, 1)

class MultiTaskModel(nn.Module):
    def __init__(self, model, vocab_size):
        super(MultiTaskModel, self).__init__()
        self.feature_extract = model.features
        # self.feature_extract = torch.nn.Sequential(*list(model.children())[:-1])

        self.disease_classifier = nn.Sequential(nn.Linear(512, 512),
                nn.ReLU(), nn.Linear(512, 4))
        self.fine_disease_classifier = nn.Sequential(nn.Linear(512, 512),
                nn.ReLU(), nn.Linear(512, 320))
        self.language_classifier = LanguageModel(vocab_size = vocab_size)

    def forward(self, data,text):
        features = self.feature_extract(data).squeeze()
        # out = features
        out = F.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        return (self.disease_classifier(out),
                self.fine_disease_classifier(out),
                self.language_classifier(out, text)
                )

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

def compute_topk(topk_vals, gt, k):
    _, preds = topk_vals.topk(k = k, dim = 1)
    topk_acc = 0
    for i in range(preds.size(1)):
        topk_acc += preds[:, i].eq(gt).sum().item()
    return (topk_acc / topk_vals.size(0))

def compute_bleu(text1, preds1):
    global ind2word
    bleu = 0
    sents_gt = []
    sents_pred = []
    for k in range(len(text1)):
        sent1 = []
        sent2 = []
        weights = (0.25, 0.25, 0.25, 0.25)
        for j in range(len(text1[k])):
                if text1[k][j] != 0 and text1[k][j] != 1 and text1[k][j] != 2:
                    sent1.append(ind2word[text1[k][j]])
                if preds1[k][j] != 0 and preds1[k][j] != 1 and preds1[k][j] != 2:
                    sent2.append(ind2word[preds1[k][j]])
        if len(sent2) >0 and len(sent2) < 4 and weights  == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / len(sent2),) * len(sent2)
        c_bleu = sentence_bleu([sent1], sent2, weights = weights)
        sents_gt.append(sent1)
        sents_pred.append(sent2)
        bleu += c_bleu
    return (bleu/len(text1)), sents_gt, sents_pred

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
    # if class_type == 'disease':
    #    model.classifier = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 4))
    #else:
    #    model.classifier = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 541))
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
    for e in range(epochs):
            train_loss = 0.0
            total_train_loss = 0.0
            accuracy = 0.0
            total_disease_acc = 0.0
            bleu = 0.0
            model.train()
            total_tl1 = 0
            total_tl2 =0
            total_tl3 = 0
            for i, (images, labels, f_labels, text) in enumerate(train_loader):
               images = images.to(device)
               labels = labels.to(device)
               f_labels = f_labels.to(device)
               text = text.to(device)
               optimizer.zero_grad()
               disease, f_disease, text_pred = model(images, text)
               loss1 = criterion(disease, labels)
               loss2 = criterion(f_disease, f_labels)
               # loss = criterion(disease, labels)
               # loss = (loss1 + loss2)
               loss = 0.0
               total_tl1 += loss1.item()
               total_tl2 += loss2.item()
               text_loss = 0.0
               for k in range(text_pred.size(1)):
                   text_loss += criterion(text_pred[:, k].squeeze(), text[:, k + 1].squeeze())
               loss += text_loss
               total_tl3 += text_loss.item()
               loss.backward()
               optimizer.step()

               train_loss += loss.item()
               total_train_loss += loss.item()
               pred = F.log_softmax(f_disease, dim = -1).argmax(dim=-1)
               accuracy += pred.eq(f_labels).sum().item()
               d_pred = F.log_softmax(disease, dim= -1).argmax(dim=-1)
               total_disease_acc += d_pred.eq(labels).sum().item()
               preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
               text1 = text[:, 1:].squeeze().tolist()
               preds1 = preds.tolist()
               tbleu, _, _ = compute_bleu(text1, preds1)
               bleu += tbleu

               if i != 0 and i % print_every == 0:
                  avg_loss = train_loss / print_every
                  accuracy = accuracy / print_every
                  total_disease_acc = total_disease_acc / print_every
                  avg_text_loss = text_loss / print_every
                  bleu = bleu / print_every
                  print('Epoch: {}\tIter:{}\tTraining Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}\tTextLoss:{:.8f}'.format(e, i, avg_loss,
                              accuracy/batch_size,
                              total_disease_acc / batch_size,
                              bleu,
                              text_loss.item()))
                  train_loss = 0.0
                  text_loss = 0.0
                  accuracy = 0.0
                  total_disease_acc = 0.0
                  bleu = 0.0

            model.eval()
            val_loss = 0.0
            total_acc = 0.0
            total_recall = 0.0
            total_precision = 0.0
            total_f1 = 0.0
            total_cm = 0
            total_d_acc = 0.0
            bleu = 0.0
            total_l1 = 0
            total_l2  = 0
            total_l3 = 0
            for i, (images, labels, f_labels, text) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                f_labels = f_labels.to(device)
                text = text.to(device)
                diseases, fine_diseases, text_pred = model(images, text)
                loss1 = criterion(diseases, labels)
                loss2 = criterion(fine_diseases, f_labels)
                loss = (loss1 + loss2)
                # loss = (loss1)
                # loss = criterion(diseases, labels)
                # val_loss += loss.item()
                text_loss = 0.0
                for k in range(text_pred.size(1)):
                    text_loss += criterion(text_pred[:,k].squeeze(), text[:,k + 1].squeeze())
                val_loss += (text_loss.item())
                pred = F.log_softmax(fine_diseases, dim = -1).argmax(dim=-1)
                d_pred = F.log_softmax(diseases, dim = -1).argmax(dim=-1)
                total_acc += (pred.eq(f_labels).sum().item() / batch_size)
                total_d_acc += (d_pred.eq(labels).sum().item() / batch_size)
                acc, recall, precision, f1 = accuracy_recall_precision_f1(pred,
                        f_labels)
                total_recall += np.mean(recall)
                total_precision += np.mean(precision)
                total_f1 += np.mean(f1)
                preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
                text1 = text[:, 1:].squeeze().tolist()
                preds1 = preds.tolist()
                t_bleu, sent_gt, sent_pred = compute_bleu(text1, preds1)
                bleu += t_bleu
                total_l1 += loss1.item()
                total_l2 += loss2.item()
                total_l3 += text_loss.item()
                # cm = calculate_confusion_matrix(pred, labels)
                # total_cm += (cm / batch_size)
            bleu = bleu / (len(val_loader))
            total_train_loss = total_train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            total_l1 /= len(val_loader)
            total_l2 /= len(val_loader)
            total_l3 /= len(val_loader)
            total_tl1 /= len(train_loader)
            total_tl2 /= len(train_loader)
            total_tl3 /= len(train_loader)
            total_acc = total_acc / len(val_loader)
            total_d_acc = total_d_acc / len(val_loader)
            total_f1 = total_f1 / len(val_loader)
            total_precision = total_precision / len(val_loader)
            total_recall = total_recall / len(val_loader)
            total_cm = total_cm / len(val_loader)

            scheduler.step(val_loss)
            if val_loss <= min_val_loss:
               torch.save(model.state_dict(), 'best_model.pth')
               min_val_loss = val_loss
            #print(e, total_train_loss,',',val_loss,',',total_tl1,',',total_l1,',',total_tl2,',',total_l2,',',total_tl3,',',total_l3)
            print('Epoch: {}\tTrain Loss:{:.8f}\tVal Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}'.format(e, total_train_loss, val_loss, total_acc, total_d_acc))
            print('BLEU', bleu)
            print('F1', total_f1, np.mean(total_f1))
            print('Pr', total_precision, np.mean(total_precision))
            print('Recall', total_recall, np.mean(total_recall))
            print('-----------CM------------')
            print(total_cm)
            print('-----------------------')
            for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
                print(sent_gt[k])
                print(sent_pred[k])
                print('---------------------')

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
