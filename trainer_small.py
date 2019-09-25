import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from collections import defaultdict
from utils import compute_bleu, compute_topk, accuracy_recall_precision_f1, calculate_confusion_matrix

class Trainer(object):
    def __init__(self, model, optimizer, scheduler, criterion, epochs, print_every = 100, min_val_loss = 100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.criterion = criterion
        self.print_every = print_every
        self.min_val_loss = min_val_loss
        self.save_location_dir = os.path.join('models', str(datetime.now()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not os.path.exists(self.save_location_dir):
            os.mkdir(self.save_location_dir)
        self.save_path = os.path.join(self.save_location_dir, 'best_model.pt')
        self.summary_writer =  SummaryWriter(os.path.join(self.save_location_dir, 'logs'), 300)

    def train(self, train_loader, val_loader):
        for e in range(self.epochs):
                self.model.train()
                total_train_loss, accuracy = self.train_iteration(train_loader)
                print("Epoch", e)
                self.summary_writer.add_scalar('training/total_train_loss', total_train_loss, e)
                self.summary_writer.add_scalar('training/acc', accuracy, e)
                val_loss, total_d_acc, total_f1, total_recall, total_precision, total_cm = self.validate(val_loader)

                self.summary_writer.add_scalar('validation/val_loss', val_loss, e)
                self.summary_writer.add_scalar('validation/t1_acc', total_d_acc, e)

                self.summary_writer.add_scalar('validation/f1_mean', total_f1, e)
                self.summary_writer.add_scalar('validation/recall_mean', total_recall, e)
                self.summary_writer.add_scalar('validation/precision_mean', total_precision, e)
                print('Val Loss',val_loss, 'total_d_acc',total_d_acc, 'F1', total_f1, 'R', total_recall,'P', total_precision)
                print(total_cm)


    def train_iteration(self, train_loader):
        train_loss = 0.0
        accuracy = 0.0
        total_disease_acc = 0.0
        total_train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
               batch_size = images.size(0)
               images = images.to(self.device)
               labels = labels.to(self.device)
               
               self.optimizer.zero_grad()
               disease = self.model(images)
               loss = self.criterion(disease, labels)

               loss.backward()
               self.optimizer.step()

               train_loss += loss.item()
               total_train_loss += loss.item()

               d_pred = F.log_softmax(disease, dim= -1).argmax(dim=-1)
               total_disease_acc += d_pred.eq(labels).sum().item()

               if i != 0 and i % self.print_every == 0:
                  avg_loss = train_loss / self.print_every
                  total_disease_acc = total_disease_acc / self.print_every
                  total_train_loss = total_train_loss / self.print_every

                  print('Iter:{}\tTraining Loss:{:.8f}\tAcc:{:.8f}'.format(i,
                      avg_loss, total_disease_acc / batch_size))

                  train_loss = 0.0
                  total_disease_acc = 0.0
        return (total_train_loss, total_disease_acc/batch_size)

    def validate(self, val_loader, epoch = 0):
        self.model.eval()
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

        k_vals = [1, 2, 3, 4, 5]
        total_topk = {k:0.0 for k in k_vals}
        per_disease_topk = defaultdict(lambda: {str(k):0.0 for k in k_vals})
        losses = []
        for i, (images, labels) in enumerate(val_loader):
            batch_size = images.size(0)
            images = images.to(self.device)
            labels = labels.to(self.device)
            diseases = self.model(images)
            loss1 = self.criterion(diseases, labels)

            val_loss = loss1

            # Evaluation of P, R, F1, BLEU
            d_pred = F.log_softmax(diseases, dim = -1).argmax(dim=-1)
            total_d_acc += (d_pred.eq(labels).sum().item() / batch_size)
            acc, recall, precision, f1 = accuracy_recall_precision_f1(d_pred,
                    labels)

            total_recall += np.mean(recall)
            total_precision += np.mean(precision)
            total_f1 += np.mean(f1)

            cm = calculate_confusion_matrix(d_pred, labels)
            total_cm += (cm / batch_size)
        val_loss = val_loss / len(val_loader)
        total_d_acc = total_d_acc / len(val_loader)
        total_f1 = total_f1 / len(val_loader)
        total_precision = total_precision / len(val_loader)
        total_recall = total_recall / len(val_loader)
        total_cm = total_cm / len(val_loader)

        self.scheduler.step(val_loss)
        if val_loss <= self.min_val_loss:
           torch.save(self.model.state_dict(), self.save_path)
           self.min_val_loss = val_loss

        disease_f1 = {}
        disease_precision = {}
        disease_recall = {}

        # for i in range(len(total_f1)):
        #   disease_f1[i] = total_f1[i]
        #   disease_precision[i] = total_precision[i]
        #   disease_recall[i] = total_recall[i]

        return (val_loss, total_d_acc, total_f1, total_recall,
                total_precision, total_cm) 
