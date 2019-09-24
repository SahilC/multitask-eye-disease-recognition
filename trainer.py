import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from collections import defaultdict
from utils import compute_bleu, compute_topk, accuracy_recall_precision_f1, calculate_confusion_matrix

class Trainer(object):
    def __init__(self, model, optimizer, scheduler, criterion, tasks, epochs, lang, print_every = 100, min_val_loss = 100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.criterion = criterion
        self.print_every = print_every
        self.min_val_loss = min_val_loss
        self.lang = lang
        self.save_location_dir = os.path.join('models', str(datetime.now()))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tasks = tasks
        if not os.path.exists(self.save_location_dir):
            os.mkdir(self.save_location_dir)
        self.save_path = os.path.join(self.save_location_dir, 'best_model.pt')
        self.summary_writer =  SummaryWriter(os.path.join(self.save_location_dir, 'logs'), 300)

    def train(self, train_loader, val_loader, test_loader):
        for e in range(self.epochs):
                self.model.train()
                total_train_loss, total_tl1, total_tl2, total_tl3, total_disease_acc, accuracy, bleu = self.train_iteration(train_loader)
                print("Epoch", e)
                self.summary_writer.add_scalar('training/total_train_loss', total_train_loss, e)
                self.summary_writer.add_scalar('training/total_t1_loss', total_tl1, e)
                self.summary_writer.add_scalar('training/total_t2_loss', total_tl2, e)
                self.summary_writer.add_scalar('training/total_t3_loss', total_tl3, e)
                self.summary_writer.add_scalar('training/t1_acc', total_disease_acc, e)
                self.summary_writer.add_scalar('training/t2_acc', accuracy, e)
                self.summary_writer.add_scalar('training/t3_bleu', bleu, e)
                val_loss, total_d_acc, total_acc, bleu, total_f1, total_recall, total_precision, sent_gt, sent_pred, total_topk, per_disease_topk = self.validate(val_loader)

                self.summary_writer.add_scalar('validation/val_loss', val_loss, e)
                self.summary_writer.add_scalar('validation/t1_acc', total_d_acc, e)
                self.summary_writer.add_scalar('validation/t2_acc', total_acc, e)
                self.summary_writer.add_scalar('validation/BLEU', bleu, e)
                # self.summary_writer.add_scalars('validation/f1_scores', disease_f1, e)
                # self.summary_writer.add_scalars('validation/recall_scores',
                #        disease_recall, e)
                # self.summary_writer.add_scalars('validation/precision_scores',
                #        disease_precision, e)

                self.summary_writer.add_scalar('validation/f1_mean', total_f1, e)
                self.summary_writer.add_scalar('validation/recall_mean', total_recall, e)
                self.summary_writer.add_scalar('validation/precision_mean', total_precision, e)
                self.summary_writer.add_scalars('validation/topk', total_topk, e)
                for i in per_disease_topk:
                    self.summary_writer.add_scalars('validation/topk_'+str(i), per_disease_topk[i], e)

                for i, k in enumerate(np.random.choice(list(range(len(sent_gt))), size=10, replace=False)):
                    self.summary_writer.add_text('validation/sentence_gt'+str(i),
                            ' '.join(sent_gt[k]), e)
                    self.summary_writer.add_text('validation/sentence_pred'+str(i),
                            ' '.join(sent_pred[k]), e)

    def train_iteration(self, train_loader):
        train_loss = 0.0
        accuracy = 0.0
        total_disease_acc = 0.0
        bleu = 0.0
        total_tl1 = 0
        total_tl2 = 0
        total_tl3 = 0
        total_train_loss = 0.0
        losses = []
        for i, (images, labels, f_labels, text) in enumerate(train_loader):
               batch_size = images.size(0)
               images = images.to(self.device)
               labels = labels.to(self.device)
               f_labels = f_labels.to(self.device)
               text = text.to(self.device)
               self.optimizer.zero_grad()
               disease, f_disease, text_pred = self.model(images, text)
               loss1 = self.criterion(disease, labels)
               losses.append(loss1)

               loss2 = self.criterion(f_disease, f_labels)
               losses.append(loss2)

               loss3 = 0.0
               for k in range(text_pred.size(1)):
                   loss3 += self.criterion(text_pred[:, k].squeeze(), text[:, k + 1].squeeze())
               losses.append(loss3)

               # Only consider tasks defined in the task list
               loss = torch.stack(losses)[self.tasks].sum()
               
               loss.backward()
               self.optimizer.step()

               train_loss += loss.item()
               total_train_loss += loss.item()
               total_tl1 += loss1.item()
               total_tl2 += loss2.item()
               total_tl3 += text_loss.item()

               pred = F.log_softmax(f_disease, dim = -1).argmax(dim=-1)
               accuracy += pred.eq(f_labels).sum().item()
               d_pred = F.log_softmax(disease, dim= -1).argmax(dim=-1)
               total_disease_acc += d_pred.eq(labels).sum().item()
               preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
               text1 = text[:, 1:].squeeze().tolist()
               preds1 = preds.tolist()
               tbleu, _, _ = compute_bleu(self.lang, text1, preds1)
               bleu += tbleu

               if i != 0 and i % self.print_every == 0:
                  avg_loss = train_loss / self.print_every
                  accuracy = accuracy / self.print_every
                  total_disease_acc = total_disease_acc / self.print_every
                  avg_text_loss = loss3 / self.print_every
                  bleu = bleu / self.print_every
                  total_train_loss = total_train_loss / self.print_every
                  total_tl1 = total_tl1 / self.print_every
                  total_tl2 = total_tl2 / self.print_every
                  total_tl3 = total_tl3 / self.print_every

                  print('Iter:{}\tTraining Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}\tTextLoss:{:.8f}'.format(i, avg_loss,
                              accuracy/batch_size,
                              total_disease_acc / batch_size,
                              bleu,
                              loss3.item()))

                  train_loss = 0.0
                  text_loss = 0.0
                  accuracy = 0.0
                  total_disease_acc = 0.0
                  bleu = 0.0
                  total_tl1 = 0
                  total_tl2 = 0
                  total_tl3 = 0
                  total_train_loss = 0.0
        return (total_train_loss, total_tl1, total_tl2, total_tl3, total_disease_acc/batch_size, accuracy/batch_size, bleu)

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
        for i, (images, labels, f_labels, text) in enumerate(val_loader):
            batch_size = images.size(0)
            images = images.to(self.device)
            labels = labels.to(self.device)
            f_labels = f_labels.to(self.device)
            text = text.to(self.device)
            diseases, fine_diseases, text_pred = self.model(images, text)
            loss1 = self.criterion(diseases, labels)
            losses.append(loss1)
            loss2 = self.criterion(fine_diseases, f_labels)
            losses.append(loss2)
            text_loss = 0.0
            for k in range(text_pred.size(1)):
                text_loss += self.criterion(text_pred[:,k].squeeze(), text[:,k + 1].squeeze())
            losses.append(text_loss)

            val_loss = torch.stack(losses)[self.tasks].sum()

            # Evaluation of P, R, F1, BLEU
            preds = F.log_softmax(fine_diseases, dim = -1)
            pred = preds.argmax(dim=-1)
            d_pred = F.log_softmax(diseases, dim = -1).argmax(dim=-1)
            total_acc += (pred.eq(f_labels).sum().item() / batch_size)
            total_d_acc += (d_pred.eq(labels).sum().item() / batch_size)
            acc, recall, precision, f1 = accuracy_recall_precision_f1(d_pred,
                    labels)

            for k in k_vals:
                total_topk[k] += compute_topk(preds, f_labels, k)
                for d in [0, 1, 2, 3]:
                    mask = labels.eq(d)
                    if mask.sum() > 0:
                        per_disease_topk[d][str(k)] += compute_topk(preds[mask], f_labels[mask], k)

            total_recall += np.mean(recall)
            total_precision += np.mean(precision)
            total_f1 += np.mean(f1)
            preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
            text1 = text[:, 1:].squeeze().tolist()
            preds1 = preds.tolist()
            t_bleu, sent_gt, sent_pred = compute_bleu(self.lang, text1, preds1)

            # Book-keeping
            bleu += t_bleu
            total_l1 += loss1.item()
            total_l2 += loss2.item()
            total_l3 += text_loss.item()
            # cm = calculate_confusion_matrix(pred, labels)
            # total_cm += (cm / batch_size)
        bleu = bleu / (len(val_loader))
        val_loss = val_loss / len(val_loader)
        total_l1 /= len(val_loader)
        total_l2 /= len(val_loader)
        total_l3 /= len(val_loader)
        total_acc = total_acc / len(val_loader)
        total_d_acc = total_d_acc / len(val_loader)
        total_f1 = total_f1 / len(val_loader)
        total_precision = total_precision / len(val_loader)
        total_recall = total_recall / len(val_loader)
        # total_cm = total_cm / len(val_loader)

        self.scheduler.step(val_loss)
        if val_loss <= self.min_val_loss:
           torch.save(self.model.state_dict(), self.save_path)
           self.min_val_loss = val_loss

        disease_f1 = {}
        disease_precision = {}
        disease_recall = {}

        #for i in range(len(total_f1)):
        #   disease_f1[i] = total_f1[i]
        #   disease_precision[i] = total_precision[i]
        #   disease_recall[i] = total_recall[i]

        total_topk = {str(k) : total_topk[k] / len(val_loader) for k in k_vals}
        for d in [0,1,2,3]:
            for k in k_vals:
                per_disease_topk[d][str(k)] = per_disease_topk[d][str(k)] / len(val_loader)

        # print('-----------CM------------')
        # print(total_cm)
        # print('-----------------------')
        return (val_loss, total_d_acc, total_acc, bleu, total_f1, total_recall,
                total_precision, sent_gt, sent_pred, total_topk, per_disease_topk) 
