import gin
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
from collections import defaultdict
from utils import compute_bleu, compute_topk, accuracy_recall_precision_f1, calculate_confusion_matrix
class BaseTrainer(object):
    def __init__(self, model, optimizer, scheduler, criterion, epochs, print_every, min_val_loss = 100):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.criterion = criterion
        self.print_every = print_every
        self.min_val_loss = min_val_loss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Save experiment configuration 
        self.save_location_dir = os.path.join('models', str(datetime.now()).replace(' ',''))

    def init_saves(self):
        if not os.path.exists(self.save_location_dir):
            os.mkdir(self.save_location_dir)
        with open(os.path.join(self.save_location_dir,'config.gin'), 'w') as conf:
            conf.write(gin.operative_config_str())
        self.output_log = os.path.join(self.save_location_dir,'output_log.txt')
        self.save_path = os.path.join(self.save_location_dir, 'best_model.pt')
        self.summary_writer =  SummaryWriter(os.path.join(self.save_location_dir, 'logs'), 300)

    def train(self, train_loader, val_loader):
        raise NotImplementedError

    def train_iteration(self, train_loader, val_loader):
        raise NotImplementedError

    def validate(self, train_loader, val_loader):
        raise NotImplementedError

    def test(self, test_loader):
        raise NotImplementedError


class MultiTaskTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, criterion, tasks, epochs, lang, print_every = 100, min_val_loss = 100):
        super(Trainer, self).__init__(self, model, optimizer, scheduler, criterion, epochs, print_every, min_val_loss)
        self.lang = lang
        self.tasks = tasks
        self.save_location_dir = os.path.join('models', '_'.join(str(t) for t in self.tasks) +'-'+ str(datetime.now()).replace(' ',''))
        self.init_saves()

    def train(self, train_loader, val_loader):
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
                val_loss, total_d_acc, total_acc, bleu, total_f1, total_recall, total_precision, sent_gt, sent_pred, total_topk, per_disease_topk, per_disease_bleu, total_cm = self.validate(val_loader)
                with open(self.output_log, 'a+') as out:
                    print('Epoch: {}\tVal Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(e,val_loss, total_acc, total_d_acc, bleu), file=out)
                    print('total_topk',total_topk, file=out)
                    print('per_disease_topk', per_disease_topk, file=out)
                    print('per_disease_bleu', per_disease_bleu, file=out)
                    print(total_cm, file=out)
                    for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
                        print(sent_gt[k], file=out)
                        print(sent_pred[k], file=out)
                        print('---------------------', file=out)

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

    def test(self, test_loader):
        results = open('predictions.csv','w')
        ind2word = {0:'Melanoma',1:'Glaucoma',2:'AMD',3:'DR',4:'Normal'}
        with torch.no_grad():
            for i, (name, images, labels, f_labels, text) in enumerate(test_loader):
                    batch_size = images.size(0)
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    f_labels = f_labels.to(self.device)
                    text = text.to(self.device)
                    diseases, fine_diseases, text_pred = self.model(images, text)
                    pred = F.log_softmax(diseases, dim= -1).argmax(dim=-1)
                    for j in range(diseases.size(0)):
                        results.write(name[j]+','+ind2word[labels[j].item()] +','+ind2word[pred[j].item()] +'\n')


    def train_iteration(self, train_loader):
        train_loss = 0.0
        accuracy = 0.0
        total_disease_acc = 0.0
        bleu = 0.0
        total_tl1 = 0
        total_tl2 = 0
        total_tl3 = 0
        total_train_loss = 0.0
        loss = torch.tensor(0).to(self.device)
        for i, (_, images, labels, f_labels, text) in enumerate(train_loader):
               batch_size = images.size(0)
               images = images.to(self.device)
               labels = labels.to(self.device)
               f_labels = f_labels.to(self.device)
               text = text.to(self.device)
               self.optimizer.zero_grad()
               disease, f_disease, text_pred = self.model(images, text)
               loss1 = self.criterion(disease, labels)

               loss2 = self.criterion(f_disease, f_labels)

               loss3 = 0.0
               for k in range(text_pred.size(1)):
                   loss3 += self.criterion(text_pred[:, k].squeeze(), text[:, k + 1].squeeze())

               # Only consider tasks defined in the task list
               loss = torch.stack((loss1,loss2, loss3))[self.tasks].sum()
               
               loss.backward()
               self.optimizer.step()

               train_loss += loss.item()
               total_train_loss += loss.item()
               total_tl1 += loss1.item()
               total_tl2 += loss2.item()
               total_tl3 += loss3.item()

               pred = F.log_softmax(f_disease, dim = -1).argmax(dim=-1)
               accuracy += pred.eq(f_labels).sum().item()
               d_pred = F.log_softmax(disease, dim= -1).argmax(dim=-1)
               total_disease_acc += d_pred.eq(labels).sum().item()
               # preds = torch.argmax(F.log_softmax(text_pred,dim=-1), dim=-1)
               # text1 = text[:, 1:].squeeze().tolist()
               # preds1 = preds.tolist()
               # tbleu, _, _ = compute_bleu(self.lang, text1, preds1, labels, per_disease_bleu)
               # bleu += tbleu

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
        per_disease_bleu = defaultdict(list)
        with torch.no_grad():
            for i, (_, images, labels, f_labels, text) in enumerate(val_loader):
                batch_size = images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device)
                f_labels = f_labels.to(self.device)
                text = text.to(self.device)
                diseases, fine_diseases, text_pred = self.model(images, text)
                loss1 = self.criterion(diseases, labels)
                loss2 = self.criterion(fine_diseases, f_labels)
                text_loss = 0.0
                for k in range(text_pred.size(1)):
                    text_loss += self.criterion(text_pred[:,k].squeeze(), text[:,k + 1].squeeze())

                val_loss += torch.stack((loss1, loss2, text_loss))[self.tasks].sum()

                preds = F.log_softmax(fine_diseases, dim = -1)
                pred = preds.argmax(dim=-1)
                d_pred = F.log_softmax(diseases, dim = -1).argmax(dim=-1)

                # Evaluation of P, R, F1, CM, BLEU
                total_acc += (pred.eq(f_labels).sum().item() / batch_size)
                total_d_acc += (d_pred.eq(labels).sum().item() / batch_size)
                acc, recall, precision, f1 = accuracy_recall_precision_f1(d_pred,
                        labels)
                cm = calculate_confusion_matrix(d_pred, labels)
                try:
                    total_cm += (cm / batch_size)
                except:
                    print("Error occured for this CM")
                    print(cm / batch_size)

                # Top-k evaluation
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
                t_bleu, sent_gt, sent_pred = compute_bleu(self.lang, text1, preds1, labels, per_disease_bleu)

                # Book-keeping
                bleu += t_bleu
                total_l1 += loss1.item()
                total_l2 += loss2.item()
                total_l3 += text_loss.item()
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
        total_cm = total_cm / len(val_loader)

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
        for d in per_disease_bleu:
            per_disease_bleu[d] = np.mean(per_disease_bleu[d])

        total_topk = {str(k) : total_topk[k] / len(val_loader) for k in k_vals}
        for d in [0,1,2,3]:
            for k in k_vals:
                per_disease_topk[d][str(k)] = per_disease_topk[d][str(k)] / len(val_loader)

        return (val_loss, total_d_acc, total_acc, bleu, total_f1, total_recall,
                total_precision, sent_gt, sent_pred, total_topk,
                per_disease_topk, per_disease_bleu, total_cm)


class SmallTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, criterion, epochs, print_every = 100, min_val_loss = 100, trainset_split = 0.85):
        super(SmallTrainer, self).__init__(self, model, optimizer, scheduler, criterion, print_every, min_val_loss)
        self.save_location_dir = os.path.join('models', str(trainset_split)+'-'+str(datetime.now()))
        self.init_saves()

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
                with open(self.output_log, 'a+') as out: 
                    print('Val Loss',val_loss, 'total_d_acc',total_d_acc, 'F1',
                            total_f1, 'R', total_recall,'P', total_precision,
                            file=out)
                    print(total_cm, file=out)

    def test(self, test_loader):
        results = open('self_trained_extra_labels.csv','w')
        self.model.eval()
        # ind2disease = {0:'Disease',1:'Normal'}
        ind2disease = {0:'Melanoma' , 1: 'Glaucoma', 2: 'AMD', 3:'DR', 4:'Normal'}
        # ind2disease2 = {0:'Melanoma' , 1: 'Glaucoma', 2: 'AMD', 3:'DR'}
        ind2disease2 = {0:'not applicable' , 1: 'not classifed', 2: 'diabetes no retinopathy'}
        for i, data in tqdm.tqdm(enumerate(test_loader)):
               image_name = data[0]
               images = data[1]
               labels = data[2]
               batch_size = images.size(0)
               images = images.to(self.device)
               disease = self.model(images)
               d_pred = F.log_softmax(disease, dim= -1).argmax(dim=-1)
               probs, _ = F.softmax(disease, dim=-1).max(dim=-1)
               for j in range(d_pred.size(0)):
                   results.write(image_name[j]+','+ '{:.8f}'.format(probs[j].item()) +',' + ind2disease2[labels[j].item()] +','+ind2disease[d_pred[j].item()]+'\n')

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

            val_loss += loss1

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
