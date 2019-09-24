from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime
class Trainer(object):
    def __init__(self, model, optimizer, scheduler, epochs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.save_location_dir = os.path.join('models', datetime.now())
        if not os.path.exists(self.save_location):
            os.mkdir(self.save_location)
        self.save_path = os.path.join(self.save_location_dir, 'best_model.pt')
        self.summary_writer =  SummaryWriter(os.path.join(self.save_location_dir, 'logs'), 300)

    def train(self):
        for e in range(self.epochs):
                model.train()
                train_loss = 0.0
                total_train_loss = 0.0
                accuracy = 0.0
                total_disease_acc = 0.0
                bleu = 0.0
                total_tl1 = 0
                total_tl2 = 0
                total_tl3 = 0
                for i, (images, labels, f_labels, text) in enumerate(self.train_loader):
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
        total_tl1 /= len(train_loader)
        total_tl2 /= len(train_loader)
        total_tl3 /= len(train_loader)
        total_train_loss = total_train_loss / len(train_loader)
        self.summary_writer.add_scalar('training/total_train_loss', total_train_loss, e)
        self.summary_writer.add_scalar('training/total_t1_loss', total_tl1, e)
        self.summary_writer.add_scalar('training/total_t2_loss', total_tl2, e)
        self.summary_writer.add_scalar('training/total_t3_loss', total_tl3, e)
        self.summary_writer.add_scalar('training/t1_acc', total_disease_acc / batch_size, e)
        self.summary_writer.add_scalar('training/t2_acc', accuracy / batch_size, e)
        self.summary_writer.add_scalar('training/t3_bleu', bleu, e)

    def validate(self, epoch = 0):
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
        if val_loss <= min_val_loss:
           torch.save(model.state_dict(), self.save_path)
           min_val_loss = val_loss
        self.summary_writer.add_scalar('validation/val_loss', val_loss, epoch)
        self.summary_writer.add_scalar('validation/t1_acc', total_d_acc, epoch)
        self.summary_writer.add_scalar('validation/t2_acc', total_acc, epoch)
        self.summary_writer.add_scalar('validation/BLEU', bleu, epoch)

        disease_f1 = {}
        disease_precision = {}
        disease_recall = {}

        for i in range(len(total_f1[0])):
            disease_f1[i] = total_f1[i]
            disease_precision[i] = total_precision[i]
            disease_recall[i] = total_recall[i]
        self.summary_writer.add_scalars('validation/f1_scores', disease_f1, epoch)
        self.summary_writer.add_scalars('validation/recall_scores',
                disease_recall, epoch)
        self.summary_writer.add_scalars('validation/precision_scores',
                disease_precision, epoch)
        
        self.summary_writer.add_scalars('validation/f1_mean', np.mean(total_f1), epoch)
        self.summary_writer.add_scalars('validation/recall_mean', np.mean(total_recall), epoch)
        self.summary_writer.add_scalars('validation/precision_mean',np.mean(total_precision), epoch)
    
        # print('-----------CM------------')
        # print(total_cm)
        # print('-----------------------')
        for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
            self.summary_writer.add_text('validation/sentence_gt'+str(k),
                    sent_gt[k], epoch)
            self.summary_writer.add_text('validation/sentence_pred'+str(k),
                    sent_preds[k], epoch)
