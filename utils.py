import torch
from nltk import word_tokenize
import re
import unicodedata
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, confusion_matrix

import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
import torch


SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"UNK":3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD", 3:"UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    # s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lines, max_length, lang1='eng'):
    print("Reading lines...")
    input_lang = Lang(lang1)

    pairs = []
    for e,l in enumerate(lines):
        val = []
        words_list = word_tokenize(l)
        if len(words_list) >= max_length:
            val.append(normalizeString(' '.join(words_list[:max_length - 1])))
        else:
            val.append(normalizeString(l))
        input_lang.addSentence(val[0])
        pairs.append(val)

    print("Read %s sentence pairs" % len(pairs))
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(input_lang.word2count)
    return input_lang, pairs

def indexFromSentence(lang, sentence, unk_ratio = 15, max_length = 15):
    out = []
    out.append(SOS_token)
    # out = [lang.word2index[word] for word in word_tokenize(sentence)]

    # If word freq is small then replace with UNK
    for word in word_tokenize(normalizeString(sentence)):
        try:
            if lang.word2count[word] > unk_ratio:
                out.append(lang.word2index[word])
            else:
                out.append(lang.word2index['UNK'])
        except:
            pass
            # print("error while processing word", word)
            # out.append(lang.word2index['UNK'])
    out.append(EOS_token)

    if len(out) > max_length:
        sentence_length = max_length
    else:
        sentence_length = len(out)

    # If sentence is small then pad
    if len(out) < max_length:
        for i in range(max_length - len(out)):
            out.append(PAD_token)
    elif len(out) > max_length:
        out = out[:max_length]
    return out, sentence_length

def variableFromSentence(lang, sentence, max_length = 15):
    indexes, sentence_length = indexesFromSentence(lang, sentence, max_length = max_length)
    result = torch.LongTensor(indexes).view(-1, 1)
    return result, sentence_length

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    return input_variable

#Metrics
def accuracy_recall_precision_f1(y_pred, y_target):

    """Computes the accuracy, recall, precision and f1 score for given predictions and targets
    Args:
        y_pred: Logits of the predictions for each class
        y_target: Target values
    """

    predictions = y_pred.cpu().detach().numpy()
    y_target = y_target.cpu().numpy()

    correct = np.sum(predictions == y_target)
    accuracy = correct / len(predictions)

    recall = recall_score(y_target, predictions, average=None) #average=None (the scores for each class are returned)
    precision = precision_score(y_target, predictions, average=None)
    f1 = f1_score(y_target, predictions, average=None)

    return accuracy, recall, precision, f1

def calculate_confusion_matrix(y_pred, y_target):

    predictions = y_pred.cpu().detach().numpy()
    y_target = y_target.cpu().numpy()

    #Confusion matrix
    cm = confusion_matrix(y_target, predictions)

    #multi_cm = multilabel_confusion_matrix(y_target, predictions)
    #print(multi_cm)
    #print(confusion_matrix(y_target, predictions))

    #Classification report
    #print(classification_report(y_target, predictions))

    return cm

def compute_topk(topk_vals, gt, k):
    _, preds = topk_vals.topk(k = k, dim = 1)
    topk_acc = 0
    for i in range(preds.size(1)):
        topk_acc += preds[:, i].eq(gt).sum().item()
    return (topk_acc / topk_vals.size(0))

def compute_bleu(lang, text1, preds1):
    ind2word = lang.index2word
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
        if len(sent2) > 0 and len(sent2) < 4 and weights  == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / len(sent2),) * len(sent2)
        c_bleu = sentence_bleu([sent1], sent2, weights = weights)
        sents_gt.append(sent1)
        sents_pred.append(sent2)
        bleu += c_bleu
    return (bleu/len(text1)), sents_gt, sents_pred
