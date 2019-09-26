import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class LanguageModel(nn.Module):
    def __init__(self, vocab_size = 193, embed_size = 256, inp_size = 1024, hidden_size = 512,
            num_layers = 1, dropout_p = 0.1):
        super(LanguageModel, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=2)
        self.project = nn.Linear(inp_size, hidden_size)
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
            use_teacher_forcing = True if (self.training and random.random() < self.teacher_forcing_ratio) else False
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


class AbnormalNet(nn.Module):
    def __init__(self):
        super(AbnormalNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, (4, 4), 2, 1), # 64 -> 32
                                   nn.PReLU(),
                                   nn.Conv2d(32, 64, (4, 4), 2,1), # 32 -> 16
                                   nn.PReLU(),
                                   nn.Conv2d(64, 64, (4, 4), 2, 1), # 16 -> 8
                                   nn.PReLU(),
                                   nn.Conv2d(64, 2, (8, 8), 1, 0)) 

    def forward(self, x):
        return self.conv(x).squeeze()

@gin.configurable
class MultiTaskModel(nn.Module):
    def __init__(self, model, vocab_size, model_type = 'densenet121', in_feats = gin.REQUIRED):
        super(MultiTaskModel, self).__init__()
        self.model_type = model_type
        if self.model_type == 'densenet121':
            self.feature_extract = model.features
        else:
            self.feature_extract = torch.nn.Sequential(*list(model.children())[:-1])

        self.disease_classifier = nn.Sequential(nn.Linear(in_feats, 512),
                nn.ReLU(), nn.Linear(512, 5))
        self.fine_disease_classifier = nn.Sequential(nn.Linear(in_feats, 512),
                nn.ReLU(), nn.Linear(512, 321))
        self.language_classifier = LanguageModel(inp_size = in_feats, vocab_size = vocab_size)

    def forward(self, data,text):
        features = self.feature_extract(data).squeeze()
        out = F.relu(features)
        if self.model_type == 'densenet121':
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)

        return (self.disease_classifier(out),
                self.fine_disease_classifier(out),
                self.language_classifier(out, text)
                )
