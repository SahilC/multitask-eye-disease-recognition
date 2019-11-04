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

@gin.configurable 
class AutoEncoder(nn.Module):
    def __init__(self, model_type, model = None):
        super(AutoEncoder, self).__init__()
        self.model_type = model_type
        if model_type == 'self':
            self.conv = nn.Sequential(nn.Conv2d(3, 32, (4, 4), 2, 1), # 224 -> 112
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, (4, 4), 2,1), # 112 -> 56
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, (4, 4), 2, 1), # 56 -> 28
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, (4, 4), 2, 1), # 28 -> 14
                                   nn.ReLU(),
                                   nn.Conv2d(128, 64, (4, 4), 2, 1), # 14 -> 7
                                   nn.ReLU(),
                                   nn.Conv2d(64, 5, (7, 7), 1, 0))
        else:
            self.conv = torch.nn.Sequential(*list(model.children())[:-1])
            self.lin = nn.Linear(2048, 256)
            self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(256, 256, 7, 1, 0),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    # state size: (ngf * 8) x 4 x 4
                    nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    # state size: ngf x 32 x 32
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    # state size: (ngf * 4) x 8 x 8
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    # state size: (ngf * 2) x 16 x 16
                    nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    # state size: ngf x 32 x 32
                    nn.ConvTranspose2d(32, 3, 4, 2, 1),
                    nn.Tanh()
           )

    def forward(self, x):
        out = self.conv(x).squeeze()
        if self.model_type != 'self':
            out = self.lin(F.relu(out))
        out = self.deconv(out.view(-1, out.size(1), 1, 1))
        return out

@gin.configurable 
class AbnormalNet(nn.Module):
    def __init__(self, model_type, model = None):
        super(AbnormalNet, self).__init__()
        self.model_type = model_type
        if model_type == 'self':
            self.conv = nn.Sequential(nn.Conv2d(3, 32, (4, 4), 2, 1), # 224 -> 112
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, (4, 4), 2,1), # 112 -> 56
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, (4, 4), 2, 1), # 56 -> 28
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, (4, 4), 2, 1), # 28 -> 14
                                   nn.ReLU(),
                                   nn.Conv2d(128, 64, (4, 4), 2, 1), # 14 -> 7
                                   nn.ReLU(),
                                   nn.Conv2d(64, 5, (7, 7), 1, 0))
        else:
            self.conv = torch.nn.Sequential(*list(model.children())[:-1])
            self.lin = nn.Linear(2048, 5)

    def forward(self, x):
        out = self.conv(x).squeeze()
        if self.model_type != 'self':
            out = self.lin(F.relu(out))
        return out

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

    def forward(self, data, text):
        features = self.feature_extract(data).squeeze()
        out = F.relu(features)
        if self.model_type == 'densenet121':
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)

        return (self.disease_classifier(out),
                self.fine_disease_classifier(out),
                self.language_classifier(out, text)
                )
