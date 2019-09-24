import os
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from trainer import Trainer
from models import MultiTaskModel
from dataset import CustomDatasetFromImages

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hacks for Reproducibility
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES']='2, 3'

# from cnn_model import MnistCNNModel

def run():
    batch_size = 64
    epochs = 20
    val_split = 0.15
    num_workers = 32
    print_every = 100
    trainval_csv_path = 'trainset.csv'
    test_csv_path = 'testset.csv'

    lr = 1e-3
    weight_decay = 1e-6
    momentum = 0.9
    dataset_dir = '/data2/fundus_images/'

    trainval_from_images = CustomDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)
    test_dataset = CustomDatasetFromImages(test_csv_path, data_dir = dataset_dir)

    dset_len = len(trainval_from_images)
    val_size = int(val_split * dset_len)
    train_size = dset_len - val_size


    train_dataset, val_dataset = torch.utils.data.random_split(trainval_from_images,
                                                               [train_size,
                                                                val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               pin_memory=False,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             pin_memory=False,
                                             drop_last=True,
                                             shuffle=True, 
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              pin_memory=False,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=num_workers)

    lang = train_dataset.dataset.get_lang()

    # model = MnistCNNModel()
    # model = models.densenet121(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.resnet101(pretrained=True)
    # model = models.resnet34(pretrained=True)
    model = models.vgg19(pretrained=True)
    model = MultiTaskModel(model, vocab_size=lang.n_words)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('best_model.pth'))

    print(model)

    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=weight_decay,
                                momentum=momentum,
                                lr=lr,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    trainer = Trainer(model, optimizer, scheduler, criterion, epochs, lang, print_every =
            print_every)
    trainer.train(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    run()
