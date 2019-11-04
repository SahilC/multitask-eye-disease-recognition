import os
import gin
import numpy as np
import copy

import torch
import torch.nn as nn
import torchvision.models as models

from trainer import MultiTaskTrainer
from models import MultiTaskModel
from dataset import CustomDatasetFromImages
from dataset import GradedDatasetFromImages

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hacks for Reproducibility
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# from cnn_model import MnistCNNModel
@gin.configurable
def run(batch_size, epochs, val_split, num_workers, print_every,
        trainval_csv_path, test_csv_path, model_type, tasks, lr, weight_decay, 
        momentum, dataset_dir):

    all_dataset = CustomDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)
    # test_dataset = CustomDatasetFromImages(test_csv_path, data_dir = dataset_dir)
    val_from_images = GradedDatasetFromImages(test_csv_path, data_dir = dataset_dir)

    dset_len = len(all_dataset)
    val_size = int(val_split * dset_len)
    test_size = int(0.15 * dset_len)
    train_size = dset_len - val_size


    train_dataset, val_dataset = torch.utils.data.random_split(all_dataset,
                                                               [train_size,
                                                                val_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=2 * batch_size,
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
    test_loader = torch.utils.data.DataLoader(dataset=val_from_images,
                                              batch_size=batch_size,
                                              pin_memory=False,
                                              drop_last=True,
                                              shuffle=True,
                                              num_workers=num_workers)

    lang = train_dataset.dataset.get_lang()

    if model_type == 'densenet121':
        model = models.densenet121(pretrained=False)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=False)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif model_type == 'vgg19':
        model = models.vgg19(pretrained=False)

    model = MultiTaskModel(model, vocab_size=lang.n_words, model_type = model_type)

    model = nn.DataParallel(model)

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

    trainer = MultiTaskTrainer(model, optimizer, scheduler, criterion, tasks, epochs, lang, print_every = print_every)

    trainer.train(train_loader, val_loader)

    val_loss, total_d_acc, total_acc, bleu, total_f1,total_recall, total_precision, sent_gt, sent_pred, total_topk,per_disease_topk, per_disease_bleu, total_cm = trainer.validate(test_loader)
    with open(trainer.output_log, 'a+') as out:
        print('Test Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(val_loss, total_acc, total_d_acc, bleu), file=out)
        print('total_topk',total_topk, file=out)
        print('per_disease_topk', per_disease_topk, file=out)
        print('per_disease_bleu', per_disease_bleu, file=out)
        print(total_cm, file=out)
        for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
            print(sent_gt[k], file=out)
            print(sent_pred[k], file=out)
            print('---------------------', file=out)
    trainer.test(test_loader)

if __name__ == "__main__":
    task_configs =[[0],[1],[2],[0,1], [1,2],[0,2],[0, 1, 2]]
    for i, t in enumerate(task_configs):
        print("Running", t)
        gin.parse_config_file('config.gin')
        gin.bind_parameter('run.tasks', t)
        run()
        gin.clear_config()

