import os
import gin
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

from trainer import MultiTaskTrainer, SmallTrainer
from models import MultiTaskModel
from models import AbnormalNet
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

    # trainval_from_images = CustomDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)
    # test_dataset = CustomDatasetFromImages(test_csv_path, data_dir = dataset_dir)
    trainval_from_images = GradedDatasetFromImages(trainval_csv_path, data_dir = dataset_dir)

    dset_len = len(trainval_from_images)
    val_size = int(val_split * dset_len)
    test_size = int(0.15 * dset_len)
    train_size = dset_len - val_size - test_size


    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(trainval_from_images,
                                                               [train_size,
                                                                val_size,
                                                                test_size])

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

    # lang = train_dataset.dataset.get_lang()

    # model = MnistCNNModel()
    if model_type == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_type == 'vgg19':
        model = models.vgg19(pretrained=True)

    # model = models.googlenet(pretrained=True)
    # model = MultiTaskModel(model, vocab_size=lang.n_words, model_type =  model_type)
    model = AbnormalNet() 
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('models/2019-09-27 07:17:11.148440/best_model.pt'))
    # All Device training
    # model.load_state_dict(torch.load('models/0_1_2-2019-09-3007:56:52.568630/best_model.pt'))
    # BEST WITH NORMAL MODEL
    # model.load_state_dict(torch.load('models/2019-09-26 06:39:37.468635/best_model.pt'))
    # model.load_state_dict(torch.load('models/2019-09-2609:17:12.0260540_1_2/best_model.pt'))
    # OISCapture

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
    # trainer = MultiTaskTrainer(model, optimizer, scheduler, criterion, tasks, epochs, lang, print_every = print_every)
    trainset_percent = (1 - val_split - 0.15)
    trainer = SmallTrainer(model, optimizer, scheduler, criterion, epochs,
            print_every =  print_every, trainset = trainset_percent)
    trainer.train(train_loader, val_loader)

    model.load_state_dict(torch.load(os.path.join(trainer.save_location_dir,'best_model.pt'))

    # val_loss, total_d_acc, total_acc, bleu, total_f1,total_recall, total_precision, sent_gt, sent_pred, total_topk,per_disease_topk, per_disease_bleu, total_cm = trainer.validate(test_loader)
    # with open(trainer.output_log, 'a+') as out:
    #    print('Test Loss:{:.8f}\tAcc:{:.8f}\tDAcc:{:.8f}\tBLEU:{:.8f}'.format(val_loss, total_acc, total_d_acc, bleu), file=out)
    #    print('total_topk',total_topk, file=out)
    #    print('per_disease_topk', per_disease_topk, file=out)
    #    print('per_disease_bleu', per_disease_bleu, file=out)
    #    print(total_cm, file=out)
    #    for k in np.random.choice(list(range(len(sent_gt))), size=10, replace=False):
    #        print(sent_gt[k], file=out)
    #        print(sent_pred[k], file=out)
    #        print('---------------------', file=out)
    # trainer.test(test_loader)
    val_loss, total_d_acc, total_f1, total_recall, total_precision, total_cm = trainer.validate(test_loader)
    with open(trainer.output_log, 'a+') as out:
        print('Test Loss',val_loss,'total_d_acc',total_d_acc, 'F1', total_f1, 'R', total_recall,'P', total_precision, file=out)
        print(total_cm, file=out)

    # trainer.test(test_loader)

if __name__ == "__main__":
    # task_configs =[[0],[1],[2],[0,1], [1,2],[0,2],[0, 1, 2]]
    task_configs = [ 0.15, 0.25, 0.4, 0.55, 0.7, 0.85]
    for i, t in enumerate(task_configs):
        print("Running", (1 - t - 0.15))
    # gin.parse_config_file('config.gin')
        gin.parse_config_file('config_small.gin')
        gin.bind_parameter('run.val_split', task_configs[i])
        run()
        gin.clear_config()

