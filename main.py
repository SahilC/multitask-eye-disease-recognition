import torch
import torch.nn as nn
import torchvision.models as models

from trainer import Trainer
from models import MultiTaskModel
from dataset import CustomDatasetFromImages
def run():
    batch_size = 64
    epochs = 20
    val_split = 0.15
    num_workers = 32
    print_every = 100
    class_type = 'fine-grained-disease'
    csv_path = 'trainset.csv'

    # custom_from_images =  CustomDatasetFromImages('all_data_filtered.csv', class_type=class_type)
    custom_from_images = CustomDatasetFromImages(csv_path, class_type=class_type)

    dset_len = len(custom_from_images)
    test_size = int(val_split * dset_len)
    val_size = int(val_split * dset_len)
    train_size = dset_len - val_size - test_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_from_images,
                                                                             [train_size,
                                                                              val_size,
                                                                              test_size])


    lang1 = train_dataset.get_lang()

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


    # model = MnistCNNModel()
    # model = models.densenet121(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # model = models.resnet101(pretrained=True)
    # model = models.resnet34(pretrained=True)
    model = models.vgg19(pretrained=True)
    model = MultiTaskModel(model, vocab_size=lang1.n_words)
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load('best_model.pth'))

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                weight_decay=1e-6,
                                momentum=0.9,
                                lr=1e-3,
                                nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=3,
                                  min_lr=1e-7,
                                  verbose=True)
    trainer = Trainer(model, optimizer, scheduler, criterion, epochs, print_every =
            print_every)
    trainer.train(train_loader, val_loader, test_loader)

if __name__ == "__main__":
    run()
