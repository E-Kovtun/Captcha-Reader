import torch
from torch import nn
from model import CaptchaNet
import os
from reader import CaptchaReader
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from utils.metrics import captcha_len, num_classes, captcha_accuracy
from tqdm import tqdm


def train_net(prepared_data_folder, train_folder, valid_folder,
              num_epochs, batch_size, dataloader_num_workers,
              optimizer_lr, weight_decay, scheduler_factor, scheduler_patience, early_stopping_patience):
    model_name = 'resnet18_synthetic'
    results_folder = './results/'
    os.makedirs(results_folder, exist_ok=True)
    checkpoint = os.path.join(results_folder, f'{model_name}_checkpoint.pt')

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = CaptchaReader(prepared_data_folder, train_folder)
    valid_dataset = CaptchaReader(prepared_data_folder, valid_folder)
    # test_dataset = CaptchaReader(prepared_data_folder, test_folder)

    net = CaptchaNet().to(device)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_num_workers)

    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True, path=checkpoint)

    for epoch in range(1, num_epochs + 1):
        net.train(True)
        epoch_train_loss = 0
        print('Training...')
        for batch_images, batch_targets in tqdm(train_dataloader):
            batch_images, batch_target = batch_images.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            output = net(batch_images)
            sample_size = output.shape[0]
            loss = cross_entropy_loss(output.reshape(sample_size*captcha_len, num_classes),
                                      batch_target.reshape(sample_size*captcha_len))
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}/{num_epochs} || Train loss {epoch_train_loss}')

        print('Validation...')
        net.train(False)
        epoch_valid_loss = 0
        for batch_images, batch_targets in tqdm(valid_dataloader):
            batch_images, batch_target = batch_images.to(device), batch_targets.to(device)
            output = net(batch_images)
            sample_size = output.shape[0]
            loss = cross_entropy_loss(output.reshape(sample_size*captcha_len, num_classes),
                                      batch_target.reshape(sample_size*captcha_len))
            epoch_valid_loss += loss.item()

        print(f'Epoch {epoch}/{num_epochs} || Valid loss {epoch_valid_loss}')

        scheduler.step(epoch_valid_loss)

        early_stopping(epoch_valid_loss, net)
        if early_stopping.early_stop:
            print('Early stopping')
            break

#----------------------------------------------
    # net = CaptchaNet().to(device)
    #
    # net.load_state_dict(torch.load(checkpoint, map_location=device))
    # net.train(False)
    # print('Testing...')
    #
    # all_predictions = []
    # all_gt = []
    # for batch_images, batch_targets in test_dataloader:
    #     batch_images, batch_target = batch_images.to(device), batch_targets.to(device)
    #     output = net(batch_images)
    #     predicted_classes = torch.argmax(output, dim=2)
    #     all_predictions.extend(predicted_classes.detach().cpu().tolist())
    #     all_gt.extend(batch_targets.detach().cpu().tolist())
    #
    # test_acc = captcha_accuracy(all_predictions, all_gt)
    # print(f'Test Accuracy {test_acc}')


if __name__ == '__main__':
    # prepared_data_folder = './prepared_datasets/'
    # train_folder = 'train_dataset'
    # valid_folder = 'valid_dataset'
    # test_folder = 'test_dataset'

    prepared_data_folder = './synthetic_datasets/'
    train_folder = 'train_synthetic'
    valid_folder = 'valid_synthetic'

    num_epochs = 600
    batch_size = 32
    dataloader_num_workers = 2

    optimizer_lr = 1e-3
    weight_decay = 1e-5

    scheduler_factor = 0.6
    scheduler_patience = 4

    early_stopping_patience = 10

    train_net(prepared_data_folder, train_folder, valid_folder,
              num_epochs, batch_size, dataloader_num_workers,
              optimizer_lr, weight_decay, scheduler_factor, scheduler_patience, early_stopping_patience)
