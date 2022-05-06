import torch
from torch import nn
from model import CaptchaNet
import os
from reader import CaptchaReader
from torch.utils.data import DataLoader
from utils.earlystopping import EarlyStopping
from utils.metrics import captcha_len, num_classes, captcha_accuracy
from tqdm import tqdm
import json
import sys


def train_net(train_path, valid_path,
              num_epochs, batch_size, dataloader_num_workers,
              optimizer_lr, scheduler_factor, scheduler_patience, early_stopping_patience,
              checkpoint_name, pretrained_checkpoint_name=None):
    results_folder = './results/'
    os.makedirs(results_folder, exist_ok=True)
    checkpoint = os.path.join(results_folder, checkpoint_name)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    train_dataset = CaptchaReader(train_path)
    valid_dataset = CaptchaReader(valid_path)

    net = CaptchaNet().to(device)
    if pretrained_checkpoint_name:
        net.load_state_dict(torch.load(f'./results/{pretrained_checkpoint_name}', map_location=device))

    cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_num_workers)

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


if __name__ == '__main__':
    if sys.argv[1] == 'pretrain':
        with open('./configs/pretraining.json') as json_file:
            parameters_dict = json.load(json_file)
        train_net(parameters_dict["train_path"], parameters_dict["valid_path"],
                  parameters_dict["num_epochs"], parameters_dict["batch_size"], parameters_dict["dataloader_num_workers"],
                  parameters_dict["optimizer_lr"], parameters_dict["scheduler_factor"],
                  parameters_dict["scheduler_patience"], parameters_dict["early_stopping_patience"],
                  parameters_dict["checkpoint_name"])

    elif sys.argv[1] == 'finetune':
        with open('./configs/finetuning.json') as json_file:
            parameters_dict = json.load(json_file)
        train_net(parameters_dict["train_path"], parameters_dict["valid_path"],
                  parameters_dict["num_epochs"], parameters_dict["batch_size"], parameters_dict["dataloader_num_workers"],
                  parameters_dict["optimizer_lr"], parameters_dict["scheduler_factor"],
                  parameters_dict["scheduler_patience"], parameters_dict["early_stopping_patience"],
                  parameters_dict["checkpoint_name"], parameters_dict["pretrained_checkpoint_name"])

    else:
        print('Incorrect regime name')
