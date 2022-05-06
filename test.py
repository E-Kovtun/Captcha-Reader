import torch
from torch import nn
from model import CaptchaNet
import os
from reader import CaptchaReader
from torch.utils.data import DataLoader
from utils.metrics import all_symbols, captcha_len, num_classes, captcha_accuracy
from tqdm import tqdm
import sys
import json
import cv2 as cv
from torchvision.transforms import ToTensor


def test_net(data_path, checkpoint_path):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    net = CaptchaNet().to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.train(False)
    print('Testing...')

    if os.path.isdir(data_path):
        test_dataset = CaptchaReader(data_path)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
        all_predictions = []
        all_gt = []
        for batch_images, batch_targets in tqdm(test_dataloader):
            batch_images, batch_target = batch_images.to(device), batch_targets.to(device)
            output = net(batch_images)
            predicted_classes = torch.argmax(output, dim=2)
            all_predictions.extend(predicted_classes.detach().cpu().tolist())
            all_gt.extend(batch_targets.detach().cpu().tolist())
        test_acc = captcha_accuracy(all_predictions, all_gt)
        print(f'Test Accuracy {test_acc}')
        os.makedirs('./results/', exist_ok=True)
        with open('results/test_metric.json', 'w', encoding='utf-8') as f:
            json.dump({'test_accuracy': test_acc}, f)

    elif os.path.isfile(data_path):
        image = cv.imread(data_path)
        tensor_image = ToTensor()(image).unsqueeze(0).to(device)
        net_output = net(tensor_image).argmax(dim=2).squeeze(0).detach().cpu().numpy()
        predicted_symbols = [all_symbols[n] for n in  net_output]
        print('Predicted Symbols:', predicted_symbols)
        os.makedirs('./results/', exist_ok=True)
        with open('results/predicted_symbols.json', 'w', encoding='utf-8') as f:
            json.dump({'net_predicted_symbols': predicted_symbols}, f)

    else:
        print('Incorrect data path')


if __name__ == '__main__':
    data_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    test_net(data_path, checkpoint_path)

