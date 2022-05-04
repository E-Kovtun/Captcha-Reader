import numpy as np

all_symbols = []
with open('./symbols.txt') as f:
    for line in f:
        all_symbols.append(line.split('\n')[0])

captcha_len = 5
num_classes = len(all_symbols)
img_h = 50
img_w = 200


def captcha_accuracy(preds, gt):
    concurrencies = [int(preds[b] == gt[b]) for b in range(len(preds))]
    captcha_acc = np.sum(concurrencies) / len(concurrencies)
    return captcha_acc
