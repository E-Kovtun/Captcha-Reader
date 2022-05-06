import os
import random
import shutil
from tqdm import tqdm
import numpy as np


def split_data(init_data_folder, train_size=550, valid_size=120, test_size=400):
    images = os.listdir(init_data_folder)

    encountered_symbols = list(''.join([image_name.split('.')[0] for image_name in images]))
    all_symbols = np.unique(encountered_symbols)
    with open('./symbols.txt', 'w') as f:
        for s in all_symbols:
            f.write(s)
            f.write('\n')

    assert (train_size + valid_size + test_size) == len(images)
    random.Random(1).shuffle(images)
    train = images[:train_size]
    valid = images[train_size:train_size+valid_size]
    test = images[-test_size:]

    prepared_data_folder = './prepared_datasets/'
    os.makedirs(os.path.join(prepared_data_folder, 'train_dataset'), exist_ok=True)
    os.makedirs(os.path.join(prepared_data_folder, 'valid_dataset'), exist_ok=True)
    os.makedirs(os.path.join(prepared_data_folder, 'test_dataset'), exist_ok=True)

    for dataset_images, dataset_folder in zip([train, valid, test], ['train_dataset', 'valid_dataset', 'test_dataset']):
        for img in tqdm(dataset_images):
            shutil.copyfile(os.path.join(init_data_folder, img), os.path.join(prepared_data_folder, dataset_folder, img))


if __name__ == '__main__':
    init_data_folder = '../dataset/'
    split_data(init_data_folder)
