import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import cv2 as cv
from utils.metrics import all_symbols


class CaptchaReader(Dataset):
    def __init__(self, prepared_data_folder, dataset_folder):
        self.dataset_path = os.path.join(prepared_data_folder, dataset_folder)
        self.dataset_images = os.listdir(self.dataset_path)

    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, index):
        image = cv.imread(os.path.join(self.dataset_path, self.dataset_images[index]))
        tensor_image = ToTensor()(image)
        gt_symbols = list(self.dataset_images[index].split('.')[0])
        target = torch.tensor([all_symbols.index(s) for s in gt_symbols], dtype=torch.int64)
        return tensor_image, target
