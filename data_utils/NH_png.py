import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
import glob

# Try to import opt from option, handle case where it might be run standalone
try:
    from option import opt
    BS = opt.bs
    crop_size = opt.crop_size if opt.crop else 'whole_img'
except ImportError:
    BS = 1
    crop_size = 'whole_img'

class NH_PNG_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size):
        super(NH_PNG_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.path = path
        self.hazy_files = sorted(glob.glob(os.path.join(path, '*_hazy.png')))
        print(f"Loaded {len(self.hazy_files)} images from {path}")

    def __getitem__(self, index):
        hazy_path = self.hazy_files[index]
        gt_path = hazy_path.replace('_hazy.png', '_GT.png')
        
        haze = Image.open(hazy_path).convert('RGB')
        clear = Image.open(gt_path).convert('RGB')
        
        # Ensure they are the same size (they should be, but just in case)
        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze, clear)
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90*rand_rot)
                target = FF.rotate(target, 90*rand_rot)

        data = tfs.ToTensor()(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.hazy_files)

def get_nh_png_loaders(root_path, bs=BS):
    # For NH-HAZE, it's a small dataset, often split manually. 
    # If the user has a single directory, we can treat it as test.
    test_dataset = NH_PNG_Dataset(root_path, train=False, size='whole_img')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    return None, test_loader

if __name__ == "__main__":
    # Test the loader
    root = '/home/23uec549/btp/NH-HAZE/'
    _, loader = get_nh_png_loaders(root)
    for hazy, gt in loader:
        print(hazy.shape, gt.shape)
        break
