import os
from typing import Tuple

import torch
import torch.utils.data as data
import torchvision.transforms
from torch import Tensor
from Config import *

import numpy as np
import pandas as pd

import numpy as np 
import matplotlib.pyplot as plt

def FFT2(x):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))

def iFFT2(k):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k)))

def truncate_PE(kspace):

    n_lines = len(kspace)

    AccelerationFactors = np.linspace(start=2, stop=4, num=8)
    AccelerationFactor  = np.random.choice(AccelerationFactors)
    DeleteLines = n_lines * (1 - 1/AccelerationFactor)
    DeleteLines = int(DeleteLines/2)

    kspace_pad = np.zeros_like(kspace) + 1e-12
    
    start, stop = DeleteLines, n_lines - DeleteLines
    
    kspace_pad[start:stop] = kspace[start:stop]

    return kspace_pad, DeleteLines

def synthesize(image):

    kspace = FFT2(image)
    kspace_truncated, delete_line = truncate_PE(kspace)

    image_truncated = iFFT2(kspace_truncated)

    return image_truncated

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataroot: str) -> None:
        super(CustomDataset, self).__init__()

        self.lr_dir_path = os.path.join(dataroot, "Dirty_mag")
        self.hr_dir_path = os.path.join(dataroot, "Clear_mag")
        self.batch_size  = batch_size

        if os.path.exists(os.path.join(dataroot, "filenames.csv")):
            print("Loading filenames from csv")
            self.df = pd.read_csv(os.path.join(dataroot, "filenames.csv"))
            self.df['MatrixSize'] = self.df.lr.apply(os.path.basename).str.split("_").str[1:3].str.join("_")

            
            self.lr_filenames = self.df['lr'].tolist()
            self.hr_filenames = self.df['hr'].tolist()
        else:
            print("Generating filenames")
            self.lr_filenames = []
            self.hr_filenames = []
            for subjectnames in os.listdir(self.lr_dir_path):

                LR = os.path.join(self.lr_dir_path, subjectnames)
                HR = os.path.join(self.hr_dir_path, subjectnames)
                FileList = os.listdir(LR)
                for filename in FileList:
                    self.lr_filenames.append(os.path.join(LR, filename))
                    self.hr_filenames.append(os.path.join(HR, filename))

            self.df = pd.DataFrame({'lr': self.lr_filenames, 'hr': self.hr_filenames})
            self.df.to_csv(os.path.join(dataroot, "filenames.csv"), index=False)
            self.df['MatrixSize'] = self.df.lr.apply(os.path.basename).str.split("_").str[1:3].str.join("_")

    def crop(img, i, j, h, w):
        return img[..., i : i + h, j : j + w]

    def transform(self, lr_image, hr_image, resample):

        lr_tensor = torchvision.transforms.functional.to_pil_image(lr_image)

        self.crop_indices = torchvision.transforms.RandomCrop.get_params((lr_tensor), output_size=(image_size, image_size))

        i, j, h, w =  self.crop_indices


        lr_image = torchvision.transforms.functional.crop(lr_image, i, j, h, w)
        hr_image = torchvision.transforms.functional.crop(hr_image, i, j, h, w)

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            lr_image = torchvision.transforms.functional.hflip(lr_image)
            hr_image = torchvision.transforms.functional.hflip(hr_image)

        # Random vertical flipping
        if torch.rand(1) > 0.5:
            lr_image = torchvision.transforms.functional.vflip(lr_image)
            hr_image = torchvision.transforms.functional.vflip(hr_image)

        return lr_image, hr_image

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        #lr = np.load(self.lr_filenames[index])
        hr = np.load(self.hr_filenames[index])
        lr = synthesize(hr)
        
        if torch.rand(1) > 0.5:
            lr = np.tanh(lr.real)
            hr = np.tanh(hr.real)
        else:
            lr = np.tanh(lr.imag)
            hr = np.tanh(hr.imag)

        lr = torch.from_numpy(np.array(lr, np.float32, copy=False))
        hr = torch.from_numpy(np.array(hr, np.float32, copy=False))

        lrts, hrts = self.transform(lr, hr, True)


        return lrts.unsqueeze(0), hrts.unsqueeze(0)


    def __len__(self) -> int:
        return len(self.lr_filenames)