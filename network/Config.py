import os
import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from Models import *
import sys

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(8888)
cudnn.benchmark = True
cudnn.deterministic = False

use_gpu=True

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda:0")

mode = "train"


isPreTrained=False


# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    #sys.path.append("/mnt/alp/users/siyeopyoon/")
    print(os.getcwd())



    # 1. Dataset path.
    #dataroot = "/mnt/alp/users/siyeopyoon/F02Data/1_SR/Pair0930_AfterIQA_SameSize"
    #dataroot = "/mnt/alp/users/siyeopyoon/F02Data/1_REGAIN_Cine/Training_full"
    coderoot = '/mnt/alp/Users/Manuel/F01Code/Pythoncodes/1_Flow/Training_old/001_cine_model_7'
    dataroot = "/data1/manuel/0_REGAIN_Cine/Training_full"
    
    image_size=image_In

#    image_size = 16  #"Image size of high resolution image."
    upscale_factor=scale_factor
    batch_size = 64

    # 2. Define model.
    generator = GeneratorRRDB(1,filters=64, num_res_blocks=23).to(device)
    discriminator = Discriminator(input_shape=(1, image_size, image_size)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # 3. Resume training.
    seed=123
    start_p_epoch = 35
    start_g_epoch = 35
    resume = False
    resume_p_weight = ""
    resume_d_weight = "D_Trasfer_epoch34.pth"
    resume_g_weight = "G_Trasfer_epoch34.pth"
    discriminator.load_state_dict(torch.load("./samples/"+resume_d_weight, map_location=device))
    generator.load_state_dict(torch.load("./samples/"+resume_g_weight, map_location=device))


    # 4. Number of epochs.
    g_epochs = 40000

    warmup_batches=500

    # 5. Loss function.
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
    criterion_content = torch.nn.MSELoss().to(device)
    criterion_FFT = torch.nn.L1Loss().to(device)
    criterion_pixel = torch.nn.MSELoss().to(device)


    # Loss function weight.
    lambda_adv = 5e-03
    lambda_pixel = 1e-02
    lambda_fft = 1e-02

    # 6. Optimizer.
    p_lr = 1e-4
    d_lr = 1e-4
    g_lr = 1e-4

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

   
    # 8. Training log.
    os.makedirs(os.path.join(coderoot, "samples"), exist_ok=True)
#nohup python /mnt/alp/Users/Manuel/F01Code/Pythoncodes/1_Flow/Training_old/001_cine_model_7/Training.py