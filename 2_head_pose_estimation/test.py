#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys; sys.path.append(".")
from src.display import image_grid
from src.datasets import LFW
from src.datasets.lfw import Draw, Image
from src.trainer import Trainer, DebugModule

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tfm
from torchvision.transforms.transforms import RandomAffine, Resize
from tqdm import tqdm


if __name__ == "__main__":


    from argparse import ArgumentParser
    parser = ArgumentParser("Head pose Estimation Test")
    parser.add_argument("--vis_images", action="store_true")
    parser.add_argument("-b","--batch_size", default=32, type=int)
    parser.add_argument("-j","--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("-c", "--ckpt", default="data/weights/resnet18-lfw-pose.pth")
    args = parser.parse_args()


    image_size = (224,224)

    norm = dict(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    unorm = dict(
        mean= torch.tensor(norm["mean"]).view(1,3,1,1),
        std= torch.tensor(norm["std"]).view(1,3,1,1)
    )
    valid_tform = tfm.Compose([
        tfm.Resize(image_size),
        tfm.ToTensor(),
        # tfm.Normalize(**norm)
    ])

    params = dict(batch_size = args.batch_size, num_workers = args.num_workers)
    valid_data   = LFW(is_train=False, transform=valid_tform)
    valid_loader = DataLoader(valid_data, **params, shuffle=False)
    
    ## Load Model
    model = torch.load(args.ckpt).to(args.device)    
    for data in tqdm(valid_loader):
        with torch.no_grad():
            images = data["image"].to(args.device)
            target = data["pose"].to(args.device)
            outputs = model(images).cpu()

            # images = (images.cpu()*unorm["std"]+unorm["mean"]).permute(0,2,3,1)*255
            images = images.cpu().permute(0,2,3,1)*255
            images2 = []
            for i in range(images.size(0)):
                img  = Image.fromarray(images[i].numpy().astype(np.uint8))
                pin  = target[i].cpu()
                pout = outputs[i]
                proj = valid_data.getPoseProjection(pout, data["T"][i])
                img = Draw.pose_box(img, proj)
                img = Draw.pose_axis(img, proj)
                images2.append(img)
            image_grid(images2,rows=4)
