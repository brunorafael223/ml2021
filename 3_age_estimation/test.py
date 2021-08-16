#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys; sys.path.append(".")
from src.display import image_grid
from src.datasets import UTKFace
from src.metrics import getMetricbyName

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tfm
from tqdm import tqdm
from torch import nn
import torchvision
import numpy as np


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser("Age Estimation Train")
    parser.add_argument("--vis_images", action="store_true")
    parser.add_argument("-b","--batch_size", default=64, type=int)
    parser.add_argument("-j","--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("-c", "--ckpt", default="data/weights/resnet18-utk-age.pth")
    args = parser.parse_args()

    image_size = (224,224)
    norm = dict(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    
    unorm = dict(
        mean= torch.tensor(norm["mean"]).view(1,3,1,1),
        std= torch.tensor(norm["std"]).view(1,3,1,1)
    )
    valid_tform = tfm.Compose([
        tfm.Resize(image_size), tfm.ToTensor(), tfm.Normalize(**norm)
    ])

    valid_data   = UTKFace(is_train=False, transform=valid_tform)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)

    model = torch.load(args.ckpt).to(args.device)
    model.eval()

    metrics = []
    for data in tqdm(valid_loader):
        with torch.no_grad():
            images = data["image"].to(args.device)
            target = data["age"].to(args.device)
            outputs = model(images)

            images = (images.cpu()*unorm["std"]+unorm["mean"]).permute(0,2,3,1)
                
            print(outputs)
            image_grid(images,rows=8)



