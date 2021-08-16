#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from tqdm import tqdm
import sys; sys.path.append(".")
from src.display import image_grid
from src.trainer import Trainer, DebugModule


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser("MNIST Demo")
    parser.add_argument("--vis_images", action="store_true")
    parser.add_argument("-b","--batch_size", default=64, type=int)
    parser.add_argument("-j","--num_workers", default=2, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=10, type=int, required=False)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    args = parser.parse_args()
    
    ## Load MNIST Dataset (train/test)
    train_data = MNIST('data', train=True, download=True)
    valid_data = MNIST('data', train=False)
    
    ## Plot samples: creates 8x8 image grid
    if args.vis_images:
        image_grid( [train_data[i][0] for i in range(64)] ,8)
       
    ## Setup transformations
    # convert PIL image format to Tensor 
    train_data.transform = ToTensor()
    valid_data.transform = ToTensor()

    ## DataLoader: Creates Mini-Batches of images and labels
    
    params = dict(
        batch_size  = args.batch_size, # Number of Samples of the Mini-Batch
        num_workers = args.num_workers # Number of CPU cores used to collect images
    )

    # train samples are shuffled
    train_loader = DataLoader(train_data, shuffle=True, **params)
    valid_loader = DataLoader(valid_data, **params)

    ## Setup Model
    kernel_size = (5, 5)
    image_size  = (28, 28)
    output_size = (
        image_size[0]-2*(kernel_size[0]//2), # integer division by 2
        image_size[1]-2*(kernel_size[1]//2)
    )
    print("Image size after Conv2d without padding: ", output_size)
    
    # Prepare Model with one Convolutional Layer
    model = nn.Sequential(
        nn.Conv2d(
            in_channels=1, 
            out_channels=20, 
            kernel_size=kernel_size, 
            stride=(1,1), 
            padding=kernel_size[0]//2
            ), 
        nn.BatchNorm2d(20),
        nn.ReLU(inplace=False),
        nn.Flatten(start_dim=1),
        nn.Linear(28*28*20, 10), # 
    )
    
    ## Setup Device
    model = model.to(args.device)

    ## Setup Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.9, 
        weight_decay=5e-4
    )

    ## Setup loss function
    loss_fun = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        img_id=0, 
        attr_id=1, 
        num_epochs=10, 
        device=args.device,
        metrics=["accuracy"]
    )

    trainer.fit(
        model=model, 
        optimizer=optimizer, 
        loss_fn=loss_fun, 
        train_loader=train_loader, 
        test_loader=valid_loader
        )
    