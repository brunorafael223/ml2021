#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys; sys.path.append(".")
from src.display import image_grid
from src.datasets import UTKFace
from src.trainer import Trainer, DebugModule

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tfm


def set_model_freeze(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not(freeze)


if __name__ == "__main__":


    from argparse import ArgumentParser
    parser = ArgumentParser("Age Estimation Train")
    parser.add_argument("--vis_images", action="store_true")
    parser.add_argument("-b","--batch_size", default=64, type=int)
    parser.add_argument("-j","--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=20, type=int, required=False)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("-o", "--output", default="data/weights")
    args = parser.parse_args()
    

    image_size = (224,224)

    norm = dict(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    

    train_tform = tfm.Compose([
        tfm.Resize(image_size),
        tfm.RandomAffine(degrees=(-10,10)),
        tfm.RandomResizedCrop(image_size,scale=(0.8,1.1),ratio=(1,1)),
        tfm.ColorJitter(
            brightness=(0.8,1.2), contrast=(0.8, 1.), saturation=(0.4,1.2)
        ),
        tfm.ToTensor(),
        tfm.Normalize(**norm)
    ])

    valid_tform = tfm.Compose([
        tfm.Resize(image_size),
        tfm.ToTensor(),
        tfm.Normalize(**norm)
    ])

    train_data = UTKFace(is_train= True, transform=train_tform)
    valid_data = UTKFace(is_train=False, transform=valid_tform)
    
    params = dict(
        batch_size = args.batch_size,  # Number of Samples of the Mini-Batch
        num_workers = args.num_workers # Number of CPU cores used to collect images
    )

    train_loader = DataLoader(train_data, shuffle=True, **params)
    test_loader  = DataLoader(valid_data, **params)
    
    ## Setup Model

    # Prepare Model with one Convolutional Layer
    model = torchvision.models.resnet18(pretrained=True)
    # set_model_freeze(model, freeze=True)
    model.fc = nn.Sequential(
        torch.nn.Linear(512, 1),
    )

    # model = torchvision.models.vgg16(pretrained=True)
    # set_model_freeze(model, freeze=True)
    # model.classifier[-1] = nn.Linear(4096,1)

    ## Setup Device
    model = model.to(args.device)

    ## Setup loss function
    loss_fun = nn.MSELoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=5e-4
    )

    # trainer = Trainer(attribute="age", num_epochs=10, device=device)
    # trainer.fit(model, optimizer, loss_fun, train_loader, test_loader)
    # set_model_freeze(model, False)
    trainer = Trainer(
        img_id="image", 
        attr_id="age", 
        num_epochs=args.max_epochs, 
        device=args.device,
        metrics=["rmse"]
    )
    trainer.fit(model, optimizer, loss_fun, train_loader, test_loader)

    destination = os.path.join(args.output, "resnet18-utk-age.pth")
    print(f"Saving model to: {destination}")
    torch.save(model, destination)
    