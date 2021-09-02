#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from torch.nn.modules.activation import Hardtanh; sys.path.append(".")
from src.display import image_grid
from src.datasets import LFW
from src.trainer import Trainer, DebugModule

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tfm
from torchvision.transforms.transforms import RandomAffine, RandomCrop, Resize


def set_model_freeze(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not(freeze)


if __name__ == "__main__":


    from argparse import ArgumentParser
    parser = ArgumentParser("Head Pose Estimation Train")
    parser.add_argument("--vis_images", action="store_true")
    parser.add_argument("-b","--batch_size", default=64, type=int)
    parser.add_argument("-j","--num_workers", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--max_epochs", default=100, type=int, required=False)
    parser.add_argument("--device", default="cuda", type=str, required=False)
    parser.add_argument("-o", "--output", default="data/weights")
    args = parser.parse_args()
    

    Isize = (224,224)
    norm = dict(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    train_tform = tfm.Compose([
        tfm.Resize(Isize),
        # tfm.RandomCrop(Isize,pad_if_needed=True),
        tfm.RandomAffine(degrees=0, scale=(0.8,1.2), translate=(0.1,0.2)),
        # tfm.RandomResizedCrop(image_size,scale=(0.8,1.1),ratio=(1,1)),
        tfm.ColorJitter(
            brightness=(0.8,1.2), contrast=(0.8, 1.), saturation=(0.4,1.2)
        ),
        tfm.ToTensor(),
        # tfm.Normalize(**norm)
    ])

    valid_tform = tfm.Compose([
        tfm.Resize(Isize),
        tfm.ToTensor(),
        # tfm.Normalize(**norm)
    ])

    params = dict(batch_size = args.batch_size, num_workers = args.num_workers)
    train_data = LFW(is_train= True, transform=train_tform)
    valid_data = LFW(is_train=False, transform=valid_tform)
    train_loader = DataLoader(train_data, shuffle=True, **params)
    test_loader  = DataLoader(valid_data, **params)
    
    ## Setup Model

    # Prepare Model with one Convolutional Layer
    
    # set_model_freeze(model, freeze=True)
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        # nn.Linear(512, 512),
        # nn.BatchNorm1d(512),
        # nn.Sigmoid(),
        # nn.Dropout(0.5),
        # nn.Linear(512, 256),
        # nn.BatchNorm1d(256),
        # nn.Sigmoid(),
        # nn.Linear(512, 512, bias=False),
        # nn.Tanh(),
        nn.Linear(512, 3, bias=True),
    )

    # model = torchvision.models.vgg16(pretrained=True)
    # model.classifier[-1] =  torch.nn.Linear(4096, 3)

    # model = torchvision.models.resnet50(pretrained=True)
    # model.fc = nn.Sequential(torch.nn.Linear(2048, 3))

    ## Setup Device
    model = model.to(args.device)

    class HalfMSELoss(nn.Module):
        def __init__(self):
            super().__init__()
            pass
        def forward(self, x,y):
            squared = (y-x)**2
            # print(squared)
            loss = 0
            for i in range(squared.size(1)):
                loss += torch.sum(squared[:,i]/squared.size(0))/2
            return loss
        
    ## Setup Train
    loss_fun = HalfMSELoss()#nn.MSELoss(reduction="sum")    
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
    #     momentum=0.4,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    trainer = Trainer(
        img_id="image", 
        attr_id="pose", 
        num_epochs=args.max_epochs, 
        device=args.device,
        metrics=[]
    )
    trainer.fit(model, optimizer, loss_fun, train_loader, test_loader)
    destination = os.path.join(args.output, "resnet18-lfw-pose.pth")
    print(f"Saving model to: {destination}")
    torch.save(model, destination)
    
    