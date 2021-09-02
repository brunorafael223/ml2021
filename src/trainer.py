#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from .metrics import Stats


class DebugModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x


class Trainer(object):
    
    def __init__(self, img_id, attr_id, num_epochs, device="cpu", metrics=[]):
        self.num_epochs = num_epochs
        self.device = device
        self.current_epoch = 0
        self.img_id  = img_id
        self.attr_id = attr_id
        self.loss_fn = None
        self.metrics = metrics

    def fit(self, model, optimizer, loss_fn, train_loader, test_loader):
        history = []
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        progress = tqdm(range(self.current_epoch, self.num_epochs),desc="Progress")
        try:
            for epoch in progress:
                self.current_epoch = epoch
                train_stats = self.__train_loop(model, train_loader)
                test_stats  = self.__test_loop(model, test_loader)
                history.append({"epoch":self.current_epoch, 
                    **train_stats,**test_stats})        
                ch = history[-1]
                progress.write(" | ".join([f"{c}: {ch[c]:3.5f}" for c in ch]))
        except KeyboardInterrupt:
            print("Stop training...")
        return history

    def __train_loop(self, model, loader):
        model.train()
        progress = tqdm(loader,leave=False)
        stats = Stats(self.metrics)
        for batch in progress:
            self.optimizer.zero_grad()
            images = batch[self.img_id].to(self.device)
            targets = batch[self.attr_id].to(dtype=torch.float, device=self.device)
            # targets = batch[self.attr_id].to(device=self.device)
            outputs = model(images)
            # print(outputs.shape)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()            
            stats.update(loss, outputs, targets)
            progress.set_description(stats.getline(self.current_epoch, self.num_epochs))
        stats =  stats.get()
        return {f"train_{k}": round(stats[k],5) for k in stats}

    def __test_loop(self, model, loader):
        model.eval()
        stats = Stats(self.metrics)
        progress = tqdm(loader,leave=False)
        for batch in progress:
            with torch.no_grad():
                images = batch[self.img_id].to(self.device)
                # targets = batch[self.attr_id].to(dtype=torch.float, device=self.device)
                targets = batch[self.attr_id].to(device=self.device)
                outputs = model(images)
                loss = self.loss_fn(outputs, targets)
            stats.update(loss, outputs,targets)
            progress.set_description(stats.getline(self.current_epoch, self.num_epochs))
        stats =  stats.get()
        return {f"test_{k}": round(stats[k],5) for k in stats}

