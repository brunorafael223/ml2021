#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch


class Accuracy(object):
    
    def __init__(self):
        self.tp = 0
        self.size = 0

    def update(self, X, y):
        self.size += X.size(0)
        x = X.argmax(dim=1) # select indices were score was max
        self.tp += torch.sum((x == y))  # sum true positives
        
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def compute(self):
        return float(self.tp/float(self.size))
    
    def reset(self):
        self.tp = 0
        self.size = 0


class RegressionMetric(object):

    def __init__(self):
        super().__init__()
        self.err  = 0
        self.size = 0
    
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def reset(self):
        self.err = 0
        self.size = 0


class MAE(RegressionMetric):
    
    def update(self, x, y):
        self.size += y.size(0)*y.size(1)
        self.err  += torch.abs(y-x).sum()

    def compute(self):
        return float(self.err/self.size)


class MSE(RegressionMetric):

    def update(self, x, y):
        self.size += y.size(0)*y.size(1)
        self.err  += ((y-x)**2).sum()

    def compute(self):
        return float(self.err/self.size)


class RMSE(RegressionMetric):
    
    def update(self, x, y):
        self.size += y.size(0)*y.size(1)
        self.err  += ((y-x)**2).sum()

    def compute(self):
        return np.sqrt(float(self.err/self.size))


def getMetricbyName(name):
    return {"accuracy":Accuracy(),
     "mae": MAE(),
     "mse": MSE(),
     "rmse": RMSE()
    }[name]


class Stats(object):

    def __init__(self, metrics=[]):
        super().__init__()
        self.loss = 0.0
        self.size = 0.0
        self.metrics = {m: getMetricbyName(m) for m in metrics}

    def update(self, loss, x, y):
        # detach variables from graph and move to cpu
        loss = loss.detach().cpu()
        x = x.detach().cpu()
        y = y.detach().cpu()
        # update running loss
        self.size += y.size(0)
        self.loss += loss * y.size(0)
        # update Metrics
        for m in self.metrics:
            self.metrics[m](x, y)

    def getline(self, current_epoch, num_epochs):
        lines = []
        for m in self.metrics:
            value = self.metrics[m].compute()
            lines.append("{} = {:.3f}".format(m, value))
        
        return " | ".join([
            "Epoch {:03d}/{:03d}".format(current_epoch, num_epochs),
            "loss = {:.3f}".format(self.loss/self.size),
            *lines
        ])

    def get(self):
        return {"loss": float(self.loss/self.size),
            **{m:self.metrics[m].compute() for m in self.metrics}
        }


