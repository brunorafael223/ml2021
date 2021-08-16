#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def image_grid(images, rows=2):
    c = len(images)//rows
    r = rows + (len(images)-(c*rows))
    
    fig = plt.figure(figsize=(8., 4.))
    grid = ImageGrid(fig, 111,  nrows_ncols=(r, c))
    for i, ax in enumerate(grid):
        if i > len(images)-1:
            break
        ax.imshow(images[i])
        ax.axis("off")
    plt.show()