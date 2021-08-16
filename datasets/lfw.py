#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from PIL import Image, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from ipywidgets import interact
import cv2
import os
from torch.utils.data.dataset import Dataset


def getCube(cube_type="", cubeScale=6e4):
    cube = np.array([
        [ 1,-1, 1], #A [0]
        [ 1, 1, 1], #B [1]
        [ 1, 1,-1], #C [2]
        [ 1,-1,-1], #D [3]
        [-1,-1, 1], #E [4]
        [-1, 1, 1], #F [5]
        [-1, 1,-1], #G [6]
        [-1,-1,-1], #H [7]
    ])*cubeScale
    # 3D 'Frontal' Plane
    if cube_type == "front":
        return cube[[0,1,5,4,0],:].T
    else:
        return cube[[0,3,2,1,5,6,2,3,7,6,7,4],:].T


def xyhw2xyxy(rect):
    return [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]


class Draw(object):

    @staticmethod
    def landmarks(image, landmarks, r=1):
        draw = ImageDraw.ImageDraw(image)
        # draw.point(list(map(tuple,landmarks)),'red')
        rb = r+1
        for (x,y) in landmarks:
            draw.ellipse((x-rb, y-rb, x+rb, y+rb), fill=(0,0,0,0))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,0))
        return image

    @staticmethod
    def pose_axis(image, project):
        axis = project(np.eye(3, 4, k=1) * 5e4)
        axis = axis.round(0).astype(int)
        image = np.array(image, np.uint8)
        for i, color in enumerate(np.eye(3)*255, 1):
            pt1 = tuple(axis[0,:])
            pt2 = tuple(axis[i,:])
            cv2.arrowedLine(image,pt1,pt2, color=tuple(color), thickness=4)
        return Image.fromarray(image)

    @staticmethod
    def pose_box(image, project):

        fmt_pts = lambda x: list(map(tuple,x))

        # Set 3D point (origin+axes)
        sp = project(getCube("")) 
        image_1 = image.copy()
        draw = ImageDraw.Draw(image_1)
        draw.polygon(fmt_pts(sp), outline="white")
        image = Image.blend(image, image_1, alpha=.3)

        # Set 3D point (origin+axes)
        sp = project(getCube("front")) 
        draw = ImageDraw.Draw(image)
        draw.polygon(fmt_pts(sp), outline="red")

        return image

    @staticmethod
    def detection(image, rect):
        x0, y0, x1, y1 = xyhw2xyxy(rect)
        draw = ImageDraw.ImageDraw(image)
        draw.rectangle([y0,x0,y1,x1],outline='green', width=2)
        return image


class LFW:

    def __init__(self, root = "../datasets/LFW", is_train=True):

        # select file split
        n_split = "Train" if is_train else "Test"
        f_split = os.path.join(root, 'LFW-'+n_split+'-filenames.txt')
        images = pd.read_csv(f_split, header=None, names=["filename"])
        images["path"] = images["filename"].apply(lambda x: os.path.join(root,"All", x))
        assert images["path"].apply(lambda x: os.path.exists(x)).all()
        
        # load parameters
        load_par = dict(sep="  ",header=None, engine="python")

        # load landmarks and convert to [Nimg, Npts,[x,y]] format
        filepath = os.path.join(root, f"LFW-Landmarks.txt")
        assert os.path.exists(filepath), filepath
        landmarks = pd.read_csv(filepath, **load_par).transpose()

        cols, v = landmarks.columns, len(landmarks.columns)//2
        x = landmarks[cols[:v]].values/2
        y = landmarks[cols[v:]].values/2
        landmarks = np.stack([x,y], axis=1).transpose(0,2,1)

        # load poses
        filepath = os.path.join(root, f"LFW-Poses.txt")
        assert os.path.exists(filepath), filepath
        poses = pd.read_csv(filepath, **load_par, 
            names=["alpha","beta","gamma","tx","ty","tz"])
        # Fix Gamma angle (Rx) discontinuity around -pi
        poses[poses["gamma"] < 0] = poses[poses["gamma"] < 0] + 2*np.pi
        poses = poses.values
        #display(poses)

        # load rectangles
        filepath = os.path.join(root, f"LFW-Rectangles.txt")
        assert os.path.exists(filepath), filepath
        rectangles = pd.read_csv(filepath, **load_par, names=["x","y","h","w"])
        rectangles = rectangles[["x","y","h","w"]].values/2

        size_train = 10586
        srange = slice(0,size_train,1) if is_train else slice(size_train,None,1)
        #display(bboxes)
        self.images = images
        self.landmarks = landmarks[srange]
        self.rectangles = rectangles[srange]
        self.poses = poses[srange]
    
    def getPoseProjection(self, angles, T):
        f = 1000
        a,b,g = angles[:3]    
        # Compute 3D Rotation Matrix
        Rx = np.array([[1,0,0],[0,cos(g),-sin(g)],[0,sin(g),cos(g)]])
        Ry = np.array([[cos(b),0,sin(b)],[0,1,0],[sin(b),0,cos(b)]])
        Rz = np.array([[cos(a),-sin(a),0],[sin(a), cos(a),0],[0,0,1]])
        R  = Rz @ Ry @ Rx
        T  = T.reshape(3,1)
        def project(box): # Perspective Projection function
            box = np.vstack([box, np.ones((1, box.shape[1]))])
            S = (np.eye(3)*[f,f,1] @ np.hstack([R,T]) @ box).T
            return S[:,:-1]/S[:,-1].reshape(-1,1)
        return project
    
    def show(self, index, show_landmarks=False, show_detection=False, 
        show_pose_box=False, show_pose_axis=False):
        image = Image.open(self.images["path"].iloc[index]).convert("RGB")
        pose = self.poses[index]
        land = self.landmarks[index]
        rect = self.rectangles[index]
        proj = self.getPoseProjection(angles=pose[:3], T=pose[3:])
        
        if show_detection:
            image = Draw.detection(image, rect)
        if show_landmarks:
            image = Draw.landmarks(image, land)   
        if show_pose_box:
            image = Draw.pose_box(image, proj)
        if show_pose_axis:
            image = Draw.pose_axis(image, proj)

        return image

    def __getitem__(self, index):
        image = Image.open(self.images["path"].iloc[index]).convert("RGB")
        pose = self.poses[index]
        land = self.landmarks[index]
        rect = self.rectangles[index]
        return {"image":image, "pose": pose[:3], "rect":rect, "landmarks":land}

    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    
    dataset = LFW()