#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from itertools import groupby
import os
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import RandomCrop
import numpy as np

class UTKFace(Dataset):

    def __init__(self, root="../datasets/UTKFace/", is_train=False,
        transform=None):
        # File format: [age]_[gender]_[race]_[date&time].jpg
        # [age] is an integer from 0 to 116, indicating the age
        # [race] is an integer from 0 to 4, denoting White, Black, 
        #   Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
        # [date&time] is in the format of yyyymmddHHMMSSFFF, 
        #   showing the date and time an image was collected to UTKFace

 
        # select dataset split
        n_split = "Train" if is_train else "Test"
        f_split = 'UTK-age-'+n_split+'-filenames.txt'
        # f_split = os.path.join(root, 'UTK-'+n_split+'-filenames.txt')
        
        relp = "/../../data/UTKFace-Split/"
        cwd = os.path.abspath(os.path.dirname(__file__)+relp)
        if not os.path.exists(f_split):
            f_split = os.path.join(cwd, f_split)
        
        # read file_list
        with open(f_split,mode="r") as fid:
            lines = fid.read().splitlines()
     
        cols = ["age","gender","race", "datetime"]
        df = pd.DataFrame(lines, columns=["filenames"])
        df[cols] = df["filenames"].str.split("_", expand=True)
        
        # exclude images with incomplete data 
        df = df.loc[~df["datetime"].isna()]
        df["datetime"] = df["datetime"].str.split(".",n=1, expand=True)[0]

        df["age"] = df["age"].apply(float)
        df["gender"] = df["gender"].apply(int)
        df["race"] = df["race"].apply(int)

        df["path"] = root+df["filenames"]
        assert df["path"].apply(lambda x: os.path.exists(x)).all()

        #print(df["race"].unique())

        self.df = df
        self.race_names = ["white","black","asian", "indian", "others"]
        self.gender_names = ["male", "female"]
        self.transform = transform
            
    def decode_item(self, x):
        print("Age: {} Race: {} Gender: {}".format(
            x["age"], self.race_names[x["race"]], self.gender_names[x["gender"]]
        ))
    
    def __getitem__(self, index):
        data = self.df.iloc[index]
        image = Image.open(data["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, 
                "gender": data["gender"],
                "age": np.array([data["age"]]),
                "race": data["race"]
                }
    
    def __len__(self):
        return len(self.df)

    def show(self, index):
        x = self.__getitem__(index)
        print(self.decode_item(x))
        return x["image"]

## Test functions

def plot_hist(train_data, valid_data):
    train_cnts = train_data.value_counts().sort_index()
    valid_cnts = valid_data.value_counts().sort_index()

    age_counts = pd.concat([train_cnts, valid_cnts],axis=1)
    age_counts.columns = ["train","valid"]
    age_counts.fillna(0, inplace=True)

    fig, ax = plt.subplots()
    ax.step(age_counts.index, age_counts.train, label="train")
    ax.step(age_counts.index, age_counts.valid, label="valid")
    plt.grid()
    plt.title("age samples per split")
    plt.legend()
    plt.show()


def check_data_distribution(root):
    train_data = UTKFace(root, is_train=True).df
    valid_data = UTKFace(root, is_train=False).df
    plot_hist(train_data.age, valid_data.age)


def create_balanced_split(root, train_perc=0.5, verbose=False):
    train_data = UTKFace(root, is_train=True).df
    valid_data = UTKFace(root, is_train=False).df
    all_data = pd.concat([train_data,valid_data],axis=0)
    
    train_samples = []
    valid_samples = []
    # group images by age
    for age, rows in all_data.groupby("age"):
        N = int(round(len(rows)*train_perc,0))
        rows = rows.sample(frac=1) # random shuffle age samples
        train_samples.append(rows.iloc[:N])
        valid_samples.append(rows.iloc[N:])

    train_data = pd.concat(train_samples)
    valid_data = pd.concat(valid_samples)
    
    if verbose:
        plot_hist(train_data.age, valid_data.age)

    train_data.filenames.to_csv("UTK-age-Train-filenames.txt",header=False, index=False)
    valid_data.filenames.to_csv("UTK-age-Test-filenames.txt",header=False, index=False)

  
def test_sample_augmentation(root, index=100, aug_prof=1):
    image_size = (112,112)
    tform = tfm.Compose([
        [tfm.RandomAffine(degrees=(-10,10)),
        tfm.RandomResizedCrop(image_size,scale=(0.4,1.1),ratio=(1,1)),
        tfm.ColorJitter(
            brightness=(0.8,1.2), contrast=(0.8, 1.), saturation=(0.4,1.2)
        )],
        [tfm.RandomAffine(degrees=(-10,10), translate=(0.1,0.1)),
        tfm.RandomResizedCrop(image_size,scale=(0.8,1.1),ratio=(1,1)),
        tfm.ColorJitter(
            brightness=(0.8,1.2), contrast=(0.8, 1.), saturation=(0.4,1.2)
        )
        ]
    ][aug_prof])

    dataset = UTKFace(root, is_train=True, transform=tform)
    plot_images2([dataset[index]["image"] for i in range(64)], 4)
    plt.show()    


def test_dataloader(root):
    tform = tfm.Compose([tfm.ToTensor()])
    dataset = UTKFace(root, is_train=True, transform=tform)  
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
    for d in dataloader:
        print(d["age"])
        break


if __name__ == "__main__":

    import sys; sys.path.append(".")
    from torch.utils.data import DataLoader
    from torchvision import transforms as tfm
    from transforms import plot_images2
    root = "../../datasets/UTKFace/"
    
    # create_balanced_split(root)
    # check_data_distribution(root)
    test_sample_augmentation(root, index=0)
    # test_dataloader(root)


    
   